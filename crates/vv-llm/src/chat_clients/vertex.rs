use crate::{ChatRequest, ChatResponse, VvLlmError};
use async_trait::async_trait;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;

use super::{ChatClient, ChatStream, OpenAiCompatibleChatClient};

const TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const REFRESH_THRESHOLD_SECONDS: f64 = 300.0;

#[derive(Debug, Clone, PartialEq)]
pub struct GoogleAccessToken {
    pub token: String,
    pub expires_at: f64,
}

#[derive(Debug, Clone)]
pub struct GoogleAccessTokenProvider {
    credentials: Value,
    cached_token: Arc<Mutex<Option<GoogleAccessToken>>>,
}

impl GoogleAccessTokenProvider {
    pub fn new(credentials: Value) -> Result<Self, VvLlmError> {
        if !credentials.is_object() {
            return Err(VvLlmError::Configuration(
                "Vertex endpoint requires credentials object".to_string(),
            ));
        }
        let cached_token = cached_token_from_credentials(&credentials);
        Ok(Self {
            credentials,
            cached_token: Arc::new(Mutex::new(cached_token)),
        })
    }

    pub async fn access_token(&self) -> Result<GoogleAccessToken, VvLlmError> {
        {
            let cached = self.cached_token.lock().await;
            if let Some(token) = cached.as_ref().filter(|token| token.is_fresh()) {
                return Ok(token.clone());
            }
        }

        let token = if credential_value(&self.credentials, "refresh_token").is_some() {
            self.refresh_user_token().await
        } else if credential_value(&self.credentials, "private_key").is_some() {
            self.refresh_service_account_token().await
        } else {
            Err(VvLlmError::Configuration(
                "Vertex credentials require refresh_token or private_key".to_string(),
            ))
        }?;
        *self.cached_token.lock().await = Some(token.clone());
        Ok(token)
    }

    async fn refresh_user_token(&self) -> Result<GoogleAccessToken, VvLlmError> {
        let token_uri = credential_value(&self.credentials, "token_uri")
            .unwrap_or_else(|| TOKEN_URI.to_string());
        let params = [
            (
                "client_id",
                required_credential(&self.credentials, "client_id")?,
            ),
            (
                "client_secret",
                required_credential(&self.credentials, "client_secret")?,
            ),
            (
                "refresh_token",
                required_credential(&self.credentials, "refresh_token")?,
            ),
            ("grant_type", "refresh_token".to_string()),
        ];
        let response = reqwest::Client::new()
            .post(token_uri)
            .form(&params)
            .send()
            .await
            .map_err(|error| VvLlmError::Http(error.to_string()))?;
        parse_token_response(response).await
    }

    async fn refresh_service_account_token(&self) -> Result<GoogleAccessToken, VvLlmError> {
        let token_uri = credential_value(&self.credentials, "token_uri")
            .unwrap_or_else(|| TOKEN_URI.to_string());
        let assertion = service_account_assertion(&self.credentials, now_seconds())?;
        let params = [
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:jwt-bearer".to_string(),
            ),
            ("assertion", assertion),
        ];
        let response = reqwest::Client::new()
            .post(token_uri)
            .form(&params)
            .send()
            .await
            .map_err(|error| VvLlmError::Http(error.to_string()))?;
        parse_token_response(response).await
    }
}

impl GoogleAccessToken {
    fn is_fresh(&self) -> bool {
        now_seconds_f64() < self.expires_at - REFRESH_THRESHOLD_SECONDS
    }
}

fn cached_token_from_credentials(credentials: &Value) -> Option<GoogleAccessToken> {
    let token = credential_value(credentials, "access_token")?;
    let expires_at = credentials
        .get("access_token_expires_at")
        .and_then(Value::as_f64)?;
    let token = GoogleAccessToken { token, expires_at };
    token.is_fresh().then_some(token)
}

#[derive(Debug, Clone)]
pub struct VertexOpenAiChatClient {
    model: String,
    api_base: String,
    token_provider: GoogleAccessTokenProvider,
}

impl VertexOpenAiChatClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        credentials: Value,
    ) -> Result<Self, VvLlmError> {
        Ok(Self {
            model: model.into(),
            api_base: api_base.into(),
            token_provider: GoogleAccessTokenProvider::new(credentials)?,
        })
    }

    async fn inner(&self) -> Result<OpenAiCompatibleChatClient, VvLlmError> {
        let token = self.token_provider.access_token().await?;
        Ok(OpenAiCompatibleChatClient::new(
            self.model.clone(),
            self.api_base.clone(),
            token.token,
        ))
    }
}

#[async_trait]
impl ChatClient for VertexOpenAiChatClient {
    fn provider_name(&self) -> &'static str {
        "openai-vertex"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        self.inner().await?.create_completion(request).await
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        self.inner().await?.create_stream(request).await
    }
}

#[derive(Debug, Serialize)]
struct ServiceAccountClaims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
    scope: &'a str,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default = "default_expires_in")]
    expires_in: u64,
}

async fn parse_token_response(
    response: reqwest::Response,
) -> Result<GoogleAccessToken, VvLlmError> {
    let status = response.status();
    let retry_after =
        crate::utilities::parse_retry_after_headers(response.headers(), SystemTime::now());
    let body = response
        .text()
        .await
        .map_err(|error| VvLlmError::Http(error.to_string()))?;
    if !status.is_success() {
        return Err(VvLlmError::from_status_with_retry_after(
            status.as_u16(),
            format!("Google token endpoint returned {status}: {body}"),
            retry_after,
        ));
    }
    let token: TokenResponse = serde_json::from_str(&body)?;
    Ok(GoogleAccessToken {
        token: token.access_token,
        expires_at: now_seconds_f64() + token.expires_in as f64,
    })
}

fn service_account_assertion(credentials: &Value, now: u64) -> Result<String, VvLlmError> {
    let token_uri =
        credential_value(credentials, "token_uri").unwrap_or_else(|| TOKEN_URI.to_string());
    let email = required_credential(credentials, "client_email")?;
    let private_key = required_credential(credentials, "private_key")?;
    let claims = ServiceAccountClaims {
        iss: &email,
        sub: &email,
        aud: &token_uri,
        iat: now,
        exp: now + 3600,
        scope: CLOUD_PLATFORM_SCOPE,
    };
    jsonwebtoken::encode(
        &Header::new(Algorithm::RS256),
        &claims,
        &EncodingKey::from_rsa_pem(private_key.as_bytes())
            .map_err(|error| VvLlmError::Configuration(error.to_string()))?,
    )
    .map_err(|error| VvLlmError::Configuration(error.to_string()))
}

fn required_credential(credentials: &Value, key: &str) -> Result<String, VvLlmError> {
    credential_value(credentials, key)
        .ok_or_else(|| VvLlmError::Configuration(format!("Vertex credentials missing {key}")))
}

fn credential_value(credentials: &Value, key: &str) -> Option<String> {
    credentials
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn default_expires_in() -> u64 {
    3600
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn now_seconds_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}
