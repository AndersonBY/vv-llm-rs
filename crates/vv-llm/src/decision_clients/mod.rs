//! Independent, provider-neutral decision types and an OpenAI adapter.
use crate::{BackendType, ErrorKind, ModelCapabilities, ResolvedModelConfig, VvLlmError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashSet, sync::OnceLock, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    Predicate,
    Choice,
    Score,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    String(String),
    Bool(bool),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionChoice {
    Id(ChoiceValue),
    Detailed {
        id: ChoiceValue,
        description: String,
    },
}
impl DecisionChoice {
    fn id(&self) -> &ChoiceValue {
        match self {
            Self::Id(id) | Self::Detailed { id, .. } => id,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionLevel {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionRubric {
    Text(String),
    Levels(Vec<DecisionLevel>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionQuestion {
    Predicate {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Choice {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        choices: Vec<DecisionChoice>,
    },
    Score {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        rubric: DecisionRubric,
    },
}
impl DecisionQuestion {
    pub fn kind(&self) -> DecisionType {
        match self {
            Self::Predicate { .. } => DecisionType::Predicate,
            Self::Choice { .. } => DecisionType::Choice,
            Self::Score { .. } => DecisionType::Score,
        }
    }
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name.as_deref()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionInput {
    Text(String),
    Messages(Vec<DecisionMessage>),
}
impl From<&str> for DecisionInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}
impl From<String> for DecisionInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionMessage {
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub role: String,
    pub content: DecisionMessageContent,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionMessageContent {
    Text(String),
    Parts(Vec<DecisionContent>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionContent {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<DecisionImageDetail>,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionImageDetail {
    Auto,
    Low,
    High,
    Original,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    #[serde(default)]
    pub model: String,
    pub input: DecisionInput,
    pub questions: Vec<DecisionQuestion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}
impl DecisionRequest {
    pub fn new(input: impl Into<DecisionInput>, questions: Vec<DecisionQuestion>) -> Self {
        Self {
            model: String::new(),
            input: input.into(),
            questions,
            safety_identifier: None,
        }
    }
    pub fn from_contract(value: &Value) -> Result<Self, VvLlmError> {
        validate_schema(value, false)?;
        let request: Self = serde_json::from_value(value.clone())?;
        request.validate()?;
        Ok(request)
    }
    pub fn to_contract(&self) -> Result<Value, VvLlmError> {
        let value = serde_json::to_value(self)?;
        validate_schema(&value, false)?;
        self.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), VvLlmError> {
        let mut names = HashSet::new();
        for question in &self.questions {
            if question.name().is_some_and(|name| !names.insert(name)) {
                return Err(VvLlmError::Configuration(
                    "question names must be unique".into(),
                ));
            }
            if let DecisionQuestion::Choice { choices, .. } = question {
                let ids: HashSet<_> = choices.iter().map(DecisionChoice::id).collect();
                if ids.len() != choices.len() {
                    return Err(VvLlmError::Configuration(
                        "choice IDs must be unique".into(),
                    ));
                }
            }
        }

        if let DecisionInput::Messages(messages) = &self.input {
            let images = messages
                .iter()
                .map(|message| match &message.content {
                    DecisionMessageContent::Text(_) => 0,
                    DecisionMessageContent::Parts(parts) => parts
                        .iter()
                        .filter(|part| matches!(part, DecisionContent::ImageUrl { .. }))
                        .count(),
                })
                .sum::<usize>();
            if images > 128 {
                return Err(VvLlmError::Configuration(
                    "decision input supports at most 128 images".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceProbability {
    pub choice: ChoiceValue,
    pub probability: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LevelProbability {
    pub value: u64,
    pub label: String,
    pub probability: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Predicate {
        #[serde(default)]
        name: Option<String>,
        probability: f64,
    },
    Choice {
        #[serde(default)]
        name: Option<String>,
        choice: ChoiceValue,
        options: Vec<ChoiceProbability>,
    },
    Score {
        #[serde(default)]
        name: Option<String>,
        score: f64,
        confidence: f64,
        probabilities: Vec<LevelProbability>,
    },
    Refusal {
        #[serde(default)]
        name: Option<String>,
    },
}
impl DecisionAnswer {
    fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. }
            | Self::Choice { name, .. }
            | Self::Score { name, .. }
            | Self::Refusal { name, .. } => name.as_deref(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DecisionUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<DecisionInputTokenDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<DecisionOutputTokenDetails>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DecisionInputTokenDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DecisionOutputTokenDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Vec<DecisionAnswer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<DecisionUsage>,
}
impl DecisionResponse {
    pub fn from_contract(value: &Value) -> Result<Self, VvLlmError> {
        validate_schema(value, true)?;
        Ok(serde_json::from_value(value.clone())?)
    }
    pub fn to_contract(&self) -> Result<Value, VvLlmError> {
        let value = serde_json::to_value(self)?;
        validate_schema(&value, true)?;
        Ok(value)
    }
    pub fn validate_for(&self, request: &DecisionRequest) -> Result<(), VvLlmError> {
        let fail = || {
            VvLlmError::classified(
                ErrorKind::Serialization,
                "decision answers do not match questions",
            )
        };
        if self.answers.len() != request.questions.len() {
            return Err(fail());
        }
        for (answer, question) in self.answers.iter().zip(&request.questions) {
            if answer.name() != question.name() {
                return Err(fail());
            }
            match (answer, question) {
                (DecisionAnswer::Refusal { .. }, _)
                | (DecisionAnswer::Predicate { .. }, DecisionQuestion::Predicate { .. }) => {}
                (
                    DecisionAnswer::Choice {
                        choice, options, ..
                    },
                    DecisionQuestion::Choice { choices, .. },
                ) => {
                    let supplied: HashSet<_> = choices.iter().map(DecisionChoice::id).collect();
                    let observed: HashSet<_> =
                        options.iter().map(|option| &option.choice).collect();
                    if observed.len() != options.len()
                        || observed != supplied
                        || !supplied.contains(choice)
                    {
                        return Err(fail());
                    }
                }
                (
                    DecisionAnswer::Score {
                        score,
                        probabilities,
                        ..
                    },
                    DecisionQuestion::Score { rubric, .. },
                ) => {
                    if let DecisionRubric::Levels(levels) = rubric {
                        let values: HashSet<_> =
                            probabilities.iter().map(|item| item.value).collect();
                        if probabilities.len() != levels.len()
                            || values.len() != probabilities.len()
                            || *score < 0.0
                            || *score > levels.len().saturating_sub(1) as f64
                            || probabilities.iter().any(|item| {
                                usize::try_from(item.value)
                                    .ok()
                                    .and_then(|i| levels.get(i))
                                    .is_none_or(|level| level.label != item.label)
                            })
                        {
                            return Err(fail());
                        }
                    }
                }
                _ => return Err(fail()),
            }
        }
        Ok(())
    }
}

fn validate_schema(value: &Value, response: bool) -> Result<(), VvLlmError> {
    static REQUEST: OnceLock<jsonschema::Validator> = OnceLock::new();
    static RESPONSE: OnceLock<jsonschema::Validator> = OnceLock::new();
    let (slot, source) = if response {
        (
            &RESPONSE,
            include_str!("../../contract/v1.3.0/schemas/decision-response.v1.schema.json"),
        )
    } else {
        (
            &REQUEST,
            include_str!("../../contract/v1.3.0/schemas/decision-request.v1.schema.json"),
        )
    };
    let validator = slot.get_or_init(|| {
        jsonschema::validator_for(
            &serde_json::from_str::<Value>(source).expect("vendored decision schema JSON"),
        )
        .expect("vendored decision schema")
    });
    validator.validate(value).map_err(|error| {
        VvLlmError::classified(
            if response {
                ErrorKind::Serialization
            } else {
                ErrorKind::Configuration
            },
            format!(
                "decision schema validation failed at {}",
                error.instance_path()
            ),
        )
    })
}

#[allow(
    clippy::double_must_use,
    reason = "async_trait adds must_use to futures"
)]
#[async_trait]
pub trait DecisionClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn create(&self, request: DecisionRequest) -> Result<DecisionResponse, VvLlmError>;
}

/// Uses the typed vv-llm protocol over reqwest because async-openai 0.40.2 has no Decisions resource.
#[derive(Clone)]
pub struct OpenAiDecisionClient {
    model: String,
    public_model: Option<String>,
    api_base: String,
    api_key: String,
    http: reqwest::Client,
    headers: reqwest::header::HeaderMap,
    capabilities: ModelCapabilities,
}
impl OpenAiDecisionClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let model = model.into();
        Self {
            capabilities: crate::defaults::default_model_capabilities(&model),
            model,
            public_model: None,
            api_base: api_base.into(),
            api_key: api_key.into(),
            http: reqwest::Client::new(),
            headers: reqwest::header::HeaderMap::new(),
        }
    }
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
    pub fn to_openai_json(&self, request: &DecisionRequest) -> Result<Value, VvLlmError> {
        let mut request = request.clone();
        if request.model.is_empty() || self.public_model.as_deref() == Some(request.model.as_str())
        {
            request.model.clone_from(&self.model);
        }
        let capabilities = if request.model == self.model {
            self.capabilities.clone()
        } else {
            crate::defaults::default_model_capabilities(&request.model)
        };
        if !capabilities.decision_types.as_ref().is_some_and(|types| {
            request
                .questions
                .iter()
                .all(|question| types.contains(&question.kind()))
        }) {
            return Err(VvLlmError::Configuration(
                "selected model does not declare support for the requested decision types".into(),
            ));
        }
        request.to_contract()
    }
    pub fn normalize_response(value: Value) -> Result<DecisionResponse, VvLlmError> {
        let response: DecisionResponse = serde_json::from_value(value)?;
        response.to_contract()?;
        Ok(response)
    }
}
#[async_trait]
impl DecisionClient for OpenAiDecisionClient {
    fn provider_name(&self) -> &'static str {
        "openai"
    }
    async fn create(&self, request: DecisionRequest) -> Result<DecisionResponse, VvLlmError> {
        let body = self.to_openai_json(&request)?;
        let response = self
            .http
            .post(format!("{}/decisions", self.api_base.trim_end_matches('/')))
            .headers(self.headers.clone())
            .bearer_auth(&self.api_key)
            .timeout(Duration::from_secs(60))
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                VvLlmError::classified(
                    if error.is_timeout() {
                        ErrorKind::Timeout
                    } else {
                        ErrorKind::Network
                    },
                    error.to_string(),
                )
            })?;
        let response = crate::utilities::http::ensure_success(response).await?;
        let raw = response.json::<Value>().await.map_err(|_| {
            VvLlmError::classified(
                ErrorKind::Serialization,
                "provider returned invalid decision JSON",
            )
        })?;
        let response = Self::normalize_response(raw)?;
        response.validate_for(&request)?;
        Ok(response)
    }
}

pub fn create_decision_client(
    backend: BackendType,
    model: impl Into<String>,
    api_base: impl Into<String>,
    api_key: impl Into<String>,
) -> Result<Box<dyn DecisionClient>, VvLlmError> {
    if backend != BackendType::OpenAI {
        return Err(VvLlmError::Configuration(
            "unsupported decision backend".into(),
        ));
    }
    Ok(Box::new(OpenAiDecisionClient::new(
        model, api_base, api_key,
    )))
}
pub fn create_decision_client_from_resolved(
    resolved: &ResolvedModelConfig,
) -> Result<Box<dyn DecisionClient>, VvLlmError> {
    let endpoint = &resolved.endpoint;
    if resolved.backend != "openai"
        || endpoint.is_azure
        || endpoint.is_vertex
        || endpoint.is_bedrock
        || endpoint
            .endpoint_type
            .as_deref()
            .is_some_and(|kind| !["default", "openai"].contains(&kind))
    {
        return Err(VvLlmError::Configuration(
            "decision client requires an OpenAI endpoint".into(),
        ));
    }
    let mut client = OpenAiDecisionClient::new(
        &resolved.model_id,
        endpoint
            .api_base
            .as_deref()
            .unwrap_or("https://api.openai.com/v1"),
        endpoint.api_key.as_deref().unwrap_or(""),
    )
    .with_capabilities(resolved.model.capabilities());
    client.public_model = Some(resolved.model.id.clone());
    let mut builder = reqwest::Client::builder();
    if let Some(proxy) = endpoint
        .proxy
        .as_ref()
        .filter(|proxy| !proxy.trim().is_empty())
    {
        builder = builder.proxy(
            reqwest::Proxy::all(proxy)
                .map_err(|_| VvLlmError::Configuration("invalid decision proxy".into()))?,
        );
    }
    client.http = builder
        .build()
        .map_err(|_| VvLlmError::Configuration("invalid decision HTTP client".into()))?;
    if let Some(headers) = &endpoint.headers {
        for (key, value) in headers {
            client.headers.insert(
                reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|_| {
                    VvLlmError::Configuration("invalid decision header name".into())
                })?,
                reqwest::header::HeaderValue::from_str(value).map_err(|_| {
                    VvLlmError::Configuration("invalid decision header value".into())
                })?,
            );
        }
    }
    Ok(Box::new(client))
}
