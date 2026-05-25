use crate::{RerankResponse, RerankResult, VvLlmError};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::RerankClient;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankMapping {
    pub method: String,
    pub path: String,
}

impl RerankMapping {
    pub fn default_siliconflow() -> Self {
        Self {
            method: "POST".to_string(),
            path: "/rerank".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CustomJsonHttpRerankClient {
    model: String,
    api_base: String,
    api_key: String,
    mapping: RerankMapping,
    http: reqwest::Client,
}

impl CustomJsonHttpRerankClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        mapping: RerankMapping,
    ) -> Self {
        Self {
            model: model.into(),
            api_base: api_base.into(),
            api_key: api_key.into(),
            mapping,
            http: reqwest::Client::new(),
        }
    }

    pub fn build_request_body(&self, query: &str, documents: &[&str]) -> Result<Value, VvLlmError> {
        Ok(json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        }))
    }

    fn endpoint_url(&self) -> String {
        format!(
            "{}{}",
            self.api_base.trim_end_matches('/'),
            self.mapping.path.as_str()
        )
    }
}

#[async_trait]
impl RerankClient for CustomJsonHttpRerankClient {
    fn provider_name(&self) -> &'static str {
        "custom-json-http"
    }

    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<RerankResponse, VvLlmError> {
        let response = self
            .http
            .post(self.endpoint_url())
            .bearer_auth(&self.api_key)
            .json(&self.build_request_body(query, documents)?)
            .send()
            .await
            .map_err(|error| VvLlmError::Http(error.to_string()))?
            .error_for_status()
            .map_err(|error| VvLlmError::Http(error.to_string()))?
            .json::<Value>()
            .await
            .map_err(|error| VvLlmError::Http(error.to_string()))?;

        let results = response
            .get("results")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| RerankResult {
                        index: item.get("index").and_then(Value::as_u64).unwrap_or(0) as usize,
                        relevance_score: item
                            .get("relevance_score")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0) as f32,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(RerankResponse { results })
    }
}
