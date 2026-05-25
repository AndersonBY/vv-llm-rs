use crate::{BackendType, VvLlmError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub id: String,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    #[serde(default)]
    pub endpoints: Vec<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub function_call_available: Option<bool>,
    #[serde(default)]
    pub response_format_available: Option<bool>,
    #[serde(default)]
    pub native_multimodal: Option<bool>,
    #[serde(default)]
    pub protocol: Option<String>,
    #[serde(default)]
    pub request_mapping: Option<Value>,
    #[serde(default)]
    pub response_mapping: Option<Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlmSettings {
    #[serde(rename = "VERSION", default)]
    pub version: Option<String>,
    #[serde(default)]
    pub endpoints: Vec<EndpointConfig>,
    #[serde(default)]
    pub backends: HashMap<String, BackendConfig>,
    #[serde(default)]
    pub embedding_backends: HashMap<String, BackendConfig>,
    #[serde(default)]
    pub rerank_backends: HashMap<String, BackendConfig>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModelConfig {
    pub backend: String,
    pub model: ModelConfig,
    pub endpoint: EndpointConfig,
}

impl LlmSettings {
    pub fn from_json_str(raw: &str) -> Result<Self, VvLlmError> {
        Ok(serde_json::from_str(raw)?)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, VvLlmError> {
        let raw = fs::read_to_string(path.as_ref())
            .map_err(|error| VvLlmError::Configuration(error.to_string()))?;
        Self::from_json_str(&raw)
    }

    pub fn resolve_chat_model(
        &self,
        backend: BackendType,
        model_id: &str,
    ) -> Result<ResolvedModelConfig, VvLlmError> {
        self.resolve_model_in_map(&self.backends, backend.as_str(), model_id)
    }

    pub fn resolve_embedding_model(
        &self,
        backend: &str,
        model_id: &str,
    ) -> Result<ResolvedModelConfig, VvLlmError> {
        self.resolve_model_in_map(&self.embedding_backends, backend, model_id)
    }

    pub fn resolve_rerank_model(
        &self,
        backend: &str,
        model_id: &str,
    ) -> Result<ResolvedModelConfig, VvLlmError> {
        self.resolve_model_in_map(&self.rerank_backends, backend, model_id)
    }

    fn resolve_model_in_map(
        &self,
        map: &HashMap<String, BackendConfig>,
        backend: &str,
        model_id: &str,
    ) -> Result<ResolvedModelConfig, VvLlmError> {
        let backend_config = map.get(backend).ok_or_else(|| VvLlmError::ModelNotFound {
            backend: backend.to_string(),
            model: model_id.to_string(),
        })?;
        let model = backend_config
            .models
            .get(model_id)
            .or_else(|| {
                backend_config
                    .models
                    .values()
                    .find(|model| model.id == model_id)
            })
            .ok_or_else(|| VvLlmError::ModelNotFound {
                backend: backend.to_string(),
                model: model_id.to_string(),
            })?;
        let endpoint_id = model.endpoints.first().ok_or_else(|| {
            VvLlmError::Configuration(format!("model {model_id} has no endpoints"))
        })?;
        let endpoint = self
            .endpoints
            .iter()
            .find(|endpoint| endpoint.id == *endpoint_id)
            .ok_or_else(|| VvLlmError::EndpointNotFound(endpoint_id.clone()))?;

        Ok(ResolvedModelConfig {
            backend: backend.to_string(),
            model: model.clone(),
            endpoint: endpoint.clone(),
        })
    }
}
