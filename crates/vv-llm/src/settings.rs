use crate::{
    default_chat_backends, BackendType, Modality, ModelCapabilities, StructuredOutputCapability,
    VvLlmError,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{collections::HashMap, fs, path::Path};

const DEFAULT_CHAT_CONTEXT_LENGTH: u32 = 32_768;

fn default_true() -> bool {
    true
}

fn default_endpoint_rpm() -> u32 {
    60
}

fn default_endpoint_tpm() -> u32 {
    300_000
}

fn default_endpoint_concurrent_requests() -> u32 {
    20
}

fn default_rate_limit_backend() -> String {
    "memory".to_string()
}

fn default_rate_limit_rpm() -> u32 {
    60
}

fn default_rate_limit_tpm() -> u32 {
    1_000_000
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub id: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub response_api: bool,
    #[serde(default)]
    pub endpoint_type: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub is_azure: bool,
    #[serde(default)]
    pub is_bedrock: bool,
    #[serde(default)]
    pub is_vertex: bool,
    #[serde(default)]
    pub credentials: Value,
    #[serde(default = "default_endpoint_rpm")]
    pub rpm: u32,
    #[serde(default = "default_endpoint_tpm")]
    pub tpm: u32,
    #[serde(default = "default_endpoint_concurrent_requests")]
    pub concurrent_requests: u32,
    #[serde(default)]
    pub proxy: Option<String>,
    #[serde(default)]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub access_token: Option<String>,
    #[serde(default)]
    pub access_token_expires_at: Option<f64>,
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BackendConfig {
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    #[serde(default)]
    pub default_endpoint: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub endpoints: Vec<EndpointBinding>,
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
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub default_top_n: Option<u32>,
    #[serde(default)]
    pub request_mapping: Option<Value>,
    #[serde(default)]
    pub response_mapping: Option<Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl ModelConfig {
    pub fn capabilities(&self) -> ModelCapabilities {
        if let Some(value) = self.extra.get("capabilities") {
            if let Ok(capabilities) = serde_json::from_value(value.clone()) {
                return capabilities;
            }
        }

        let mut capabilities = ModelCapabilities {
            tools: self.function_call_available.unwrap_or(false),
            structured_output: if self.response_format_available.unwrap_or(false) {
                StructuredOutputCapability::JsonSchema
            } else {
                StructuredOutputCapability::None
            },
            ..Default::default()
        };
        if self.native_multimodal.unwrap_or(false) {
            capabilities.input_modalities.insert(Modality::Image);
        }
        capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_rate_limit_backend")]
    pub backend: String,
    #[serde(default)]
    pub redis: Option<Value>,
    #[serde(default)]
    pub diskcache: Option<Value>,
    #[serde(default = "default_rate_limit_rpm")]
    pub default_rpm: u32,
    #[serde(default = "default_rate_limit_tpm")]
    pub default_tpm: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LlmSettings {
    #[serde(rename = "VERSION", default)]
    pub version: Option<String>,
    #[serde(default)]
    pub endpoints: Vec<EndpointConfig>,
    #[serde(default)]
    pub token_server: Option<ServerConfig>,
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
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
    pub model_id: String,
    pub endpoint: EndpointConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EndpointBinding {
    Id(String),
    Config {
        endpoint_id: String,
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        enabled: Option<bool>,
        #[serde(default)]
        rpm: Option<u32>,
        #[serde(default)]
        tpm: Option<u32>,
        #[serde(default)]
        concurrent_requests: Option<u32>,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
}

impl EndpointBinding {
    pub fn endpoint_id(&self) -> &str {
        match self {
            Self::Id(endpoint_id) => endpoint_id,
            Self::Config { endpoint_id, .. } => endpoint_id,
        }
    }

    pub fn model_id<'a>(&'a self, default_model_id: &'a str) -> &'a str {
        match self {
            Self::Id(_) => default_model_id,
            Self::Config { model_id, .. } => model_id.as_deref().unwrap_or(default_model_id),
        }
    }

    pub fn enabled(&self) -> bool {
        match self {
            Self::Id(_) => true,
            Self::Config { enabled, .. } => enabled.unwrap_or(true),
        }
    }
}

impl LlmSettings {
    pub fn from_json_str(raw: &str) -> Result<Self, VvLlmError> {
        let mut settings: Self = serde_json::from_str(raw)?;
        settings.normalize_after_load();
        Ok(settings)
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

    fn normalize_after_load(&mut self) {
        for endpoint in &mut self.endpoints {
            normalize_endpoint_transport_flags(endpoint);
            normalize_google_openai_base(endpoint);
            preserve_vertex_cached_token(endpoint);
        }

        self.merge_default_chat_backends();
        self.apply_python_chat_model_defaults();
    }

    fn merge_default_chat_backends(&mut self) {
        for (backend_name, default_backend) in default_chat_backends() {
            let backend = match self.backends.remove(&backend_name) {
                Some(user_backend) => merge_backend_config(default_backend, user_backend),
                None => default_backend,
            };
            self.backends.insert(backend_name, backend);
        }

        for backend in self.backends.values_mut() {
            apply_default_endpoint(backend);
        }
    }

    fn apply_python_chat_model_defaults(&mut self) {
        for backend in self.backends.values_mut() {
            for model in backend.models.values_mut() {
                model
                    .context_length
                    .get_or_insert(DEFAULT_CHAT_CONTEXT_LENGTH);
                model.function_call_available.get_or_insert(false);
                model.response_format_available.get_or_insert(false);
                model.native_multimodal.get_or_insert(false);
            }
        }
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
            .filter(|model| model.enabled)
            .or_else(|| {
                backend_config
                    .models
                    .values()
                    .find(|model| model.enabled && model.id == model_id)
            })
            .ok_or_else(|| VvLlmError::ModelNotFound {
                backend: backend.to_string(),
                model: model_id.to_string(),
            })?;
        let binding = model
            .endpoints
            .iter()
            .find(|binding| binding.enabled())
            .ok_or_else(|| {
                VvLlmError::Configuration(format!("model {model_id} has no enabled endpoints"))
            })?;
        let endpoint_id = binding.endpoint_id();
        let endpoint = self
            .endpoints
            .iter()
            .find(|endpoint| endpoint.enabled && endpoint.id == endpoint_id)
            .ok_or_else(|| VvLlmError::EndpointNotFound(endpoint_id.to_string()))?;

        Ok(ResolvedModelConfig {
            backend: backend.to_string(),
            model: model.clone(),
            model_id: binding.model_id(&model.id).to_string(),
            endpoint: endpoint.clone(),
        })
    }
}

fn merge_backend_config(
    mut default_backend: BackendConfig,
    user_backend: BackendConfig,
) -> BackendConfig {
    let default_endpoint = user_backend
        .default_endpoint
        .clone()
        .or(default_backend.default_endpoint.clone());

    for (model_name, user_model) in user_backend.models {
        let merged_model = match default_backend.models.remove(&model_name) {
            Some(default_model) => merge_model_config(default_model, user_model),
            None => user_model,
        };
        default_backend.models.insert(model_name, merged_model);
    }

    default_backend.default_endpoint = default_endpoint;
    default_backend.extra.extend(user_backend.extra);
    default_backend
}

fn merge_model_config(mut default_model: ModelConfig, user_model: ModelConfig) -> ModelConfig {
    let user_function_call_available = user_model.function_call_available;
    let user_response_format_available = user_model.response_format_available;
    let user_native_multimodal = user_model.native_multimodal;
    let user_has_capabilities = user_model.extra.contains_key("capabilities");

    default_model.id = user_model.id;
    default_model.enabled = user_model.enabled;
    if !user_model.endpoints.is_empty() {
        default_model.endpoints = user_model.endpoints;
    }
    if user_model.context_length.is_some() {
        default_model.context_length = user_model.context_length;
    }
    if user_model.max_output_tokens.is_some() {
        default_model.max_output_tokens = user_model.max_output_tokens;
    }
    if user_model.function_call_available.is_some() {
        default_model.function_call_available = user_model.function_call_available;
    }
    if user_model.response_format_available.is_some() {
        default_model.response_format_available = user_model.response_format_available;
    }
    if user_model.native_multimodal.is_some() {
        default_model.native_multimodal = user_model.native_multimodal;
    }
    if user_model.protocol.is_some() {
        default_model.protocol = user_model.protocol;
    }
    if user_model.dimensions.is_some() {
        default_model.dimensions = user_model.dimensions;
    }
    if user_model.default_top_n.is_some() {
        default_model.default_top_n = user_model.default_top_n;
    }
    if user_model.request_mapping.is_some() {
        default_model.request_mapping = user_model.request_mapping;
    }
    if user_model.response_mapping.is_some() {
        default_model.response_mapping = user_model.response_mapping;
    }
    default_model.extra.extend(user_model.extra);
    if !user_has_capabilities {
        if let Some(value) = default_model.extra.get("capabilities").cloned() {
            if let Ok(mut capabilities) = serde_json::from_value::<ModelCapabilities>(value) {
                if let Some(available) = user_function_call_available {
                    capabilities.tools = available;
                }
                if let Some(available) = user_response_format_available {
                    capabilities.structured_output = if available {
                        StructuredOutputCapability::JsonSchema
                    } else {
                        StructuredOutputCapability::None
                    };
                }
                if let Some(available) = user_native_multimodal {
                    if available {
                        capabilities.input_modalities.insert(Modality::Image);
                    } else {
                        capabilities.input_modalities.remove(&Modality::Image);
                    }
                }
                if let Ok(value) = serde_json::to_value(capabilities) {
                    default_model
                        .extra
                        .insert("capabilities".to_string(), value);
                }
            }
        }
    }
    default_model
}

fn apply_default_endpoint(backend: &mut BackendConfig) {
    let Some(default_endpoint) = backend.default_endpoint.clone() else {
        return;
    };
    for model in backend.models.values_mut() {
        if model.endpoints.is_empty() {
            model
                .endpoints
                .push(EndpointBinding::Id(default_endpoint.clone()));
        }
    }
}

fn normalize_endpoint_transport_flags(endpoint: &mut EndpointConfig) {
    let endpoint_type = endpoint
        .endpoint_type
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let normalized = if endpoint_type.is_empty() || endpoint_type == "default" {
        if endpoint.is_azure {
            Some("openai_azure")
        } else if endpoint.is_vertex {
            Some("anthropic_vertex")
        } else if endpoint.is_bedrock {
            Some("anthropic_bedrock")
        } else if endpoint_type == "default" {
            Some("default")
        } else {
            None
        }
    } else {
        Some(endpoint_type.as_str())
    };

    if let Some(endpoint_type) = normalized {
        endpoint.endpoint_type = Some(endpoint_type.to_string());
        endpoint.is_azure = endpoint_type == "openai_azure";
        endpoint.is_vertex =
            endpoint_type == "anthropic_vertex" || endpoint_type == "openai_vertex";
        endpoint.is_bedrock = endpoint_type == "anthropic_bedrock";
    }
}

fn normalize_google_openai_base(endpoint: &mut EndpointConfig) {
    let Some(api_base) = endpoint.api_base.as_deref() else {
        return;
    };
    if api_base.starts_with("https://generativelanguage.googleapis.com/v1beta")
        && !api_base.ends_with("openai/")
    {
        endpoint.api_base = Some(format!("{}/openai/", api_base.trim_end_matches('/')));
    }
}

fn preserve_vertex_cached_token(endpoint: &mut EndpointConfig) {
    if endpoint.access_token.is_none() && endpoint.access_token_expires_at.is_none() {
        return;
    }

    if !endpoint.credentials.is_object() {
        endpoint.credentials = Value::Object(Map::new());
    }
    let credentials = endpoint
        .credentials
        .as_object_mut()
        .expect("credentials was normalized to an object");

    if let Some(access_token) = &endpoint.access_token {
        credentials
            .entry("access_token")
            .or_insert_with(|| Value::String(access_token.clone()));
    }
    if let Some(expires_at) = endpoint.access_token_expires_at {
        if let Some(number) = serde_json::Number::from_f64(expires_at) {
            credentials
                .entry("access_token_expires_at")
                .or_insert_with(|| Value::Number(number));
        }
    }
}
