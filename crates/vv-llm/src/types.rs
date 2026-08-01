use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    OpenAI,
    ZhiPuAI,
    MiniMax,
    Moonshot,
    Anthropic,
    Mistral,
    DeepSeek,
    Qwen,
    Groq,
    Local,
    Yi,
    Gemini,
    Baichuan,
    StepFun,
    XAI,
    Xiaomi,
    Ernie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputCapability {
    #[default]
    None,
    JsonObject,
    JsonSchema,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingCapability {
    #[default]
    Unknown,
    Unsupported,
    Configurable,
    AlwaysEnabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub structured_output: StructuredOutputCapability,
    #[serde(default = "default_input_modalities")]
    pub input_modalities: HashSet<Modality>,
    #[serde(default = "default_output_modalities")]
    pub output_modalities: HashSet<Modality>,
    #[serde(default = "default_true")]
    pub streaming: bool,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub thinking: ThinkingCapability,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            tools: false,
            structured_output: StructuredOutputCapability::None,
            input_modalities: default_input_modalities(),
            output_modalities: default_output_modalities(),
            streaming: true,
            parallel_tool_calls: false,
            thinking: ThinkingCapability::Unknown,
        }
    }
}

impl ModelCapabilities {
    pub fn validate_request(&self, request: &ChatRequest) -> Result<(), VvLlmError> {
        let mut conflicts = Vec::new();
        if !request.tools.is_empty() && !self.tools {
            conflicts.push("the model does not support tools");
        }
        if request.options.response_format.is_some()
            && self.structured_output == StructuredOutputCapability::None
        {
            conflicts.push("the model does not support structured output");
        }
        if request.options.stream == Some(true) && !self.streaming {
            conflicts.push("the model does not support streaming");
        }
        if request.messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|content| matches!(content, MessageContent::ImageUrl { .. }))
        }) && !self.input_modalities.contains(&Modality::Image)
        {
            conflicts.push("the model does not support image input");
        }
        if let Some(thinking) = &request.options.thinking {
            match self.thinking {
                ThinkingCapability::Unsupported => {
                    conflicts.push("the model does not support thinking controls")
                }
                ThinkingCapability::Unknown => {
                    conflicts.push("the model's thinking capability is unknown")
                }
                ThinkingCapability::AlwaysEnabled
                    if thinking.get("type").and_then(serde_json::Value::as_str)
                        == Some("disabled") =>
                {
                    conflicts.push("thinking is always enabled for this model")
                }
                _ => {}
            }
        }
        if conflicts.is_empty() {
            Ok(())
        } else {
            Err(VvLlmError::Configuration(conflicts.join("; ")))
        }
    }
}

fn default_input_modalities() -> HashSet<Modality> {
    HashSet::from([Modality::Text])
}

fn default_output_modalities() -> HashSet<Modality> {
    HashSet::from([Modality::Text])
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThinkingPreference {
    Default,
    Enabled { budget_tokens: Option<u32> },
    Disabled,
    ProviderDefined(serde_json::Value),
}

impl ThinkingPreference {
    pub fn enabled() -> Self {
        Self::Enabled {
            budget_tokens: None,
        }
    }

    pub fn enabled_with_budget(budget_tokens: u32) -> Self {
        Self::Enabled {
            budget_tokens: Some(budget_tokens),
        }
    }

    pub fn into_provider_value(self) -> Option<serde_json::Value> {
        match self {
            Self::Default => None,
            Self::Enabled { budget_tokens } => {
                let mut value = serde_json::json!({"type": "enabled"});
                if let Some(budget_tokens) = budget_tokens {
                    value["budget_tokens"] = serde_json::json!(budget_tokens);
                }
                Some(value)
            }
            Self::Disabled => Some(serde_json::json!({"type": "disabled"})),
            Self::ProviderDefined(value) => Some(value),
        }
    }
}

impl BackendType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::ZhiPuAI => "zhipuai",
            Self::MiniMax => "minimax",
            Self::Moonshot => "moonshot",
            Self::Anthropic => "anthropic",
            Self::Mistral => "mistral",
            Self::DeepSeek => "deepseek",
            Self::Qwen => "qwen",
            Self::Groq => "groq",
            Self::Local => "local",
            Self::Yi => "yi",
            Self::Gemini => "gemini",
            Self::Baichuan => "baichuan",
            Self::StepFun => "stepfun",
            Self::XAI => "xai",
            Self::Xiaomi => "xiaomi",
            Self::Ernie => "ernie",
        }
    }
}

impl fmt::Display for BackendType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    ImageUrl {
        url: String,
    },
}

impl MessageContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn text_with_cache_control(
        text: impl Into<String>,
        cache_control: serde_json::Value,
    ) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: Some(cache_control),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![MessageContent::text(text)],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        }
    }

    pub fn text_content(&self) -> Option<String> {
        let mut parts = Vec::new();
        for content in &self.content {
            if let MessageContent::Text { text, .. } = content {
                parts.push(text.as_str());
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatRequestOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl ChatRequestOptions {
    pub fn with_thinking(mut self, preference: ThinkingPreference) -> Self {
        self.thinking = preference.into_provider_value();
        self
    }

    pub fn has_openai_json_extensions(&self) -> bool {
        self.max_completion_tokens.is_some()
            || self.top_p.is_some()
            || !self.stop.is_empty()
            || self.response_format.is_some()
            || self.stream_options.is_some()
            || self.audio.is_some()
            || self.frequency_penalty.is_some()
            || self.logit_bias.is_some()
            || self.logprobs.is_some()
            || self.metadata.is_some()
            || self.modalities.is_some()
            || self.n.is_some()
            || self.parallel_tool_calls.is_some()
            || self.prediction.is_some()
            || self.presence_penalty.is_some()
            || self.reasoning_effort.is_some()
            || self.thinking.is_some()
            || self.seed.is_some()
            || self.service_tier.is_some()
            || self.store.is_some()
            || self.top_logprobs.is_some()
            || self.user.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub options: ChatRequestOptions,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(default, skip_serializing_if = "is_empty_extra_body")]
    pub extra_body: serde_json::Value,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            options: ChatRequestOptions::default(),
            tools: Vec::new(),
            tool_choice: None,
            extra_body: serde_json::Value::Null,
        }
    }

    pub fn with_thinking(mut self, preference: ThinkingPreference) -> Self {
        self.options = self.options.with_thinking(preference);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

impl ChatTool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters,
            cache_control: None,
        }
    }

    pub fn with_cache_control(mut self, cache_control: serde_json::Value) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<serde_json::Value>,
}

impl ToolCall {
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            index: None,
            extra_content: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default = "default_attempts")]
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(default)]
    pub fallback_index: usize,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, serde_json::Value>,
}

impl Default for ResponseMetadata {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            response_id: None,
            request_id: None,
            finish_reason: None,
            attempts: 1,
            latency_ms: None,
            fallback_index: 0,
            attributes: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResult<Response> {
    pub response: Response,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatStreamDelta {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning_content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub done: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_usage: Option<serde_json::Value>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_attempts() -> u32 {
    1
}

fn is_empty_extra_body(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Object(object) => object.is_empty(),
        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Authentication,
    RateLimited,
    Network,
    Timeout,
    InvalidRequest,
    ContextLength,
    ContentPolicy,
    ModelNotFound,
    ProviderInternal,
    Serialization,
    Configuration,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub kind: ErrorKind,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl ErrorDetails {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider: None,
            model: None,
            status_code: None,
            provider_code: None,
            retry_after_seconds: None,
            request_id: None,
        }
    }

    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_status_code(mut self, status_code: u16) -> Self {
        self.status_code = Some(status_code);
        self
    }

    pub fn with_retry_after(mut self, seconds: f64) -> Self {
        self.retry_after_seconds = Some(seconds.max(0.0));
        self
    }
}

impl std::fmt::Display for ErrorDetails {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VvLlmError {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("model not found: backend={backend} model={model}")]
    ModelNotFound { backend: String, model: String },
    #[error("endpoint not found: {0}")]
    EndpointNotFound(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("http error: {0}")]
    Http(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("{0}")]
    Classified(Box<ErrorDetails>),
}

impl VvLlmError {
    pub fn classified(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self::Classified(Box::new(ErrorDetails::new(kind, message)))
    }

    pub fn from_status(status_code: u16, message: impl Into<String>) -> Self {
        Self::from_status_with_retry_after(status_code, message, None)
    }

    pub fn from_status_with_retry_after(
        status_code: u16,
        message: impl Into<String>,
        retry_after: Option<Duration>,
    ) -> Self {
        let message = message.into();
        let kind = classify_status(status_code, &message);
        let mut details = ErrorDetails::new(kind, message).with_status_code(status_code);
        if let Some(retry_after) = retry_after {
            details = details.with_retry_after(retry_after.as_secs_f64());
        }
        Self::Classified(Box::new(details))
    }

    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Configuration(_) | Self::EndpointNotFound(_) => ErrorKind::Configuration,
            Self::ModelNotFound { .. } => ErrorKind::ModelNotFound,
            Self::Serialization(_) => ErrorKind::Serialization,
            Self::Http(message) => classify_legacy_http_error(message),
            Self::Provider(message) => classify_legacy_provider_error(message),
            Self::Classified(details) => details.kind,
        }
    }

    pub fn retryable(&self) -> bool {
        matches!(
            self.kind(),
            ErrorKind::RateLimited
                | ErrorKind::Network
                | ErrorKind::Timeout
                | ErrorKind::ProviderInternal
        )
    }

    pub fn retry_after_seconds(&self) -> Option<f64> {
        match self {
            Self::Classified(details) => details.retry_after_seconds,
            _ => None,
        }
    }
}

fn classify_status(status_code: u16, message: &str) -> ErrorKind {
    let normalized = message.to_ascii_lowercase();
    match status_code {
        401 | 403 => ErrorKind::Authentication,
        404 => ErrorKind::ModelNotFound,
        429 => ErrorKind::RateLimited,
        400..=499 if is_context_length(&normalized) => ErrorKind::ContextLength,
        400..=499 if is_content_policy(&normalized) => ErrorKind::ContentPolicy,
        400..=499 => ErrorKind::InvalidRequest,
        500..=599 => ErrorKind::ProviderInternal,
        _ => ErrorKind::Unknown,
    }
}

fn classify_legacy_http_error(message: &str) -> ErrorKind {
    let normalized = message.to_ascii_lowercase();
    if normalized.contains("timeout") || normalized.contains("timed out") {
        ErrorKind::Timeout
    } else {
        ErrorKind::Network
    }
}

fn classify_legacy_provider_error(message: &str) -> ErrorKind {
    let normalized = message.to_ascii_lowercase();
    if contains_status(&normalized, 401)
        || contains_status(&normalized, 403)
        || normalized.contains("unauthorized")
        || normalized.contains("authentication")
        || normalized.contains("invalid api key")
    {
        ErrorKind::Authentication
    } else if contains_status(&normalized, 429)
        || normalized.contains("rate limit")
        || normalized.contains("too many requests")
    {
        ErrorKind::RateLimited
    } else if contains_status(&normalized, 404)
        || normalized.contains("model not found")
        || normalized.contains("model_not_found")
    {
        ErrorKind::ModelNotFound
    } else if is_context_length(&normalized) {
        ErrorKind::ContextLength
    } else if is_content_policy(&normalized) {
        ErrorKind::ContentPolicy
    } else if contains_status(&normalized, 400)
        || normalized.contains("invalid request")
        || normalized.contains("bad request")
    {
        ErrorKind::InvalidRequest
    } else if normalized.contains("timeout") || normalized.contains("timed out") {
        ErrorKind::Timeout
    } else if normalized.contains("connection")
        || normalized.contains("network")
        || normalized.contains("dns")
    {
        ErrorKind::Network
    } else {
        ErrorKind::ProviderInternal
    }
}

fn contains_status(message: &str, status: u16) -> bool {
    let status = status.to_string();
    message
        .split(|character: char| !character.is_ascii_digit())
        .any(|part| part == status)
}

fn is_context_length(message: &str) -> bool {
    message.contains("context")
        && (message.contains("length")
            || message.contains("token limit")
            || message.contains("too many tokens"))
}

fn is_content_policy(message: &str) -> bool {
    message.contains("content")
        && (message.contains("policy")
            || message.contains("filter")
            || message.contains("moderation"))
}
