use serde::{Deserialize, Serialize};
use std::fmt;

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
    Text { text: String },
    ImageUrl { url: String },
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
}

impl Message {
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![MessageContent::Text { text: text.into() }],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn text_content(&self) -> Option<String> {
        let mut parts = Vec::new();
        for content in &self.content {
            if let MessageContent::Text { text } = content {
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
    pub stream: Option<bool>,
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
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            options: ChatRequestOptions::default(),
            tools: Vec::new(),
            tool_choice: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
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
    pub usage: Option<ChatUsage>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

fn is_false(value: &bool) -> bool {
    !*value
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
}
