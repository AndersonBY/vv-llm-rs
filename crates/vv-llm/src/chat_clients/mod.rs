mod anthropic;
mod openai_compatible;
mod vertex;

use crate::{BackendType, ChatRequest, ChatStreamDelta, ResolvedModelConfig, VvLlmError};
use async_trait::async_trait;
use futures_core::Stream;
use std::pin::Pin;

pub use anthropic::{AnthropicBedrockChatClient, AnthropicChatClient};
pub use openai_compatible::OpenAiCompatibleChatClient;
pub use vertex::{GoogleAccessToken, GoogleAccessTokenProvider, VertexOpenAiChatClient};

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamDelta, VvLlmError>> + Send>>;

#[async_trait]
pub trait ChatClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn create_completion(
        &self,
        request: ChatRequest,
    ) -> Result<crate::ChatResponse, VvLlmError>;
    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError>;
}

pub fn create_chat_client(
    backend: BackendType,
    model: impl Into<String>,
    api_base: impl Into<String>,
    api_key: impl Into<String>,
) -> Box<dyn ChatClient> {
    let model = model.into();
    let api_base = api_base.into();
    let api_key = api_key.into();

    match backend {
        BackendType::Anthropic => Box::new(AnthropicChatClient::new(model, api_base, api_key)),
        _ => Box::new(OpenAiCompatibleChatClient::new(model, api_base, api_key)),
    }
}

pub fn create_chat_client_from_resolved(
    resolved: ResolvedModelConfig,
) -> Result<Box<dyn ChatClient>, VvLlmError> {
    let backend = resolved.backend.as_str();
    let model = resolved.model_id;
    let api_base = resolved.endpoint.api_base.unwrap_or_default();
    let api_key = resolved.endpoint.api_key.unwrap_or_default();

    if backend == BackendType::Anthropic.as_str()
        && (resolved.endpoint.is_bedrock.unwrap_or(false)
            || resolved.endpoint.endpoint_type.as_deref() == Some("anthropic_bedrock"))
    {
        return Ok(Box::new(AnthropicBedrockChatClient::new(
            model,
            api_base,
            resolved.endpoint.region,
            resolved.endpoint.credentials,
        )?));
    }

    if resolved.endpoint.endpoint_type.as_deref() == Some("openai_vertex") {
        return Ok(Box::new(VertexOpenAiChatClient::new(
            model,
            api_base,
            resolved.endpoint.credentials,
        )?));
    }

    let backend = match backend {
        "anthropic" => BackendType::Anthropic,
        "openai" => BackendType::OpenAI,
        "zhipuai" => BackendType::ZhiPuAI,
        "minimax" => BackendType::MiniMax,
        "moonshot" => BackendType::Moonshot,
        "mistral" => BackendType::Mistral,
        "deepseek" => BackendType::DeepSeek,
        "qwen" => BackendType::Qwen,
        "groq" => BackendType::Groq,
        "local" => BackendType::Local,
        "yi" => BackendType::Yi,
        "gemini" => BackendType::Gemini,
        "baichuan" => BackendType::Baichuan,
        "stepfun" => BackendType::StepFun,
        "xai" => BackendType::XAI,
        "xiaomi" => BackendType::Xiaomi,
        "ernie" => BackendType::Ernie,
        other => {
            return Err(VvLlmError::Configuration(format!(
                "unsupported chat backend: {other}"
            )))
        }
    };

    Ok(create_chat_client(backend, model, api_base, api_key))
}
