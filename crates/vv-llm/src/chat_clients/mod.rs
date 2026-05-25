mod anthropic;
mod openai_compatible;

use crate::{BackendType, ChatRequest, VvLlmError};
use async_trait::async_trait;

pub use anthropic::AnthropicChatClient;
pub use openai_compatible::OpenAiCompatibleChatClient;

#[async_trait]
pub trait ChatClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn create_completion(
        &self,
        request: ChatRequest,
    ) -> Result<crate::ChatResponse, VvLlmError>;
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
