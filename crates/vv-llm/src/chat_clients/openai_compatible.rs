use crate::{
    ChatRequest, ChatResponse, ChatUsage, Message, MessageContent, MessageRole, VvLlmError,
};
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_trait::async_trait;

use super::ChatClient;

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleChatClient {
    model: String,
    api_base: String,
    api_key: String,
}

impl OpenAiCompatibleChatClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_base: api_base.into(),
            api_key: api_key.into(),
        }
    }

    pub fn to_openai_json(&self, request: &ChatRequest) -> Result<serde_json::Value, VvLlmError> {
        let request = self.to_openai_request(request)?;
        Ok(serde_json::to_value(request)?)
    }

    fn to_openai_request(
        &self,
        request: &ChatRequest,
    ) -> Result<async_openai::types::chat::CreateChatCompletionRequest, VvLlmError> {
        let messages = request
            .messages
            .iter()
            .map(to_openai_message)
            .collect::<Result<Vec<_>, _>>()?;
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(if request.model.is_empty() {
            self.model.clone()
        } else {
            request.model.clone()
        });
        builder.messages(messages);
        if let Some(temperature) = request.options.temperature {
            builder.temperature(temperature);
        }
        if let Some(max_tokens) = request.options.max_tokens {
            builder.max_tokens(max_tokens);
        }
        if let Some(stream) = request.options.stream {
            builder.stream(stream);
        }
        builder
            .build()
            .map_err(|error| VvLlmError::Provider(error.to_string()))
    }

    fn client(&self) -> Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_key(self.api_key.clone())
            .with_api_base(self.api_base.clone());
        Client::with_config(config)
    }
}

#[async_trait]
impl ChatClient for OpenAiCompatibleChatClient {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        let response = self
            .client()
            .chat()
            .create(self.to_openai_request(&request)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();
        let usage = response.usage.map(|usage| ChatUsage {
            prompt_tokens: Some(usage.prompt_tokens),
            completion_tokens: Some(usage.completion_tokens),
            total_tokens: Some(usage.total_tokens),
        });

        Ok(ChatResponse {
            id: response.id,
            model: response.model,
            content,
            usage,
        })
    }
}

fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, VvLlmError> {
    let text = message.text_content().unwrap_or_default();
    let name = message.name.clone();

    match message.role {
        MessageRole::System => Ok(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(text),
                name,
            },
        )),
        MessageRole::User => Ok(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name,
            },
        )),
        MessageRole::Assistant => Ok(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(text)),
                name,
                ..Default::default()
            },
        )),
        MessageRole::Tool => Ok(ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text(text),
                tool_call_id: message
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| "tool-call".to_string()),
            },
        )),
    }
}

#[allow(dead_code)]
fn _uses_message_content_type(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Text { .. } | MessageContent::ImageUrl { .. }
    )
}
