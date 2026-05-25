use crate::{ChatRequest, ChatResponse, ChatUsage, Message, MessageRole, VvLlmError};
use anthropic::{
    client::{Client, ClientBuilder},
    types::{
        ContentBlock, Message as AnthropicMessage, MessagesRequest, MessagesRequestBuilder, Role,
    },
};
use async_trait::async_trait;

use super::ChatClient;

#[derive(Debug, Clone)]
pub struct AnthropicChatClient {
    model: String,
    api_base: String,
    api_key: String,
}

impl AnthropicChatClient {
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

    pub fn to_anthropic_json(
        &self,
        request: &ChatRequest,
    ) -> Result<serde_json::Value, VvLlmError> {
        Ok(serde_json::to_value(self.to_anthropic_request(request)?)?)
    }

    fn to_anthropic_request(&self, request: &ChatRequest) -> Result<MessagesRequest, VvLlmError> {
        let mut system = Vec::new();
        let mut messages = Vec::new();

        for message in &request.messages {
            match message.role {
                MessageRole::System => {
                    if let Some(text) = message.text_content() {
                        system.push(text);
                    }
                }
                MessageRole::User | MessageRole::Tool => {
                    messages.push(to_anthropic_message(message, Role::User))
                }
                MessageRole::Assistant => {
                    messages.push(to_anthropic_message(message, Role::Assistant))
                }
            }
        }

        let mut builder = MessagesRequestBuilder::default();
        builder.model(if request.model.is_empty() {
            self.model.clone()
        } else {
            request.model.clone()
        });
        builder.messages(messages);
        builder.system(system.join("\n"));
        builder.max_tokens(request.options.max_tokens.unwrap_or(1024) as usize);
        builder.stream(request.options.stream.unwrap_or(false));
        if let Some(temperature) = request.options.temperature {
            builder.temperature(temperature as f64);
        }
        builder
            .build()
            .map_err(|error| VvLlmError::Provider(error.to_string()))
    }

    fn client(&self) -> Result<Client, VvLlmError> {
        ClientBuilder::default()
            .api_key(self.api_key.clone())
            .api_base(self.api_base.clone())
            .default_model(self.model.clone())
            .build()
            .map_err(|error| VvLlmError::Provider(error.to_string()))
    }
}

#[async_trait]
impl ChatClient for AnthropicChatClient {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        let response = self
            .client()?
            .messages(self.to_anthropic_request(&request)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let content = response
            .content
            .into_iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text),
                ContentBlock::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let usage = response.usage;
        let usage = Some(ChatUsage {
            prompt_tokens: Some(usage.input_tokens as u32),
            completion_tokens: Some(usage.output_tokens as u32),
            total_tokens: Some((usage.input_tokens + usage.output_tokens) as u32),
        });

        Ok(ChatResponse {
            id: response.id,
            model: response.model,
            content,
            usage,
        })
    }
}

fn to_anthropic_message(message: &Message, role: Role) -> AnthropicMessage {
    AnthropicMessage {
        role,
        content: vec![ContentBlock::Text {
            text: message.text_content().unwrap_or_default(),
        }],
    }
}
