use crate::{
    ChatRequest, ChatResponse, ChatStreamDelta, ChatTool, ChatUsage, Message, MessageContent,
    MessageRole, ToolCall, VvLlmError,
};
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessage,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionTools,
        CreateChatCompletionRequestArgs, FunctionCall, FunctionObject, ImageUrl, ToolChoiceOptions,
    },
    Client,
};
use async_trait::async_trait;
use futures_util::StreamExt;

use super::{ChatClient, ChatStream};

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

    pub fn normalize_stream_chunk_json(
        chunk: serde_json::Value,
    ) -> Result<ChatStreamDelta, VvLlmError> {
        let chunk: async_openai::types::chat::CreateChatCompletionStreamResponse =
            serde_json::from_value(chunk)?;
        Ok(normalize_openai_stream_chunk(chunk))
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
        if !request.tools.is_empty() {
            builder.tools(
                request
                    .tools
                    .iter()
                    .cloned()
                    .map(to_openai_tool)
                    .collect::<Vec<_>>(),
            );
        }
        if let Some(tool_choice) = request.tool_choice.as_deref() {
            builder.tool_choice(map_tool_choice(tool_choice)?);
        } else if !request.tools.is_empty() {
            builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Auto,
            ));
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

        let first_choice = response.choices.first();
        let content = first_choice
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();
        let tool_calls = first_choice
            .and_then(|choice| choice.message.tool_calls.clone())
            .unwrap_or_default()
            .into_iter()
            .filter_map(from_openai_tool_call)
            .collect();
        let usage = response.usage.map(|usage| ChatUsage {
            prompt_tokens: Some(usage.prompt_tokens),
            completion_tokens: Some(usage.completion_tokens),
            total_tokens: Some(usage.total_tokens),
        });

        Ok(ChatResponse {
            id: response.id,
            model: response.model,
            content,
            tool_calls,
            usage,
        })
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let stream = self
            .client()
            .chat()
            .create_stream(self.to_openai_request(&request)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let mut normalizer = TaggedReasoningNormalizer::for_model(&request.model);
        Ok(Box::pin(stream.map(move |chunk| {
            chunk
                .map(normalize_openai_stream_chunk)
                .map(|delta| normalizer.normalize(delta))
                .map_err(|error| VvLlmError::Provider(error.to_string()))
        })))
    }
}

fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, VvLlmError> {
    let name = message.name.clone();

    match message.role {
        MessageRole::System => Ok(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: to_openai_text_or_parts(&message.content)?,
                name,
            },
        )),
        MessageRole::User => Ok(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: to_openai_user_content(&message.content)?,
                name,
            },
        )),
        MessageRole::Assistant => Ok(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content: if message.content.is_empty() {
                    None
                } else {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(
                        message.text_content().unwrap_or_default(),
                    ))
                },
                name,
                tool_calls: if message.tool_calls.is_empty() {
                    None
                } else {
                    Some(message.tool_calls.iter().map(to_openai_tool_call).collect())
                },
                ..Default::default()
            },
        )),
        MessageRole::Tool => Ok(ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text(
                    message.text_content().unwrap_or_default(),
                ),
                tool_call_id: message
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| "tool-call".to_string()),
            },
        )),
    }
}

fn to_openai_text_or_parts(
    content: &[MessageContent],
) -> Result<ChatCompletionRequestSystemMessageContent, VvLlmError> {
    if content.len() == 1 {
        if let MessageContent::Text { text } = &content[0] {
            return Ok(ChatCompletionRequestSystemMessageContent::Text(
                text.clone(),
            ));
        }
    }
    Ok(ChatCompletionRequestSystemMessageContent::Array(
        content
            .iter()
            .map(to_openai_system_part)
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn to_openai_user_content(
    content: &[MessageContent],
) -> Result<ChatCompletionRequestUserMessageContent, VvLlmError> {
    if content.len() == 1 {
        if let MessageContent::Text { text } = &content[0] {
            return Ok(ChatCompletionRequestUserMessageContent::Text(text.clone()));
        }
    }
    Ok(ChatCompletionRequestUserMessageContent::Array(
        content
            .iter()
            .map(to_openai_user_part)
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn to_openai_system_part(
    content: &MessageContent,
) -> Result<ChatCompletionRequestSystemMessageContentPart, VvLlmError> {
    match content {
        MessageContent::Text { text } => Ok(ChatCompletionRequestSystemMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: text.clone() },
        )),
        MessageContent::ImageUrl { .. } => Err(VvLlmError::Configuration(
            "system messages cannot contain image parts".to_string(),
        )),
    }
}

fn to_openai_user_part(
    content: &MessageContent,
) -> Result<ChatCompletionRequestUserMessageContentPart, VvLlmError> {
    match content {
        MessageContent::Text { text } => Ok(ChatCompletionRequestUserMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: text.clone() },
        )),
        MessageContent::ImageUrl { url } => {
            Ok(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageUrl {
                        url: url.clone(),
                        detail: None,
                    },
                },
            ))
        }
    }
}

fn to_openai_tool(tool: ChatTool) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: tool.name,
            description: tool.description,
            parameters: Some(tool.parameters),
            strict: None,
        },
    })
}

fn to_openai_tool_call(tool_call: &ToolCall) -> ChatCompletionMessageToolCalls {
    ChatCompletionMessageToolCalls::Function(
        async_openai::types::chat::ChatCompletionMessageToolCall {
            id: tool_call.id.clone(),
            function: FunctionCall {
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
            },
        },
    )
}

fn from_openai_tool_call(tool_call: ChatCompletionMessageToolCalls) -> Option<ToolCall> {
    match tool_call {
        ChatCompletionMessageToolCalls::Function(function_call) => Some(ToolCall {
            id: function_call.id,
            name: function_call.function.name,
            arguments: function_call.function.arguments,
        }),
        ChatCompletionMessageToolCalls::Custom(_) => None,
    }
}

fn normalize_openai_stream_chunk(
    chunk: async_openai::types::chat::CreateChatCompletionStreamResponse,
) -> ChatStreamDelta {
    let model = chunk.model.clone();
    let mut delta = ChatStreamDelta {
        usage: chunk.usage.map(|usage| ChatUsage {
            prompt_tokens: Some(usage.prompt_tokens),
            completion_tokens: Some(usage.completion_tokens),
            total_tokens: Some(usage.total_tokens),
        }),
        ..Default::default()
    };

    for choice in chunk.choices {
        if choice.finish_reason.is_some() {
            delta.done = true;
        }
        if let Some(content) = choice.delta.content {
            delta.content.push_str(&content);
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                let function = tool_call.function;
                delta.tool_calls.push(ToolCall {
                    id: tool_call.id.unwrap_or_default(),
                    name: function
                        .as_ref()
                        .and_then(|function| function.name.clone())
                        .unwrap_or_default(),
                    arguments: function
                        .and_then(|function| function.arguments)
                        .unwrap_or_default(),
                });
            }
        }
    }

    TaggedReasoningNormalizer::for_model(&model).normalize(delta)
}

#[derive(Debug, Clone)]
struct TaggedReasoningNormalizer {
    start_tag: &'static str,
    end_tag: &'static str,
    in_reasoning: bool,
}

impl TaggedReasoningNormalizer {
    fn for_model(model: &str) -> Self {
        if model.starts_with("gemini-3") {
            Self {
                start_tag: "<thought>",
                end_tag: "</thought>",
                in_reasoning: false,
            }
        } else {
            Self {
                start_tag: "<think>",
                end_tag: "</think>",
                in_reasoning: false,
            }
        }
    }

    fn normalize(&mut self, mut delta: ChatStreamDelta) -> ChatStreamDelta {
        if delta.content.is_empty() {
            return delta;
        }

        let mut input = std::mem::take(&mut delta.content);
        let mut output = String::new();
        let mut reasoning = String::new();

        while !input.is_empty() {
            if self.in_reasoning {
                if let Some(end) = input.find(self.end_tag) {
                    reasoning.push_str(&input[..end]);
                    input = input[end + self.end_tag.len()..].to_string();
                    self.in_reasoning = false;
                } else {
                    reasoning.push_str(&input);
                    input.clear();
                }
            } else if let Some(start) = input.find(self.start_tag) {
                output.push_str(&input[..start]);
                input = input[start + self.start_tag.len()..].to_string();
                self.in_reasoning = true;
            } else {
                output.push_str(&input);
                input.clear();
            }
        }

        delta.content = output;
        delta.reasoning_content.push_str(&reasoning);
        delta
    }
}

fn map_tool_choice(choice: &str) -> Result<ChatCompletionToolChoiceOption, VvLlmError> {
    match choice {
        "auto" => Ok(ChatCompletionToolChoiceOption::Mode(
            ToolChoiceOptions::Auto,
        )),
        "none" => Ok(ChatCompletionToolChoiceOption::Mode(
            ToolChoiceOptions::None,
        )),
        "required" => Ok(ChatCompletionToolChoiceOption::Mode(
            ToolChoiceOptions::Required,
        )),
        _ => Err(VvLlmError::Configuration(format!(
            "unsupported tool_choice value: {choice}"
        ))),
    }
}

#[allow(dead_code)]
fn _uses_message_content_type(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Text { .. } | MessageContent::ImageUrl { .. }
    )
}
