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
use serde_json::Value;

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
        let openai_request = self.to_openai_request(request)?;
        let mut json = serde_json::to_value(openai_request)?;
        merge_openai_request_extensions(&mut json, request);
        Ok(json)
    }

    pub fn normalize_stream_chunk_json(
        chunk: serde_json::Value,
    ) -> Result<ChatStreamDelta, VvLlmError> {
        let extra_content = stream_tool_call_extra_content(&chunk);
        let chunk: async_openai::types::chat::CreateChatCompletionStreamResponse =
            serde_json::from_value(chunk)?;
        let mut delta = normalize_openai_stream_chunk(chunk);
        apply_stream_tool_call_extra_content(&mut delta, extra_content);
        Ok(delta)
    }

    pub fn normalize_completion_json(
        response: serde_json::Value,
    ) -> Result<ChatResponse, VvLlmError> {
        normalize_openai_completion_json(response)
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
        let client = self.client();
        if request_needs_byot(&request) {
            let request_json = self.to_openai_json(&request)?;
            let response_json: Value = client
                .chat()
                .create_byot(&request_json)
                .await
                .map_err(|error| VvLlmError::Provider(error.to_string()))?;
            return normalize_openai_completion_json(response_json);
        }
        let response = client
            .chat()
            .create(self.to_openai_request(&request)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        Ok(normalize_openai_completion_response(response))
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let client = self.client();
        if request_needs_byot(&request) {
            let request_json = self.to_openai_json(&request)?;
            let stream: std::pin::Pin<
                Box<
                    dyn futures_core::Stream<Item = Result<Value, async_openai::error::OpenAIError>>
                        + Send,
                >,
            > = client
                .chat()
                .create_stream_byot(&request_json)
                .await
                .map_err(|error| VvLlmError::Provider(error.to_string()))?;
            let mut normalizer = TaggedReasoningNormalizer::for_model(&request.model);
            return Ok(Box::pin(stream.map(move |chunk| {
                chunk
                    .map_err(|error| VvLlmError::Provider(error.to_string()))
                    .and_then(OpenAiCompatibleChatClient::normalize_stream_chunk_json)
                    .map(|delta| normalizer.normalize(delta))
            })));
        }
        let stream = client
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
            extra_content: None,
        }),
        ChatCompletionMessageToolCalls::Custom(_) => None,
    }
}

fn normalize_openai_completion_json(response: Value) -> Result<ChatResponse, VvLlmError> {
    let reasoning_content = completion_reasoning_content(&response);
    let extra_content = completion_tool_call_extra_content(&response);
    let response: async_openai::types::chat::CreateChatCompletionResponse =
        serde_json::from_value(response)?;
    let mut normalized = normalize_openai_completion_response(response);
    normalized.reasoning_content = reasoning_content;
    apply_tool_call_extra_content(&mut normalized.tool_calls, extra_content);
    Ok(normalized)
}

fn normalize_openai_completion_response(
    response: async_openai::types::chat::CreateChatCompletionResponse,
) -> ChatResponse {
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

    ChatResponse {
        id: response.id,
        model: response.model,
        content,
        tool_calls,
        reasoning_content: None,
        usage,
    }
}

fn completion_reasoning_content(response: &Value) -> Option<String> {
    response
        .pointer("/choices/0/message/reasoning_content")
        .or_else(|| response.pointer("/choices/0/message/reasoning"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn completion_tool_call_extra_content(response: &Value) -> Vec<Option<Value>> {
    response
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|tool_calls| tool_call_extra_content(tool_calls))
        .unwrap_or_default()
}

fn stream_tool_call_extra_content(chunk: &Value) -> Vec<Option<Value>> {
    chunk
        .pointer("/choices/0/delta/tool_calls")
        .and_then(Value::as_array)
        .map(|tool_calls| tool_call_extra_content(tool_calls))
        .unwrap_or_default()
}

fn tool_call_extra_content(tool_calls: &[Value]) -> Vec<Option<Value>> {
    tool_calls
        .iter()
        .map(|tool_call| tool_call.get("extra_content").cloned())
        .collect()
}

fn apply_tool_call_extra_content(tool_calls: &mut [ToolCall], extra_content: Vec<Option<Value>>) {
    for (tool_call, extra_content) in tool_calls.iter_mut().zip(extra_content) {
        tool_call.extra_content = extra_content;
    }
}

fn apply_stream_tool_call_extra_content(
    delta: &mut ChatStreamDelta,
    extra_content: Vec<Option<Value>>,
) {
    apply_tool_call_extra_content(&mut delta.tool_calls, extra_content);
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
                    extra_content: None,
                });
            }
        }
    }

    TaggedReasoningNormalizer::for_model(&model).normalize(delta)
}

fn request_needs_byot(request: &ChatRequest) -> bool {
    !is_empty_extra_body(&request.extra_body) || request.messages.iter().any(message_needs_byot)
}

fn message_needs_byot(message: &Message) -> bool {
    message
        .reasoning_content
        .as_deref()
        .map(|value| !value.is_empty())
        .unwrap_or(false)
        || message
            .tool_calls
            .iter()
            .any(|tool_call| tool_call.extra_content.is_some())
}

fn merge_openai_request_extensions(json: &mut Value, request: &ChatRequest) {
    merge_extra_body(json, &request.extra_body);
    let Some(messages) = json.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for (payload, message) in messages.iter_mut().zip(&request.messages) {
        merge_message_extensions(payload, message);
    }
}

fn merge_extra_body(json: &mut Value, extra_body: &Value) {
    if is_empty_extra_body(extra_body) {
        return;
    }
    let Some(target) = json.as_object_mut() else {
        return;
    };
    if let Some(object) = extra_body.as_object() {
        for (key, value) in object {
            target.insert(key.clone(), value.clone());
        }
    }
}

fn merge_message_extensions(payload: &mut Value, message: &Message) {
    if let Some(reasoning_content) = message
        .reasoning_content
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        if let Some(object) = payload.as_object_mut() {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content.to_string()),
            );
        }
    }

    let Some(tool_calls) = payload.get_mut("tool_calls").and_then(Value::as_array_mut) else {
        return;
    };
    for (payload_tool_call, tool_call) in tool_calls.iter_mut().zip(&message.tool_calls) {
        let Some(extra_content) = &tool_call.extra_content else {
            continue;
        };
        if let Some(object) = payload_tool_call.as_object_mut() {
            object.insert("extra_content".to_string(), extra_content.clone());
        }
    }
}

fn is_empty_extra_body(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(object) => object.is_empty(),
        _ => false,
    }
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
