use crate::{
    ChatRequest, ChatResponse, ChatStreamDelta, ChatTool, ChatUsage, Message, MessageContent,
    MessageRole, ToolCall, ToolChoice, VvLlmError,
};
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionTools,
    CreateChatCompletionRequestArgs, FunctionCall, FunctionObject, ImageUrl, ToolChoiceOptions,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};
use std::{pin::Pin, time::SystemTime};

use super::{ChatClient, ChatStream};

#[derive(Debug, Clone, Copy)]
struct UsageNormalizationPolicy {
    omitted_cache_read: OmittedCacheRead,
}

impl UsageNormalizationPolicy {
    const GENERIC: Self = Self {
        omitted_cache_read: OmittedCacheRead::PreserveMissing,
    };
    const MOONSHOT: Self = Self {
        omitted_cache_read: OmittedCacheRead::NormalizeToZero,
    };
}

#[derive(Debug, Clone, Copy)]
enum OmittedCacheRead {
    PreserveMissing,
    NormalizeToZero,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleChatClient {
    model: String,
    api_base: String,
    api_key: String,
    usage_policy: UsageNormalizationPolicy,
    http: reqwest::Client,
}

impl OpenAiCompatibleChatClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::with_usage_policy(model, api_base, api_key, UsageNormalizationPolicy::GENERIC)
    }

    pub(super) fn for_moonshot(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::with_usage_policy(model, api_base, api_key, UsageNormalizationPolicy::MOONSHOT)
    }

    fn with_usage_policy(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
        usage_policy: UsageNormalizationPolicy,
    ) -> Self {
        Self {
            model: model.into(),
            api_base: api_base.into(),
            api_key: api_key.into(),
            usage_policy,
            http: reqwest::Client::new(),
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
        normalize_openai_stream_chunk_json(chunk, UsageNormalizationPolicy::GENERIC)
    }

    pub fn normalize_completion_json(
        response: serde_json::Value,
    ) -> Result<ChatResponse, VvLlmError> {
        normalize_openai_completion_json(response, UsageNormalizationPolicy::GENERIC)
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
        if let Some(tool_choice) = request.tool_choice.as_ref() {
            if let ToolChoice::Mode(tool_choice) = tool_choice {
                builder.tool_choice(map_tool_choice(tool_choice)?);
            }
        } else if !request.tools.is_empty() {
            builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Auto,
            ));
        }
        builder
            .build()
            .map_err(|error| VvLlmError::Provider(error.to_string()))
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.api_base.trim_end_matches('/'))
    }

    fn request_builder(
        &self,
        request: &ChatRequest,
    ) -> Result<reqwest::RequestBuilder, VvLlmError> {
        Ok(self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(&self.to_openai_json(request)?))
    }
}

#[async_trait]
impl ChatClient for OpenAiCompatibleChatClient {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        let response = self
            .request_builder(&request)?
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let response = ensure_success(response).await?;
        let response_json = response
            .json::<Value>()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        normalize_openai_completion_json(response_json, self.usage_policy)
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let request = prepare_stream_request(request);
        let response = self
            .request_builder(&request)?
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let response = ensure_success(response).await?;
        let normalizer = TaggedReasoningNormalizer::for_model(&request.model);
        Ok(openai_sse_stream(
            response.bytes_stream(),
            normalizer,
            self.usage_policy,
        ))
    }
}

async fn ensure_success(response: reqwest::Response) -> Result<reqwest::Response, VvLlmError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(openai_http_error(response).await)
    }
}

async fn openai_http_error(response: reqwest::Response) -> VvLlmError {
    let status = response.status();
    let headers = response.headers().clone();
    let retry_after = crate::utilities::parse_retry_after_headers(&headers, SystemTime::now());
    let body = response.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);
    let error_value = parsed.get("error").unwrap_or(&parsed);
    let message = error_value
        .get("message")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("OpenAI-compatible HTTP {status}")
            } else {
                body.clone()
            }
        });
    let provider_code = error_value
        .get("code")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);

    let mut error = VvLlmError::from_status_with_retry_after(status.as_u16(), message, retry_after);
    if let VvLlmError::Classified(details) = &mut error {
        details.provider_code = provider_code;
        details.request_id = request_id;
    }
    error
}

enum OpenAiSseEvent {
    Ignore,
    Done,
    Data(Value),
}

fn openai_sse_stream<S>(
    bytes: S,
    normalizer: TaggedReasoningNormalizer,
    usage_policy: UsageNormalizationPolicy,
) -> ChatStream
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
    let state: (ByteStream, Vec<u8>, TaggedReasoningNormalizer) =
        (Box::pin(bytes), Vec::new(), normalizer);
    Box::pin(stream::unfold(
        state,
        move |(mut bytes, mut buffer, mut normalizer)| async move {
            loop {
                if let Some(event) = take_sse_event(&mut buffer) {
                    match parse_openai_sse_event(&event) {
                        Ok(OpenAiSseEvent::Ignore) => continue,
                        Ok(OpenAiSseEvent::Done) => return None,
                        Ok(OpenAiSseEvent::Data(chunk)) => {
                            let delta = normalize_openai_stream_chunk_json(chunk, usage_policy)
                                .map(|delta| normalizer.normalize(delta));
                            return Some((delta, (bytes, buffer, normalizer)));
                        }
                        Err(error) => return Some((Err(error), (bytes, buffer, normalizer))),
                    }
                }

                match bytes.next().await {
                    Some(Ok(chunk)) => buffer.extend_from_slice(&chunk),
                    Some(Err(error)) => {
                        return Some((
                            Err(VvLlmError::Provider(error.to_string())),
                            (bytes, buffer, normalizer),
                        ))
                    }
                    None if buffer.is_empty() => return None,
                    None => {
                        let event = std::mem::take(&mut buffer);
                        match parse_openai_sse_event(&event) {
                            Ok(OpenAiSseEvent::Ignore | OpenAiSseEvent::Done) => return None,
                            Ok(OpenAiSseEvent::Data(chunk)) => {
                                let delta = normalize_openai_stream_chunk_json(chunk, usage_policy)
                                    .map(|delta| normalizer.normalize(delta));
                                return Some((delta, (bytes, buffer, normalizer)));
                            }
                            Err(error) => return Some((Err(error), (bytes, buffer, normalizer))),
                        }
                    }
                }
            }
        },
    ))
}

fn take_sse_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let (index, delimiter_len) =
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            (index, 4)
        } else {
            let index = buffer.windows(2).position(|window| window == b"\n\n")?;
            (index, 2)
        };
    let event = buffer.drain(..index).collect();
    buffer.drain(..delimiter_len);
    Some(event)
}

fn parse_openai_sse_event(event: &[u8]) -> Result<OpenAiSseEvent, VvLlmError> {
    let text = String::from_utf8_lossy(event);
    let mut data = String::new();
    for line in text.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    if data.trim().is_empty() {
        return Ok(OpenAiSseEvent::Ignore);
    }
    if data.trim() == "[DONE]" {
        return Ok(OpenAiSseEvent::Done);
    }
    Ok(OpenAiSseEvent::Data(serde_json::from_str(&data)?))
}

fn normalize_openai_stream_chunk_json(
    mut chunk: Value,
    usage_policy: UsageNormalizationPolicy,
) -> Result<ChatStreamDelta, VvLlmError> {
    let extra_content = stream_tool_call_extra_content(&chunk);
    let reasoning_content = stream_reasoning_content(&chunk);
    let usage =
        take_openai_usage(&mut chunk).and_then(|usage| normalize_openai_usage(usage, usage_policy));
    let chunk: async_openai::types::chat::CreateChatCompletionStreamResponse =
        serde_json::from_value(chunk)?;
    let mut delta = normalize_openai_stream_chunk(chunk);
    delta.usage = usage;
    delta.reasoning_content.push_str(&reasoning_content);
    apply_stream_tool_call_extra_content(&mut delta, extra_content);
    Ok(delta)
}

fn prepare_stream_request(mut request: ChatRequest) -> ChatRequest {
    request.options.stream = Some(true);
    if request.options.stream_options.is_none() {
        request.options.stream_options = Some(json!({"include_usage": true}));
    }
    request
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
                content: if message.content.is_empty() && !message.tool_calls.is_empty() {
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
        if let MessageContent::Text { text, .. } = &content[0] {
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
        if let MessageContent::Text { text, .. } = &content[0] {
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
        MessageContent::Text { text, .. } => {
            Ok(ChatCompletionRequestSystemMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: text.clone() },
            ))
        }
        MessageContent::ImageUrl { .. } => Err(VvLlmError::Configuration(
            "system messages cannot contain image parts".to_string(),
        )),
    }
}

fn to_openai_user_part(
    content: &MessageContent,
) -> Result<ChatCompletionRequestUserMessageContentPart, VvLlmError> {
    match content {
        MessageContent::Text { text, .. } => Ok(ChatCompletionRequestUserMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: text.clone() },
        )),
        MessageContent::ImageUrl { url, .. } => {
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
            index: None,
            extra_content: None,
            extensions: crate::JsonExtensions::new(),
        }),
        ChatCompletionMessageToolCalls::Custom(_) => None,
    }
}

fn normalize_openai_completion_json(
    mut response: Value,
    usage_policy: UsageNormalizationPolicy,
) -> Result<ChatResponse, VvLlmError> {
    let reasoning_content = completion_reasoning_content(&response);
    let extra_content = completion_tool_call_extra_content(&response);
    let usage = take_openai_usage(&mut response)
        .and_then(|usage| normalize_openai_usage(usage, usage_policy));
    let response: async_openai::types::chat::CreateChatCompletionResponse =
        serde_json::from_value(response)?;
    let mut normalized = normalize_openai_completion_response(response);
    normalized.reasoning_content = reasoning_content;
    normalized.usage = usage;
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
    let usage = response.usage.map(normalize_typed_openai_usage);

    ChatResponse {
        id: response.id,
        model: response.model,
        content,
        tool_calls,
        reasoning_content: None,
        usage,
    }
}

fn take_openai_usage(response: &mut Value) -> Option<Value> {
    response
        .as_object_mut()
        .and_then(|response| response.remove("usage"))
        .filter(|usage| !usage.is_null())
}

fn normalize_openai_usage(
    raw_usage: Value,
    usage_policy: UsageNormalizationPolicy,
) -> Option<ChatUsage> {
    if raw_usage.is_null() {
        return None;
    }

    let prompt_tokens = value_u32(raw_usage.get("prompt_tokens"))
        .or_else(|| value_u32(raw_usage.get("input_tokens")));
    let completion_tokens = value_u32(raw_usage.get("completion_tokens"))
        .or_else(|| value_u32(raw_usage.get("output_tokens")));
    let input_tokens = value_u32(raw_usage.get("input_tokens")).or(prompt_tokens);
    let output_tokens = value_u32(raw_usage.get("output_tokens")).or(completion_tokens);
    let cache_read_input_tokens = normalize_cache_read_input_tokens(&raw_usage, usage_policy);
    let cache_creation_input_tokens = first_nested_u32(
        &raw_usage,
        &[
            &["input_tokens_details", "cache_creation_tokens"],
            &["prompt_tokens_details", "cache_creation_tokens"],
            &["cache_creation_input_tokens"],
            &["cache_write_input_tokens"],
            &["cache_creation_tokens"],
            &["cache_write_tokens"],
        ],
    );

    Some(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: value_u32(raw_usage.get("total_tokens")),
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        raw_usage: Some(raw_usage),
    })
}

fn normalize_typed_openai_usage(usage: async_openai::types::chat::CompletionUsage) -> ChatUsage {
    let prompt_tokens = usage.prompt_tokens;
    let completion_tokens = usage.completion_tokens;
    let total_tokens = usage.total_tokens;
    let cache_read_input_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens);
    let raw_usage = serde_json::to_value(&usage).ok();

    ChatUsage {
        prompt_tokens: Some(prompt_tokens),
        completion_tokens: Some(completion_tokens),
        total_tokens: Some(total_tokens),
        input_tokens: Some(prompt_tokens),
        output_tokens: Some(completion_tokens),
        cache_read_input_tokens,
        cache_creation_input_tokens: None,
        raw_usage,
    }
}

fn first_nested_u32(value: &Value, paths: &[&[&str]]) -> Option<u32> {
    paths
        .iter()
        .find_map(|path| nested_value(value, path).and_then(|value| value_u32(Some(value))))
}

fn normalize_cache_read_input_tokens(
    raw_usage: &Value,
    usage_policy: UsageNormalizationPolicy,
) -> Option<u32> {
    const CACHE_READ_PATHS: &[&[&str]] = &[
        &["prompt_tokens_details", "cached_tokens"],
        &["input_tokens_details", "cached_tokens"],
        &["cached_tokens"],
        &["cache_read_input_tokens"],
        &["cache_read_tokens"],
    ];

    if !raw_usage.is_object() {
        return None;
    }

    let mut field_present = false;
    for path in CACHE_READ_PATHS {
        if let Some(value) = nested_value(raw_usage, path) {
            field_present = true;
            if let Some(value) = value_u32(Some(value)) {
                return Some(value);
            }
        }
    }

    if field_present {
        return None;
    }

    match usage_policy.omitted_cache_read {
        OmittedCacheRead::PreserveMissing => None,
        OmittedCacheRead::NormalizeToZero => Some(0),
    }
}

fn nested_value<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
}

fn value_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
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

fn stream_reasoning_content(chunk: &Value) -> String {
    chunk
        .pointer("/choices/0/delta/reasoning_content")
        .or_else(|| chunk.pointer("/choices/0/delta/reasoning"))
        .and_then(extract_reasoning_content)
        .unwrap_or_default()
}

fn extract_reasoning_content(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Object(object) => [
            "reasoning_content",
            "reasoning",
            "thinking",
            "text",
            "content",
        ]
        .iter()
        .find_map(|key| object.get(*key).and_then(extract_reasoning_content)),
        Value::Array(items) => {
            let content = items
                .iter()
                .filter_map(extract_reasoning_content)
                .collect::<String>();
            (!content.is_empty()).then_some(content)
        }
        _ => None,
    }
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
        usage: chunk.usage.map(normalize_typed_openai_usage),
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
                    index: Some(tool_call.index as usize),
                    extra_content: None,
                    extensions: crate::JsonExtensions::new(),
                });
            }
        }
    }

    TaggedReasoningNormalizer::for_model(&model).normalize(delta)
}

fn merge_openai_request_extensions(json: &mut Value, request: &ChatRequest) {
    merge_openai_option_extensions(json, &request.options);
    merge_extra_body(json, &request.extra_body);
    merge_object_extensions(json, &request.extensions);
    if let Some(ToolChoice::Object(tool_choice)) = request.tool_choice.as_ref() {
        if let Some(target) = json.as_object_mut() {
            target.insert(
                "tool_choice".to_string(),
                Value::Object(tool_choice.clone()),
            );
        }
    }
    if let Some(tools) = json.get_mut("tools").and_then(Value::as_array_mut) {
        for (payload_tool, tool) in tools.iter_mut().zip(&request.tools) {
            if let Some(object) = payload_tool.as_object_mut() {
                merge_map_extensions(object, &tool.extensions);
            }
        }
    }
    let Some(messages) = json.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for (payload, message) in messages.iter_mut().zip(&request.messages) {
        merge_message_extensions(payload, message);
    }
}

fn merge_openai_option_extensions(json: &mut Value, options: &crate::ChatRequestOptions) {
    let Some(target) = json.as_object_mut() else {
        return;
    };
    if let Some(max_completion_tokens) = options.max_completion_tokens {
        target.insert(
            "max_completion_tokens".to_string(),
            json!(max_completion_tokens),
        );
    }
    if let Some(max_tokens_details) = &options.max_tokens_details {
        target.insert("max_tokens_details".to_string(), max_tokens_details.clone());
    }
    if let Some(top_p) = options.top_p {
        target.insert("top_p".to_string(), json!(top_p));
    }
    if !options.stop.is_empty() {
        target.insert("stop".to_string(), json!(options.stop));
    }
    if let Some(response_format) = &options.response_format {
        target.insert("response_format".to_string(), response_format.clone());
    }
    if let Some(stream_options) = &options.stream_options {
        target.insert("stream_options".to_string(), stream_options.clone());
    }
    if let Some(audio) = &options.audio {
        target.insert("audio".to_string(), audio.clone());
    }
    if let Some(frequency_penalty) = options.frequency_penalty {
        target.insert("frequency_penalty".to_string(), json!(frequency_penalty));
    }
    if let Some(logit_bias) = &options.logit_bias {
        target.insert("logit_bias".to_string(), logit_bias.clone());
    }
    if let Some(logprobs) = options.logprobs {
        target.insert("logprobs".to_string(), json!(logprobs));
    }
    if let Some(metadata) = &options.metadata {
        target.insert("metadata".to_string(), metadata.clone());
    }
    if let Some(modalities) = &options.modalities {
        target.insert("modalities".to_string(), modalities.clone());
    }
    if let Some(n) = options.n {
        target.insert("n".to_string(), json!(n));
    }
    if let Some(parallel_tool_calls) = options.parallel_tool_calls {
        target.insert(
            "parallel_tool_calls".to_string(),
            json!(parallel_tool_calls),
        );
    }
    if let Some(prediction) = &options.prediction {
        target.insert("prediction".to_string(), prediction.clone());
    }
    if let Some(presence_penalty) = options.presence_penalty {
        target.insert("presence_penalty".to_string(), json!(presence_penalty));
    }
    if let Some(reasoning_effort) = &options.reasoning_effort {
        target.insert("reasoning_effort".to_string(), json!(reasoning_effort));
    }
    if let Some(thinking) = &options.thinking {
        target.insert("thinking".to_string(), thinking.clone());
    }
    if let Some(seed) = options.seed {
        target.insert("seed".to_string(), json!(seed));
    }
    if let Some(service_tier) = &options.service_tier {
        target.insert("service_tier".to_string(), json!(service_tier));
    }
    if let Some(store) = options.store {
        target.insert("store".to_string(), json!(store));
    }
    if let Some(top_logprobs) = options.top_logprobs {
        target.insert("top_logprobs".to_string(), json!(top_logprobs));
    }
    if let Some(user) = &options.user {
        target.insert("user".to_string(), json!(user));
    }
    merge_map_extensions(target, &options.extensions);
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
    if let Some(object) = payload.as_object_mut() {
        merge_map_extensions(object, &message.extensions);
    }
    if let Some(reasoning_content) = message.reasoning_content.as_deref() {
        if let Some(object) = payload.as_object_mut() {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content.to_string()),
            );
        }
    }

    if let Some(content) = payload.get_mut("content") {
        merge_message_content_extensions(content, &message.content);
    }

    if let Some(tool_calls) = payload.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for (payload_tool_call, tool_call) in tool_calls.iter_mut().zip(&message.tool_calls) {
            if let Some(object) = payload_tool_call.as_object_mut() {
                if let Some(extra_content) = &tool_call.extra_content {
                    object.insert("extra_content".to_string(), extra_content.clone());
                }
                merge_map_extensions(object, &tool_call.extensions);
            }
        }
    }
}

fn merge_message_content_extensions(payload: &mut Value, content: &[MessageContent]) {
    if content.len() == 1 {
        if let MessageContent::Text {
            text,
            cache_control,
            extensions,
        } = &content[0]
        {
            if cache_control.is_some() || !extensions.is_empty() {
                *payload = json!({"type": "text", "text": text});
                if let Some(cache_control) = cache_control {
                    payload["cache_control"] = cache_control.clone();
                }
                if let Some(object) = payload.as_object_mut() {
                    merge_map_extensions(object, extensions);
                }
            }
        }
    }

    let Some(parts) = payload.as_array_mut() else {
        return;
    };
    for (part, source) in parts.iter_mut().zip(content) {
        let Some(object) = part.as_object_mut() else {
            continue;
        };
        match source {
            MessageContent::Text {
                cache_control,
                extensions,
                ..
            } => {
                if let Some(cache_control) = cache_control {
                    object.insert("cache_control".to_string(), cache_control.clone());
                }
                merge_map_extensions(object, extensions);
            }
            MessageContent::ImageUrl {
                detail,
                cache_control,
                extensions,
                nested_extensions,
                ..
            } => {
                if let Some(image_url) = object.get_mut("image_url").and_then(Value::as_object_mut)
                {
                    if let Some(detail) = detail {
                        image_url.insert("detail".to_string(), Value::String(detail.clone()));
                    }
                    merge_map_extensions(image_url, nested_extensions);
                }
                if let Some(cache_control) = cache_control {
                    object.insert("cache_control".to_string(), cache_control.clone());
                }
                merge_map_extensions(object, extensions);
            }
        }
    }
}

fn merge_object_extensions(json: &mut Value, extensions: &crate::JsonExtensions) {
    if let Some(target) = json.as_object_mut() {
        merge_map_extensions(target, extensions);
    }
}

fn merge_map_extensions(
    target: &mut serde_json::Map<String, Value>,
    extensions: &crate::JsonExtensions,
) {
    for (key, value) in extensions {
        target.insert(key.clone(), value.clone());
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
