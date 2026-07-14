use crate::{
    ChatRequest, ChatResponse, ChatStreamDelta, ChatTool, ChatUsage, Message, MessageContent,
    MessageRole, ToolCall, VvLlmError,
};
use anthropic::client::{Client, ClientBuilder};
use anthropic::types::{
    ContentBlock as AnthropicContentBlock, ContentBlockDelta as AnthropicContentBlockDelta,
    Message as AnthropicMessage, MessagesRequest, MessagesRequestBuilder,
    MessagesStreamEvent as AnthropicStreamEvent, Role,
};
use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::{
    config::Region, types as bedrock, Client as BedrockClient, Config as BedrockConfig,
};
use aws_smithy_types::{Blob, Document, Number};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures_util::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

use super::{ChatClient, ChatStream};

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
        to_anthropic_json(&self.model, request)
    }

    fn to_anthropic_request(&self, request: &ChatRequest) -> Result<MessagesRequest, VvLlmError> {
        if !request.tools.is_empty() || request.tool_choice.is_some() {
            return Err(VvLlmError::Configuration(
                "direct Anthropic SDK path does not support tools; use Anthropic Bedrock resolved client for tool calls".to_string(),
            ));
        }

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
                    messages.push(to_anthropic_sdk_message(message, Role::User)?)
                }
                MessageRole::Assistant => {
                    messages.push(to_anthropic_sdk_message(message, Role::Assistant)?)
                }
            }
        }

        let mut builder = MessagesRequestBuilder::default();
        builder.model(request_model_or_default(&self.model, request));
        builder.messages(messages);
        builder.system(system.join("\n"));
        builder.max_tokens(request.options.max_tokens.unwrap_or(1024) as usize);
        builder.stream(request.options.stream.unwrap_or(false));
        if let Some(temperature) = request.options.temperature {
            builder.temperature(temperature as f64);
        }
        if let Some(top_p) = request.options.top_p {
            builder.top_p(top_p as f64);
        }
        if !request.options.stop.is_empty() {
            builder.stop_sequences(request.options.stop.clone());
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

    fn headers(&self) -> Result<HeaderMap, VvLlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key
                .parse::<HeaderValue>()
                .map_err(|error| VvLlmError::Provider(error.to_string()))?,
        );
        headers.insert(
            "anthropic-version",
            "2023-06-01"
                .parse::<HeaderValue>()
                .map_err(|error| VvLlmError::Provider(error.to_string()))?,
        );
        headers.insert(
            CONTENT_TYPE,
            "application/json"
                .parse::<HeaderValue>()
                .map_err(|error| VvLlmError::Provider(error.to_string()))?,
        );
        headers.insert(
            ACCEPT,
            "application/json"
                .parse::<HeaderValue>()
                .map_err(|error| VvLlmError::Provider(error.to_string()))?,
        );
        Ok(headers)
    }

    async fn post_messages_json(&self, request: Value) -> Result<Value, VvLlmError> {
        let response = reqwest::Client::new()
            .post(format!(
                "{}/v1/messages",
                self.api_base.trim_end_matches('/')
            ))
            .headers(self.headers()?)
            .json(&request)
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let value = serde_json::from_str::<Value>(&body)?;
        if !status.is_success() {
            return Err(VvLlmError::Provider(anthropic_error_message(&value)));
        }
        Ok(value)
    }

    async fn messages_json_stream(&self, request: Value) -> Result<ChatStream, VvLlmError> {
        let response = reqwest::Client::new()
            .post(format!(
                "{}/v1/messages",
                self.api_base.trim_end_matches('/')
            ))
            .headers(self.headers()?)
            .json(&request)
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .map_err(|error| VvLlmError::Provider(error.to_string()))?;
            let value = serde_json::from_str::<Value>(&body).unwrap_or(Value::String(body));
            return Err(VvLlmError::Provider(anthropic_error_message(&value)));
        }
        let bytes = response.bytes_stream();
        let state = AnthropicSseState::default();

        Ok(Box::pin(stream::unfold(
            (bytes, state),
            |(mut bytes, mut state)| async move {
                loop {
                    if let Some(event) = state.next_event() {
                        match normalize_anthropic_sse_event(&event, &mut state.usage) {
                            Ok(Some(delta)) => return Some((Ok(delta), (bytes, state))),
                            Ok(None) => continue,
                            Err(error) => return Some((Err(error), (bytes, state))),
                        }
                    }
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            let text = match std::str::from_utf8(&chunk) {
                                Ok(text) => text,
                                Err(error) => {
                                    return Some((
                                        Err(VvLlmError::Provider(error.to_string())),
                                        (bytes, state),
                                    ))
                                }
                            };
                            state.buffer.push_str(text);
                        }
                        Some(Err(error)) => {
                            return Some((
                                Err(VvLlmError::Provider(error.to_string())),
                                (bytes, state),
                            ))
                        }
                        None => return None,
                    }
                }
            },
        )))
    }
}

#[async_trait]
impl ChatClient for AnthropicChatClient {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        if request_needs_anthropic_json(&request) {
            let response = self
                .post_messages_json(self.to_anthropic_json(&request)?)
                .await?;
            return normalize_anthropic_response_json(response);
        }
        let response = self
            .client()?
            .messages(self.to_anthropic_request(&request)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let content = response
            .content
            .into_iter()
            .filter_map(|block| match block {
                AnthropicContentBlock::Text { text } => Some(text),
                AnthropicContentBlock::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let usage = normalize_anthropic_usage(serde_json::to_value(response.usage).ok());

        Ok(ChatResponse {
            id: response.id,
            model: response.model,
            content,
            tool_calls: Vec::new(),
            reasoning_content: None,
            usage,
        })
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        if request_needs_anthropic_json(&request) {
            let mut request = self.to_anthropic_json(&ChatRequest {
                options: crate::ChatRequestOptions {
                    stream: Some(true),
                    ..request.options.clone()
                },
                ..request
            })?;
            request["stream"] = Value::Bool(true);
            return self.messages_json_stream(request).await;
        }
        let stream = self
            .client()?
            .messages_stream(self.to_anthropic_request(&ChatRequest {
                options: crate::ChatRequestOptions {
                    stream: Some(true),
                    ..request.options.clone()
                },
                ..request
            })?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        Ok(Box::pin(stream::unfold(stream, |mut stream| async move {
            use futures_util::StreamExt;
            let next = stream.next().await?;
            let delta = next
                .map(normalize_anthropic_stream_event)
                .map_err(|error| VvLlmError::Provider(error.to_string()));
            Some((delta, stream))
        })))
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicBedrockChatClient {
    model: String,
    api_base: String,
    region: String,
    credentials: Value,
}

impl AnthropicBedrockChatClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        region: Option<String>,
        credentials: Value,
    ) -> Result<Self, VvLlmError> {
        let region = region.ok_or_else(|| {
            VvLlmError::Configuration("Anthropic Bedrock endpoint requires region".to_string())
        })?;
        if credential_value(&credentials, "access_key").is_none()
            || credential_value(&credentials, "secret_key").is_none()
        {
            return Err(VvLlmError::Configuration(
                "Anthropic Bedrock endpoint requires access_key and secret_key".to_string(),
            ));
        }

        Ok(Self {
            model: model.into(),
            api_base: api_base.into(),
            region,
            credentials,
        })
    }

    pub fn to_anthropic_json(
        &self,
        request: &ChatRequest,
    ) -> Result<serde_json::Value, VvLlmError> {
        to_anthropic_json(&self.model, request)
    }

    fn client(&self) -> BedrockClient {
        let credentials = Credentials::new(
            credential_value(&self.credentials, "access_key").unwrap_or_default(),
            credential_value(&self.credentials, "secret_key").unwrap_or_default(),
            credential_value(&self.credentials, "session_token"),
            None,
            "vv-llm",
        );
        let mut config = BedrockConfig::builder()
            .behavior_version_latest()
            .region(Region::new(self.region.clone()))
            .credentials_provider(credentials);
        if !self.api_base.is_empty() {
            config = config.endpoint_url(self.api_base.clone());
        }
        BedrockClient::from_conf(config.build())
    }
}

#[async_trait]
impl ChatClient for AnthropicBedrockChatClient {
    fn provider_name(&self) -> &'static str {
        "anthropic-bedrock"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        let bedrock_request = to_bedrock_request(&self.model, &request)?;
        let mut operation = self
            .client()
            .converse()
            .model_id(bedrock_request.model_id)
            .set_messages(Some(bedrock_request.messages));

        if !bedrock_request.system.is_empty() {
            operation = operation.set_system(Some(bedrock_request.system));
        }
        if let Some(inference_config) = bedrock_request.inference_config {
            operation = operation.inference_config(inference_config);
        }
        if let Some(tool_config) = bedrock_request.tool_config {
            operation = operation.tool_config(tool_config);
        }

        let response = operation
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;

        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        if let Some(bedrock::ConverseOutput::Message(message)) = response.output {
            for block in message.content {
                match block {
                    bedrock::ContentBlock::Text(text) => content_parts.push(text),
                    bedrock::ContentBlock::ToolUse(tool_use) => {
                        tool_calls.push(ToolCall {
                            id: tool_use.tool_use_id,
                            name: tool_use.name,
                            arguments: document_to_json(&tool_use.input)?.to_string(),
                            index: None,
                            extra_content: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = response.usage.map(normalize_bedrock_usage);

        Ok(ChatResponse {
            id: String::new(),
            model: self.model.clone(),
            content: content_parts.join("\n"),
            tool_calls,
            reasoning_content: None,
            usage,
        })
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let bedrock_request = to_bedrock_request(&self.model, &request)?;
        let mut operation = self
            .client()
            .converse_stream()
            .model_id(bedrock_request.model_id)
            .set_messages(Some(bedrock_request.messages));

        if !bedrock_request.system.is_empty() {
            operation = operation.set_system(Some(bedrock_request.system));
        }
        if let Some(inference_config) = bedrock_request.inference_config {
            operation = operation.inference_config(inference_config);
        }
        if let Some(tool_config) = bedrock_request.tool_config {
            operation = operation.tool_config(tool_config);
        }

        let response = operation
            .send()
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;
        let receiver = response.stream;
        let state = BedrockStreamState::default();

        Ok(Box::pin(stream::unfold(
            (receiver, state),
            |(mut receiver, mut state)| async move {
                loop {
                    let event = match receiver.recv().await {
                        Ok(Some(event)) => event,
                        Ok(None) => return None,
                        Err(error) => {
                            return Some((
                                Err(VvLlmError::Provider(error.to_string())),
                                (receiver, state),
                            ))
                        }
                    };
                    match normalize_bedrock_stream_event(event, &mut state) {
                        Ok(Some(delta)) => return Some((Ok(delta), (receiver, state))),
                        Ok(None) => continue,
                        Err(error) => return Some((Err(error), (receiver, state))),
                    }
                }
            },
        )))
    }
}

#[derive(Debug)]
struct BedrockChatRequest {
    model_id: String,
    messages: Vec<bedrock::Message>,
    system: Vec<bedrock::SystemContentBlock>,
    inference_config: Option<bedrock::InferenceConfiguration>,
    tool_config: Option<bedrock::ToolConfiguration>,
}

fn to_anthropic_json(model: &str, request: &ChatRequest) -> Result<Value, VvLlmError> {
    let mut system = Vec::<Value>::new();
    let mut messages = Vec::new();

    for message in &request.messages {
        match message.role {
            MessageRole::System => {
                system.extend(to_anthropic_content(message)?);
            }
            MessageRole::User | MessageRole::Tool => {
                messages.push(json!({
                    "role": "user",
                    "content": to_anthropic_content(message)?,
                }));
            }
            MessageRole::Assistant => {
                messages.push(json!({
                    "role": "assistant",
                    "content": to_anthropic_content(message)?,
                }));
            }
        }
    }

    let mut value = json!({
        "model": request_model_or_default(model, request),
        "messages": messages,
        "max_tokens": request.options.max_tokens.unwrap_or(1024),
        "stream": request.options.stream.unwrap_or(false),
    });
    if system.iter().any(block_has_anthropic_extension) {
        value["system"] = Value::Array(system);
    } else if !system.is_empty() {
        value["system"] = Value::String(
            system
                .iter()
                .filter_map(text_block_text)
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    if let Some(temperature) = request.options.temperature {
        value["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.options.top_p {
        value["top_p"] = json!(top_p);
    }
    if !request.options.stop.is_empty() {
        value["stop_sequences"] = json!(request.options.stop);
    }
    if !request.tools.is_empty() {
        value["tools"] = Value::Array(request.tools.iter().map(to_anthropic_tool).collect());
    }
    if let Some(tool_choice) = request.tool_choice.as_deref() {
        value["tool_choice"] = to_anthropic_tool_choice(tool_choice)?;
    } else if !request.tools.is_empty() {
        value["tool_choice"] = json!({"type": "auto"});
    }
    merge_extra_body(&mut value, &request.extra_body);

    Ok(value)
}

fn block_has_anthropic_extension(block: &Value) -> bool {
    block.get("cache_control").is_some()
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

fn to_anthropic_content(message: &Message) -> Result<Vec<Value>, VvLlmError> {
    let mut blocks = Vec::new();
    for content in &message.content {
        match content {
            MessageContent::Text {
                text,
                cache_control,
            } => {
                let mut block = json!({
                    "type": "text",
                    "text": text,
                });
                if let Some(cache_control) = cache_control {
                    block["cache_control"] = cache_control.clone();
                }
                blocks.push(block);
            }
            MessageContent::ImageUrl { url } => {
                let data = parse_data_url(url)?;
                blocks.push(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": data.media_type,
                        "data": data.base64,
                    }
                }));
            }
        }
    }

    for tool_call in &message.tool_calls {
        blocks.push(json!({
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.name,
            "input": parse_tool_arguments(&tool_call.arguments)?,
        }));
    }

    if message.role == MessageRole::Tool {
        let tool_use_id = message
            .tool_call_id
            .clone()
            .unwrap_or_else(|| "tool-call".to_string());
        let content = message.text_content().unwrap_or_default();
        blocks.clear();
        blocks.push(json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }));
    }

    Ok(blocks)
}

fn to_anthropic_tool(tool: &ChatTool) -> Value {
    let mut value = json!({
        "name": tool.name,
        "description": tool.description.clone().unwrap_or_default(),
        "input_schema": tool.parameters,
    });
    if let Some(cache_control) = &tool.cache_control {
        value["cache_control"] = cache_control.clone();
    }
    value
}

fn to_anthropic_tool_choice(choice: &str) -> Result<Value, VvLlmError> {
    match choice {
        "auto" => Ok(json!({"type": "auto"})),
        "required" => Ok(json!({"type": "any"})),
        "none" => Ok(json!({"type": "auto"})),
        other => Err(VvLlmError::Configuration(format!(
            "unsupported tool_choice value: {other}"
        ))),
    }
}

fn to_bedrock_request(
    model: &str,
    request: &ChatRequest,
) -> Result<BedrockChatRequest, VvLlmError> {
    let mut messages = Vec::new();
    let mut system = Vec::new();
    for message in &request.messages {
        match message.role {
            MessageRole::System => {
                if let Some(text) = message.text_content() {
                    system.push(bedrock::SystemContentBlock::Text(text));
                }
            }
            MessageRole::User | MessageRole::Tool => messages.push(to_bedrock_message(
                message,
                bedrock::ConversationRole::User,
            )?),
            MessageRole::Assistant => messages.push(to_bedrock_message(
                message,
                bedrock::ConversationRole::Assistant,
            )?),
        }
    }

    let inference_config = if request.options.max_tokens.is_some()
        || request.options.temperature.is_some()
        || request.options.top_p.is_some()
        || !request.options.stop.is_empty()
    {
        let mut builder = bedrock::InferenceConfiguration::builder();
        if let Some(max_tokens) = request.options.max_tokens {
            builder = builder.max_tokens(max_tokens as i32);
        }
        if let Some(temperature) = request.options.temperature {
            builder = builder.temperature(temperature);
        }
        if let Some(top_p) = request.options.top_p {
            builder = builder.top_p(top_p);
        }
        if !request.options.stop.is_empty() {
            builder = builder.set_stop_sequences(Some(request.options.stop.clone()));
        }
        Some(builder.build())
    } else {
        None
    };

    let tool_config = if request.tools.is_empty() {
        None
    } else {
        let tools = request
            .tools
            .iter()
            .map(to_bedrock_tool)
            .collect::<Result<Vec<_>, _>>()?;
        let mut builder = bedrock::ToolConfiguration::builder().set_tools(Some(tools));
        if let Some(choice) = request.tool_choice.as_deref() {
            if let Some(choice) = to_bedrock_tool_choice(choice)? {
                builder = builder.tool_choice(choice);
            }
        } else {
            builder = builder.tool_choice(bedrock::ToolChoice::Auto(
                bedrock::AutoToolChoice::builder().build(),
            ));
        }
        Some(build_bedrock(builder.build())?)
    };

    Ok(BedrockChatRequest {
        model_id: request_model_or_default(model, request),
        messages,
        system,
        inference_config,
        tool_config,
    })
}

fn to_anthropic_sdk_message(message: &Message, role: Role) -> Result<AnthropicMessage, VvLlmError> {
    let mut content = Vec::new();
    for block in &message.content {
        match block {
            MessageContent::Text { text, .. } => {
                content.push(AnthropicContentBlock::Text { text: text.clone() });
            }
            MessageContent::ImageUrl { url } => {
                let data = parse_data_url(url)?;
                content.push(AnthropicContentBlock::Image {
                    source: "base64".to_string(),
                    media_type: data.media_type,
                    data: data.base64,
                });
            }
        }
    }
    if content.is_empty() {
        content.push(AnthropicContentBlock::Text {
            text: message.text_content().unwrap_or_default(),
        });
    }
    Ok(AnthropicMessage { role, content })
}

fn to_bedrock_message(
    message: &Message,
    role: bedrock::ConversationRole,
) -> Result<bedrock::Message, VvLlmError> {
    let content = if message.role == MessageRole::Tool {
        let tool_use_id = message
            .tool_call_id
            .clone()
            .unwrap_or_else(|| "tool-call".to_string());
        vec![bedrock::ContentBlock::ToolResult(build_bedrock(
            bedrock::ToolResultBlock::builder()
                .tool_use_id(tool_use_id)
                .content(bedrock::ToolResultContentBlock::Text(
                    message.text_content().unwrap_or_default(),
                ))
                .build(),
        )?)]
    } else {
        let mut content = Vec::new();
        for block in &message.content {
            match block {
                MessageContent::Text { text, .. } => {
                    content.push(bedrock::ContentBlock::Text(text.clone()));
                }
                MessageContent::ImageUrl { url } => {
                    let data = parse_data_url(url)?;
                    let image = bedrock::ImageBlock::builder()
                        .format(to_bedrock_image_format(&data.media_type)?)
                        .source(bedrock::ImageSource::Bytes(Blob::new(data.bytes()?)))
                        .build()
                        .map_err(|error| VvLlmError::Provider(error.to_string()))?;
                    content.push(bedrock::ContentBlock::Image(image));
                }
            }
        }
        for tool_call in &message.tool_calls {
            let tool_use = bedrock::ToolUseBlock::builder()
                .tool_use_id(tool_call.id.clone())
                .name(tool_call.name.clone())
                .input(json_to_document(&parse_tool_arguments(
                    &tool_call.arguments,
                )?))
                .build()
                .map_err(|error| VvLlmError::Provider(error.to_string()))?;
            content.push(bedrock::ContentBlock::ToolUse(tool_use));
        }
        if content.is_empty() {
            content.push(bedrock::ContentBlock::Text(String::new()));
        }
        content
    };

    bedrock::Message::builder()
        .role(role)
        .set_content(Some(content))
        .build()
        .map_err(|error| VvLlmError::Provider(error.to_string()))
}

fn to_bedrock_tool(tool: &ChatTool) -> Result<bedrock::Tool, VvLlmError> {
    let spec = bedrock::ToolSpecification::builder()
        .name(tool.name.clone())
        .set_description(tool.description.clone())
        .input_schema(bedrock::ToolInputSchema::Json(json_to_document(
            &tool.parameters,
        )))
        .build()
        .map_err(|error| VvLlmError::Provider(error.to_string()))?;
    Ok(bedrock::Tool::ToolSpec(spec))
}

fn to_bedrock_tool_choice(choice: &str) -> Result<Option<bedrock::ToolChoice>, VvLlmError> {
    match choice {
        "auto" => Ok(Some(bedrock::ToolChoice::Auto(
            bedrock::AutoToolChoice::builder().build(),
        ))),
        "required" => Ok(Some(bedrock::ToolChoice::Any(
            bedrock::AnyToolChoice::builder().build(),
        ))),
        "none" => Ok(None),
        other => Err(VvLlmError::Configuration(format!(
            "unsupported tool_choice value: {other}"
        ))),
    }
}

fn request_needs_anthropic_json(request: &ChatRequest) -> bool {
    !request.tools.is_empty()
        || request.tool_choice.is_some()
        || !is_empty_extra_body(&request.extra_body)
        || request
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|content| {
                matches!(
                    content,
                    MessageContent::Text {
                        cache_control: Some(_),
                        ..
                    }
                )
            })
}

fn is_empty_extra_body(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(object) => object.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bedrock_request_maps_python_style_chat_options() {
        let request = ChatRequest {
            model: "anthropic.claude-sonnet-4-6".to_string(),
            messages: vec![Message::text(MessageRole::User, "hello")],
            options: crate::ChatRequestOptions {
                top_p: Some(0.8),
                stop: vec!["END".to_string(), "DONE".to_string()],
                ..Default::default()
            },
            tools: Vec::new(),
            tool_choice: None,
            extra_body: Value::Null,
        };

        let request = to_bedrock_request("anthropic.claude-sonnet-4-6", &request).unwrap();
        let inference = request.inference_config.expect("inference config");

        assert!((inference.top_p().unwrap() - 0.8).abs() < 0.000_001);
        assert_eq!(inference.stop_sequences(), ["END", "DONE"]);
    }

    #[test]
    fn anthropic_usage_distinguishes_missing_zero_positive_and_invalid_cache_values() {
        let cases = [
            (
                "missing",
                json!({"input_tokens": 11, "output_tokens": 7}),
                None,
                None,
            ),
            (
                "explicit-zero",
                json!({
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0
                }),
                Some(0),
                Some(0),
            ),
            (
                "positive",
                json!({
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 6,
                    "cache_creation_input_tokens": 4
                }),
                Some(6),
                Some(4),
            ),
            (
                "invalid",
                json!({
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "cache_read_input_tokens": "6",
                    "cache_creation_input_tokens": 4.5
                }),
                None,
                None,
            ),
        ];

        for (case, raw_usage, expected_read, expected_creation) in cases {
            let response = normalize_anthropic_response_json(json!({
                "id": format!("msg_{case}"),
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "usage": raw_usage.clone()
            }))
            .unwrap();
            let usage = response.usage.expect(case);

            assert_eq!(usage.prompt_tokens, Some(11), "{case}");
            assert_eq!(usage.completion_tokens, Some(7), "{case}");
            assert_eq!(usage.total_tokens, Some(18), "{case}");
            assert_eq!(usage.input_tokens, Some(11), "{case}");
            assert_eq!(usage.output_tokens, Some(7), "{case}");
            assert_eq!(usage.cache_read_input_tokens, expected_read, "{case}");
            assert_eq!(
                usage.cache_creation_input_tokens, expected_creation,
                "{case}"
            );
            assert_eq!(usage.raw_usage, Some(raw_usage), "{case}");
        }
    }

    #[test]
    fn anthropic_stream_accumulates_start_cache_usage_and_final_output_usage() {
        let mut accumulated_usage = None;
        let start = normalize_anthropic_stream_json(
            json!({
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 40
                    }
                }
            }),
            &mut accumulated_usage,
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            start
                .usage
                .as_ref()
                .and_then(|usage| usage.cache_read_input_tokens),
            Some(0)
        );

        let final_delta = normalize_anthropic_stream_json(
            json!({
                "type": "message_delta",
                "usage": {"output_tokens": 25}
            }),
            &mut accumulated_usage,
        )
        .unwrap()
        .unwrap();
        let usage = final_delta.usage.unwrap();

        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(25));
        assert_eq!(usage.total_tokens, Some(125));
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(25));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
        assert_eq!(usage.cache_creation_input_tokens, Some(40));
        assert_eq!(
            usage.raw_usage,
            Some(json!({
                "input_tokens": 100,
                "output_tokens": 25,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 40
            }))
        );
    }

    #[test]
    fn anthropic_sse_state_drains_multiple_events_from_one_network_chunk() {
        let mut state = AnthropicSseState {
            buffer: concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0,\"cache_read_input_tokens\":0}}}\n\n",
                "event: message_delta\n",
                "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\n"
            )
            .to_string(),
            usage: None,
        };

        let first = state.next_event().expect("first buffered event");
        normalize_anthropic_sse_event(&first, &mut state.usage)
            .expect("normalize first event")
            .expect("first delta");
        let second = state.next_event().expect("second buffered event");
        let second = normalize_anthropic_sse_event(&second, &mut state.usage)
            .expect("normalize second event")
            .expect("second delta");

        let usage = second.usage.expect("merged usage");
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(2));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
        assert!(state.next_event().is_none());
    }

    #[test]
    fn bedrock_usage_maps_cache_write_to_cache_creation() {
        let usage = bedrock::TokenUsage::builder()
            .input_tokens(100)
            .output_tokens(25)
            .total_tokens(125)
            .cache_read_input_tokens(0)
            .cache_write_input_tokens(40)
            .build()
            .unwrap();

        let usage = normalize_bedrock_usage(usage);

        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(25));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
        assert_eq!(usage.cache_creation_input_tokens, Some(40));
        assert_eq!(
            usage
                .raw_usage
                .as_ref()
                .and_then(|raw| raw.get("cache_write_input_tokens")),
            Some(&json!(40))
        );
    }
}

fn normalize_anthropic_response_json(value: Value) -> Result<ChatResponse, VvLlmError> {
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(text_block_text)
        .collect::<Vec<_>>()
        .join("\n");
    let usage = normalize_anthropic_usage(value.get("usage").cloned());
    Ok(ChatResponse {
        id: value_to_string(value.get("id")),
        model: value_to_string(value.get("model")),
        content,
        tool_calls: tool_calls_from_anthropic_content(value.get("content"))?,
        reasoning_content: None,
        usage,
    })
}

#[derive(Debug, Default)]
struct AnthropicSseState {
    buffer: String,
    usage: Option<ChatUsage>,
}

impl AnthropicSseState {
    fn next_event(&mut self) -> Option<String> {
        let separator = self
            .buffer
            .find("\n\n")
            .or_else(|| self.buffer.find("\r\n\r\n"))?;
        let event = self.buffer[..separator].to_string();
        let drain_to = if self.buffer[separator..].starts_with("\r\n\r\n") {
            separator + 4
        } else {
            separator + 2
        };
        self.buffer.drain(..drain_to);
        Some(event)
    }
}

fn normalize_anthropic_sse_event(
    event: &str,
    usage: &mut Option<ChatUsage>,
) -> Result<Option<ChatStreamDelta>, VvLlmError> {
    let mut event_name = String::new();
    let mut data = Vec::new();
    for line in event.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(value) = line.strip_prefix("event:") {
            event_name = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start());
        }
    }
    if event_name == "ping" || data.is_empty() {
        return Ok(None);
    }
    let data = data.join("\n");
    if event_name == "error" {
        let value = serde_json::from_str::<Value>(&data)?;
        return Err(VvLlmError::Provider(anthropic_error_message(&value)));
    }
    let value = serde_json::from_str::<Value>(&data)?;
    normalize_anthropic_stream_json(value, usage)
}

fn normalize_anthropic_stream_json(
    value: Value,
    accumulated_usage: &mut Option<ChatUsage>,
) -> Result<Option<ChatStreamDelta>, VvLlmError> {
    match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "message_start" => {
            merge_anthropic_usage(
                accumulated_usage,
                normalize_anthropic_usage(value.pointer("/message/usage").cloned()),
            );
            Ok(Some(ChatStreamDelta {
                usage: accumulated_usage.clone(),
                raw_content: Some(value),
                ..Default::default()
            }))
        }
        "content_block_start" => {
            let Some(block) = value.get("content_block") else {
                return Ok(None);
            };
            match block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "text" => Ok(text_block_text(block).map(|content| ChatStreamDelta {
                    content,
                    ..Default::default()
                })),
                "tool_use" => Ok(Some(ChatStreamDelta {
                    tool_calls: vec![tool_call_from_anthropic_block(block)?],
                    ..Default::default()
                })),
                _ => Ok(None),
            }
        }
        "content_block_delta" => {
            let Some(delta) = value.get("delta") else {
                return Ok(None);
            };
            match delta
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "text_delta" => Ok(Some(ChatStreamDelta {
                    content: value_to_string(delta.get("text")),
                    ..Default::default()
                })),
                "input_json_delta" => Ok(Some(ChatStreamDelta {
                    tool_calls: vec![ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: value_to_string(delta.get("partial_json")),
                        index: None,
                        extra_content: None,
                    }],
                    ..Default::default()
                })),
                _ => Ok(None),
            }
        }
        "message_delta" => {
            merge_anthropic_usage(
                accumulated_usage,
                normalize_anthropic_usage(value.get("usage").cloned()),
            );
            Ok(accumulated_usage.clone().map(|usage| ChatStreamDelta {
                usage: Some(usage),
                ..Default::default()
            }))
        }
        "message_stop" => Ok(Some(ChatStreamDelta {
            done: true,
            ..Default::default()
        })),
        _ => Ok(None),
    }
}

fn tool_calls_from_anthropic_content(content: Option<&Value>) -> Result<Vec<ToolCall>, VvLlmError> {
    content
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        .map(tool_call_from_anthropic_block)
        .collect()
}

fn tool_call_from_anthropic_block(block: &Value) -> Result<ToolCall, VvLlmError> {
    Ok(ToolCall {
        id: value_to_string(block.get("id")),
        name: value_to_string(block.get("name")),
        arguments: block
            .get("input")
            .cloned()
            .unwrap_or_else(|| json!({}))
            .to_string(),
        index: None,
        extra_content: None,
    })
}

fn text_block_text(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) == Some("text") {
        Some(value_to_string(block.get("text")))
    } else {
        None
    }
}

fn anthropic_error_message(value: &Value) -> String {
    value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn value_to_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(value)) => value.to_string(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Array(_)) | Some(Value::Object(_)) => {
            value.cloned().unwrap_or(Value::Null).to_string()
        }
        Some(Value::Null) | None => String::new(),
    }
}

fn value_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn normalize_anthropic_usage(raw_usage: Option<Value>) -> Option<ChatUsage> {
    let raw_usage = raw_usage.filter(|usage| !usage.is_null())?;
    let input_tokens = value_u32(raw_usage.get("input_tokens"));
    let output_tokens = value_u32(raw_usage.get("output_tokens"));
    let total_tokens = match (input_tokens, output_tokens) {
        (Some(input), Some(output)) => input.checked_add(output),
        _ => value_u32(raw_usage.get("total_tokens")),
    };

    Some(ChatUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens,
        input_tokens,
        output_tokens,
        cache_read_input_tokens: value_u32(raw_usage.get("cache_read_input_tokens")),
        cache_creation_input_tokens: value_u32(
            raw_usage
                .get("cache_creation_input_tokens")
                .or_else(|| raw_usage.get("cache_write_input_tokens")),
        ),
        raw_usage: Some(raw_usage),
    })
}

fn merge_anthropic_usage(current: &mut Option<ChatUsage>, update: Option<ChatUsage>) {
    let Some(update) = update else {
        return;
    };
    let Some(current) = current.as_mut() else {
        *current = Some(update);
        return;
    };

    if update.prompt_tokens.is_some() {
        current.prompt_tokens = update.prompt_tokens;
    }
    if update.completion_tokens.is_some() {
        current.completion_tokens = update.completion_tokens;
    }
    if update.input_tokens.is_some() {
        current.input_tokens = update.input_tokens;
    }
    if update.output_tokens.is_some() {
        current.output_tokens = update.output_tokens;
    }
    if update.cache_read_input_tokens.is_some() {
        current.cache_read_input_tokens = update.cache_read_input_tokens;
    }
    if update.cache_creation_input_tokens.is_some() {
        current.cache_creation_input_tokens = update.cache_creation_input_tokens;
    }
    current.raw_usage = merge_usage_json(current.raw_usage.take(), update.raw_usage);
    current.total_tokens = match (current.input_tokens, current.output_tokens) {
        (Some(input), Some(output)) => input.checked_add(output),
        _ => update.total_tokens.or(current.total_tokens),
    };
}

fn merge_usage_json(current: Option<Value>, update: Option<Value>) -> Option<Value> {
    match (current, update) {
        (Some(Value::Object(mut current)), Some(Value::Object(update))) => {
            current.extend(update);
            Some(Value::Object(current))
        }
        (_, Some(update)) => Some(update),
        (current, None) => current,
    }
}

fn normalize_bedrock_usage(usage: bedrock::TokenUsage) -> ChatUsage {
    let input_tokens = non_negative_u32(usage.input_tokens);
    let output_tokens = non_negative_u32(usage.output_tokens);
    let cache_read_input_tokens = usage.cache_read_input_tokens.and_then(non_negative_u32);
    let cache_creation_input_tokens = usage.cache_write_input_tokens.and_then(non_negative_u32);
    let mut raw_usage = Map::from_iter([
        ("input_tokens".to_string(), json!(usage.input_tokens)),
        ("output_tokens".to_string(), json!(usage.output_tokens)),
        ("total_tokens".to_string(), json!(usage.total_tokens)),
    ]);
    if let Some(value) = usage.cache_read_input_tokens {
        raw_usage.insert("cache_read_input_tokens".to_string(), json!(value));
    }
    if let Some(value) = usage.cache_write_input_tokens {
        raw_usage.insert("cache_write_input_tokens".to_string(), json!(value));
    }
    if let Some(cache_details) = usage.cache_details {
        raw_usage.insert(
            "cache_details".to_string(),
            Value::Array(
                cache_details
                    .into_iter()
                    .map(|detail| {
                        json!({
                            "ttl": detail.ttl.as_str(),
                            "input_tokens": detail.input_tokens,
                        })
                    })
                    .collect(),
            ),
        );
    }

    ChatUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: non_negative_u32(usage.total_tokens),
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        raw_usage: Some(Value::Object(raw_usage)),
    }
}

fn normalize_anthropic_stream_event(event: AnthropicStreamEvent) -> ChatStreamDelta {
    match event {
        AnthropicStreamEvent::ContentBlockStart { content_block, .. } => match content_block {
            AnthropicContentBlock::Text { text } => ChatStreamDelta {
                content: text,
                ..Default::default()
            },
            AnthropicContentBlock::Image { .. } => ChatStreamDelta::default(),
        },
        AnthropicStreamEvent::ContentBlockDelta { delta, .. } => match delta {
            AnthropicContentBlockDelta::TextDelta { text } => ChatStreamDelta {
                content: text,
                ..Default::default()
            },
        },
        AnthropicStreamEvent::MessageStart { message } => ChatStreamDelta {
            raw_content: serde_json::to_value(message).ok(),
            ..Default::default()
        },
        AnthropicStreamEvent::MessageDelta { .. } => ChatStreamDelta::default(),
        AnthropicStreamEvent::MessageStop => ChatStreamDelta {
            done: true,
            ..Default::default()
        },
        AnthropicStreamEvent::ContentBlockStop { .. } => ChatStreamDelta::default(),
    }
}

#[derive(Debug, Default)]
struct BedrockStreamState {
    tool_uses: HashMap<i32, ToolCall>,
}

fn normalize_bedrock_stream_event(
    event: bedrock::ConverseStreamOutput,
    state: &mut BedrockStreamState,
) -> Result<Option<ChatStreamDelta>, VvLlmError> {
    match event {
        bedrock::ConverseStreamOutput::ContentBlockStart(event) => {
            if let Some(bedrock::ContentBlockStart::ToolUse(tool_use)) = event.start {
                let tool_call = ToolCall {
                    id: tool_use.tool_use_id,
                    name: tool_use.name,
                    arguments: String::new(),
                    index: Some(event.content_block_index as usize),
                    extra_content: None,
                };
                state
                    .tool_uses
                    .insert(event.content_block_index, tool_call.clone());
                return Ok(Some(ChatStreamDelta {
                    tool_calls: vec![tool_call],
                    ..Default::default()
                }));
            }
            Ok(None)
        }
        bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                return Ok(None);
            };
            match delta {
                bedrock::ContentBlockDelta::Text(text) => Ok(Some(ChatStreamDelta {
                    content: text,
                    ..Default::default()
                })),
                bedrock::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    bedrock::ReasoningContentBlockDelta::Text(text) => Ok(Some(ChatStreamDelta {
                        reasoning_content: text,
                        ..Default::default()
                    })),
                    bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                        Ok(Some(ChatStreamDelta {
                            raw_content: Some(
                                json!({"type":"reasoning_signature","signature":signature}),
                            ),
                            ..Default::default()
                        }))
                    }
                    bedrock::ReasoningContentBlockDelta::RedactedContent(_) => Ok(None),
                    _ => Ok(None),
                },
                bedrock::ContentBlockDelta::ToolUse(tool_use) => {
                    let tool_call = state
                        .tool_uses
                        .entry(event.content_block_index)
                        .or_insert_with(|| ToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            index: Some(event.content_block_index as usize),
                            extra_content: None,
                        });
                    tool_call.arguments.push_str(&tool_use.input);
                    Ok(Some(ChatStreamDelta {
                        tool_calls: vec![ToolCall {
                            id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            arguments: tool_use.input,
                            index: Some(event.content_block_index as usize),
                            extra_content: None,
                        }],
                        ..Default::default()
                    }))
                }
                _ => Ok(None),
            }
        }
        bedrock::ConverseStreamOutput::Metadata(event) => {
            let usage = event.usage.map(normalize_bedrock_usage);
            Ok(Some(ChatStreamDelta {
                usage,
                ..Default::default()
            }))
        }
        bedrock::ConverseStreamOutput::MessageStop(_) => Ok(Some(ChatStreamDelta {
            done: true,
            ..Default::default()
        })),
        _ => Ok(None),
    }
}

fn parse_tool_arguments(arguments: &str) -> Result<Value, VvLlmError> {
    if arguments.trim().is_empty() {
        Ok(json!({}))
    } else {
        Ok(serde_json::from_str(arguments)?)
    }
}

fn json_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(value) => Document::Bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_u64() {
                Document::Number(Number::PosInt(value))
            } else if let Some(value) = value.as_i64() {
                Document::Number(Number::NegInt(value))
            } else {
                Document::Number(Number::Float(value.as_f64().unwrap_or_default()))
            }
        }
        Value::String(value) => Document::String(value.clone()),
        Value::Array(values) => Document::Array(values.iter().map(json_to_document).collect()),
        Value::Object(values) => Document::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_document(value)))
                .collect::<HashMap<_, _>>(),
        ),
    }
}

fn document_to_json(document: &Document) -> Result<Value, VvLlmError> {
    match document {
        Document::Null => Ok(Value::Null),
        Document::Bool(value) => Ok(Value::Bool(*value)),
        Document::Number(value) => Ok(number_to_json(*value)),
        Document::String(value) => Ok(Value::String(value.clone())),
        Document::Array(values) => values
            .iter()
            .map(document_to_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Document::Object(values) => values
            .iter()
            .map(|(key, value)| Ok((key.clone(), document_to_json(value)?)))
            .collect::<Result<Map<_, _>, _>>()
            .map(Value::Object),
    }
}

fn number_to_json(value: Number) -> Value {
    match value {
        Number::PosInt(value) => json!(value),
        Number::NegInt(value) => json!(value),
        Number::Float(value) => json!(value),
    }
}

#[derive(Debug, Clone)]
struct DataUrl {
    media_type: String,
    base64: String,
}

impl DataUrl {
    fn bytes(&self) -> Result<Vec<u8>, VvLlmError> {
        STANDARD
            .decode(self.base64.as_bytes())
            .map_err(|error| VvLlmError::Configuration(error.to_string()))
    }
}

fn parse_data_url(url: &str) -> Result<DataUrl, VvLlmError> {
    let (metadata, base64) = url.split_once(',').ok_or_else(|| {
        VvLlmError::Configuration("image data URL is missing comma separator".to_string())
    })?;
    let media_type = metadata
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .ok_or_else(|| {
            VvLlmError::Configuration("image URL must be base64 data URL".to_string())
        })?;
    Ok(DataUrl {
        media_type: media_type.to_string(),
        base64: base64.to_string(),
    })
}

fn to_bedrock_image_format(media_type: &str) -> Result<bedrock::ImageFormat, VvLlmError> {
    match media_type {
        "image/png" => Ok(bedrock::ImageFormat::Png),
        "image/jpeg" | "image/jpg" => Ok(bedrock::ImageFormat::Jpeg),
        "image/gif" => Ok(bedrock::ImageFormat::Gif),
        "image/webp" => Ok(bedrock::ImageFormat::Webp),
        other => Err(VvLlmError::Configuration(format!(
            "unsupported Bedrock image media type: {other}"
        ))),
    }
}

fn credential_value(credentials: &Value, key: &str) -> Option<String> {
    credentials
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn request_model_or_default(model: &str, request: &ChatRequest) -> String {
    if request.model.is_empty() {
        model.to_string()
    } else {
        request.model.clone()
    }
}

fn build_bedrock<T, E: std::fmt::Display>(result: Result<T, E>) -> Result<T, VvLlmError> {
    result.map_err(|error| VvLlmError::Provider(error.to_string()))
}

fn non_negative_u32(value: i32) -> Option<u32> {
    u32::try_from(value).ok()
}
