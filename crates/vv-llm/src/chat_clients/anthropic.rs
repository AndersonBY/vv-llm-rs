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
use futures_util::stream;
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
                AnthropicContentBlock::Text { text } => Some(text),
                AnthropicContentBlock::Image { .. } => None,
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
            tool_calls: Vec::new(),
            reasoning_content: None,
            usage,
        })
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
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
                            extra_content: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        let usage = response.usage.map(|usage| ChatUsage {
            prompt_tokens: non_negative_u32(usage.input_tokens),
            completion_tokens: non_negative_u32(usage.output_tokens),
            total_tokens: non_negative_u32(usage.total_tokens),
        });

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
    if !system.is_empty() {
        value["system"] = Value::String(system.join("\n"));
    }
    if let Some(temperature) = request.options.temperature {
        value["temperature"] = json!(temperature);
    }
    if !request.tools.is_empty() {
        value["tools"] = Value::Array(request.tools.iter().map(to_anthropic_tool).collect());
    }
    if let Some(tool_choice) = request.tool_choice.as_deref() {
        value["tool_choice"] = to_anthropic_tool_choice(tool_choice)?;
    } else if !request.tools.is_empty() {
        value["tool_choice"] = json!({"type": "auto"});
    }

    Ok(value)
}

fn to_anthropic_content(message: &Message) -> Result<Vec<Value>, VvLlmError> {
    let mut blocks = Vec::new();
    for content in &message.content {
        match content {
            MessageContent::Text { text } => blocks.push(json!({
                "type": "text",
                "text": text,
            })),
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
    json!({
        "name": tool.name,
        "description": tool.description.clone().unwrap_or_default(),
        "input_schema": tool.parameters,
    })
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

    let inference_config =
        if request.options.max_tokens.is_some() || request.options.temperature.is_some() {
            let mut builder = bedrock::InferenceConfiguration::builder();
            if let Some(max_tokens) = request.options.max_tokens {
                builder = builder.max_tokens(max_tokens as i32);
            }
            if let Some(temperature) = request.options.temperature {
                builder = builder.temperature(temperature);
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
            MessageContent::Text { text } => {
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
                MessageContent::Text { text } => {
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
                            extra_content: None,
                        });
                    tool_call.arguments.push_str(&tool_use.input);
                    Ok(Some(ChatStreamDelta {
                        tool_calls: vec![ToolCall {
                            id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            arguments: tool_use.input,
                            extra_content: None,
                        }],
                        ..Default::default()
                    }))
                }
                _ => Ok(None),
            }
        }
        bedrock::ConverseStreamOutput::Metadata(event) => {
            let usage = event.usage.map(|usage| ChatUsage {
                prompt_tokens: non_negative_u32(usage.input_tokens),
                completion_tokens: non_negative_u32(usage.output_tokens),
                total_tokens: non_negative_u32(usage.total_tokens),
            });
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
