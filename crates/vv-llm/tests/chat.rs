use futures_util::StreamExt;
use vv_llm::{
    chat_clients::{
        create_chat_client, create_chat_client_from_resolved, AnthropicChatClient,
        GoogleAccessTokenProvider, OpenAiCompatibleChatClient,
    },
    BackendType, ChatClient, ChatRequest, ChatRequestOptions, ChatStreamDelta, ChatTool,
    EndpointBinding, EndpointConfig, Message, MessageContent, MessageRole, ModelConfig,
    ResolvedModelConfig, ToolCall,
};

#[test]
fn openai_compatible_adapter_builds_json_request_shape() {
    let client = OpenAiCompatibleChatClient::new("gpt-4o", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::text(MessageRole::User, "hello")],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();
    assert_eq!(json["model"], "gpt-4o");
    assert_eq!(json["messages"][0]["role"], "user");
    assert_eq!(json["messages"][0]["content"], "hello");
}

#[test]
fn factory_routes_anthropic_to_anthropic_adapter() {
    let client = create_chat_client(
        BackendType::Anthropic,
        "claude-3-5-sonnet-latest",
        "https://api.anthropic.com",
        "sk-test",
    );
    assert_eq!(client.provider_name(), "anthropic");
}

#[test]
fn factory_routes_anthropic_bedrock_endpoint_to_bedrock_adapter() {
    let client = create_chat_client_from_resolved(ResolvedModelConfig {
        backend: "anthropic".to_string(),
        model: ModelConfig {
            id: "claude-sonnet".to_string(),
            enabled: true,
            endpoints: vec![EndpointBinding::Id("bedrock-anthropic".to_string())],
            context_length: None,
            max_output_tokens: None,
            function_call_available: Some(true),
            response_format_available: Some(false),
            native_multimodal: Some(true),
            protocol: None,
            dimensions: None,
            default_top_n: None,
            request_mapping: None,
            response_mapping: None,
            extra: Default::default(),
        },
        model_id: "global.anthropic.claude-sonnet".to_string(),
        endpoint: EndpointConfig {
            id: "bedrock-anthropic".to_string(),
            enabled: true,
            api_base: Some("https://bedrock-runtime.us-east-1.amazonaws.com".to_string()),
            api_key: None,
            organization: None,
            response_api: false,
            endpoint_type: Some("anthropic_bedrock".to_string()),
            region: Some("us-east-1".to_string()),
            is_azure: false,
            is_bedrock: true,
            is_vertex: false,
            credentials: serde_json::json!({"access_key":"AKIA_TEST","secret_key":"SECRET_TEST"}),
            rpm: 60,
            tpm: 300_000,
            concurrent_requests: 20,
            proxy: None,
            headers: None,
            access_token: None,
            access_token_expires_at: None,
            extra: Default::default(),
        },
    })
    .unwrap();

    assert_eq!(client.provider_name(), "anthropic-bedrock");
}

#[test]
fn factory_routes_openai_vertex_endpoint_to_vertex_adapter() {
    let client = create_chat_client_from_resolved(ResolvedModelConfig {
        backend: "openai".to_string(),
        model: ModelConfig {
            id: "google/gemini-2.5-flash".to_string(),
            enabled: true,
            endpoints: vec![EndpointBinding::Id("vertex-openai".to_string())],
            context_length: None,
            max_output_tokens: None,
            function_call_available: Some(true),
            response_format_available: Some(false),
            native_multimodal: Some(true),
            protocol: None,
            dimensions: None,
            default_top_n: None,
            request_mapping: None,
            response_mapping: None,
            extra: Default::default(),
        },
        model_id: "google/gemini-2.5-flash".to_string(),
        endpoint: EndpointConfig {
            id: "vertex-openai".to_string(),
            enabled: true,
            api_base: Some("https://aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/endpoints/openapi".to_string()),
            api_key: None,
            organization: None,
            response_api: false,
            endpoint_type: Some("openai_vertex".to_string()),
            region: Some("us-central1".to_string()),
            is_azure: false,
            is_bedrock: false,
            is_vertex: true,
            credentials: serde_json::json!({"access_token":"cached-token","access_token_expires_at":4102444800.0}),
            rpm: 60,
            tpm: 300_000,
            concurrent_requests: 20,
            proxy: None,
            headers: None,
            access_token: None,
            access_token_expires_at: None,
            extra: Default::default(),
        },
    })
    .unwrap();

    assert_eq!(client.provider_name(), "openai-vertex");
}

#[test]
fn openai_compatible_adapter_maps_system_assistant_tool_and_options() {
    let client =
        OpenAiCompatibleChatClient::new("fallback-model", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "".to_string(),
        messages: vec![
            Message::text(MessageRole::System, "system"),
            Message::text(MessageRole::Assistant, "assistant"),
            Message {
                role: MessageRole::Tool,
                content: vec![MessageContent::text("tool result")],
                name: None,
                tool_call_id: Some("call-1".to_string()),
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
        ],
        options: ChatRequestOptions {
            temperature: Some(0.2),
            max_tokens: Some(64),
            stream: Some(true),
            ..Default::default()
        },
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["model"], "fallback-model");
    assert_eq!(json["messages"][0]["role"], "system");
    assert_eq!(json["messages"][1]["role"], "assistant");
    assert_eq!(json["messages"][2]["role"], "tool");
    assert_eq!(json["messages"][2]["tool_call_id"], "call-1");
    let temperature = json["temperature"].as_f64().unwrap();
    assert!((temperature - 0.2).abs() < 0.000_001);
    assert_eq!(json["max_tokens"], 64);
    assert_eq!(json["stream"], true);
}

#[test]
fn openai_compatible_adapter_maps_python_style_chat_options() {
    let client =
        OpenAiCompatibleChatClient::new("fallback-model", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "gpt-5.2".to_string(),
        messages: vec![Message::text(MessageRole::User, "Return JSON.")],
        options: ChatRequestOptions {
            temperature: Some(0.1),
            max_tokens: Some(64),
            max_completion_tokens: Some(96),
            stream: Some(false),
            top_p: Some(0.9),
            stop: vec!["END".to_string(), "DONE".to_string()],
            response_format: Some(serde_json::json!({"type": "json_object"})),
            stream_options: Some(serde_json::json!({"include_usage": true})),
            audio: Some(serde_json::json!({"voice": "alloy", "format": "mp3"})),
            frequency_penalty: Some(0.2),
            logit_bias: Some(serde_json::json!({"42": -5})),
            logprobs: Some(true),
            metadata: Some(serde_json::json!({"trace_id": "trace-1"})),
            modalities: Some(serde_json::json!(["text"])),
            n: Some(2),
            parallel_tool_calls: Some(false),
            prediction: Some(serde_json::json!({"type": "content", "content": "done"})),
            presence_penalty: Some(0.3),
            reasoning_effort: Some("medium".to_string()),
            seed: Some(1234),
            service_tier: Some("default".to_string()),
            store: Some(false),
            top_logprobs: Some(3),
            user: Some("user-1".to_string()),
        },
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["model"], "gpt-5.2");
    assert!((json["temperature"].as_f64().unwrap() - 0.1).abs() < 0.000_001);
    assert_eq!(json["max_tokens"], 64);
    assert_eq!(json["max_completion_tokens"], 96);
    assert!((json["top_p"].as_f64().unwrap() - 0.9).abs() < 0.000_001);
    assert_eq!(json["stop"], serde_json::json!(["END", "DONE"]));
    assert_eq!(
        json["response_format"],
        serde_json::json!({"type": "json_object"})
    );
    assert_eq!(
        json["stream_options"],
        serde_json::json!({"include_usage": true})
    );
    assert_eq!(
        json["audio"],
        serde_json::json!({"voice": "alloy", "format": "mp3"})
    );
    assert!((json["frequency_penalty"].as_f64().unwrap() - 0.2).abs() < 0.000_001);
    assert_eq!(json["logit_bias"], serde_json::json!({"42": -5}));
    assert_eq!(json["logprobs"], true);
    assert_eq!(json["metadata"], serde_json::json!({"trace_id": "trace-1"}));
    assert_eq!(json["modalities"], serde_json::json!(["text"]));
    assert_eq!(json["n"], 2);
    assert_eq!(json["parallel_tool_calls"], false);
    assert_eq!(
        json["prediction"],
        serde_json::json!({"type": "content", "content": "done"})
    );
    assert!((json["presence_penalty"].as_f64().unwrap() - 0.3).abs() < 0.000_001);
    assert_eq!(json["reasoning_effort"], "medium");
    assert_eq!(json["seed"], 1234);
    assert_eq!(json["service_tier"], "default");
    assert_eq!(json["store"], false);
    assert_eq!(json["top_logprobs"], 3);
    assert_eq!(json["user"], "user-1");
}

#[test]
fn openai_compatible_adapter_preserves_reasoning_extra_content_and_extra_body() {
    let client =
        OpenAiCompatibleChatClient::new("fallback-model", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages: vec![Message {
            role: MessageRole::Assistant,
            content: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                name: "default_api:list_files".to_string(),
                arguments: r#"{"path":"."}"#.to_string(),
                index: None,
                extra_content: Some(serde_json::json!({
                    "google": {"thought_signature": "sig_123"}
                })),
            }],
            reasoning_content: Some("old-thought".to_string()),
        }],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::json!({
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinkingLevel": "high",
                        "include_thoughts": true
                    }
                }
            }
        }),
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["messages"][0]["role"], "assistant");
    assert_eq!(json["messages"][0]["content"], serde_json::Value::Null);
    assert_eq!(json["messages"][0]["reasoning_content"], "old-thought");
    assert_eq!(
        json["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        "sig_123"
    );
    assert_eq!(
        json["extra_body"]["google"]["thinking_config"]["thinkingLevel"],
        "high"
    );
    assert_eq!(
        json["extra_body"]["google"]["thinking_config"]["include_thoughts"],
        true
    );
}

#[test]
fn openai_compatible_adapter_preserves_empty_reasoning_content() {
    let client =
        OpenAiCompatibleChatClient::new("fallback-model", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages: vec![Message {
            role: MessageRole::Assistant,
            content: Vec::new(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: Some(String::new()),
        }],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["messages"][0]["role"], "assistant");
    assert_eq!(json["messages"][0]["reasoning_content"], "");
}

#[test]
fn openai_completion_json_preserves_reasoning_and_tool_call_extra_content() {
    let response = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-3-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "done",
                "reasoning_content": "hidden",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"a\"}"},
                    "extra_content": {
                        "google": {"thought_signature": "sig_123"}
                    }
                }]
            },
            "finish_reason": "stop"
        }]
    });

    let normalized = OpenAiCompatibleChatClient::normalize_completion_json(response).unwrap();

    assert_eq!(normalized.content, "done");
    assert_eq!(normalized.reasoning_content.as_deref(), Some("hidden"));
    assert_eq!(normalized.tool_calls[0].name, "lookup");
    assert_eq!(
        normalized.tool_calls[0]
            .extra_content
            .as_ref()
            .expect("extra content")["google"]["thought_signature"],
        "sig_123"
    );
}

#[test]
fn anthropic_adapter_extracts_system_prompt_and_user_messages() {
    let client =
        AnthropicChatClient::new("claude-sonnet-4-6", "https://api.anthropic.com", "sk-test");
    let request = ChatRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![
            Message::text(MessageRole::System, "system one"),
            Message::text(MessageRole::System, "system two"),
            Message::text(MessageRole::User, "hello"),
            Message::text(MessageRole::Assistant, "hi"),
        ],
        options: ChatRequestOptions {
            temperature: Some(0.5),
            max_tokens: Some(128),
            stream: Some(false),
            ..Default::default()
        },
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_anthropic_json(&request).unwrap();

    assert_eq!(json["model"], "claude-sonnet-4-6");
    assert_eq!(json["system"], "system one\nsystem two");
    assert_eq!(json["messages"][0]["role"], "user");
    assert_eq!(json["messages"][0]["content"][0]["text"], "hello");
    assert_eq!(json["messages"][1]["role"], "assistant");
    assert_eq!(json["max_tokens"], 128);
    assert_eq!(json["temperature"], 0.5);
}

#[test]
fn anthropic_adapter_maps_python_style_chat_options() {
    let client =
        AnthropicChatClient::new("claude-sonnet-4-6", "https://api.anthropic.com", "sk-test");
    let request = ChatRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![Message::text(MessageRole::User, "hello")],
        options: ChatRequestOptions {
            max_tokens: Some(128),
            top_p: Some(0.8),
            stop: vec!["END".to_string(), "DONE".to_string()],
            ..Default::default()
        },
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_anthropic_json(&request).unwrap();

    assert!((json["top_p"].as_f64().unwrap() - 0.8).abs() < 0.000_001);
    assert_eq!(json["stop_sequences"], serde_json::json!(["END", "DONE"]));
}

#[test]
fn anthropic_adapter_maps_image_content_for_native_multimodal_models() {
    let client =
        AnthropicChatClient::new("claude-sonnet-4-6", "https://api.anthropic.com", "sk-test");
    let request = ChatRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![Message {
            role: MessageRole::User,
            content: vec![
                MessageContent::text("describe this image"),
                MessageContent::ImageUrl {
                    url: "data:image/png;base64,AAAA".to_string(),
                },
            ],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        }],
        options: ChatRequestOptions {
            max_tokens: Some(128),
            ..Default::default()
        },
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_anthropic_json(&request).unwrap();

    assert_eq!(json["messages"][0]["content"][0]["type"], "text");
    assert_eq!(json["messages"][0]["content"][1]["type"], "image");
    assert_eq!(
        json["messages"][0]["content"][1]["source"]["media_type"],
        "image/png"
    );
    assert_eq!(json["messages"][0]["content"][1]["source"]["data"], "AAAA");
}

#[test]
fn anthropic_adapter_maps_tools_and_multi_turn_tool_messages() {
    let client =
        AnthropicChatClient::new("claude-sonnet-4-6", "https://api.anthropic.com", "sk-test");
    let request = ChatRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![
            Message::text(MessageRole::User, "What is the weather?"),
            Message {
                role: MessageRole::Assistant,
                content: Vec::new(),
                name: None,
                tool_call_id: None,
                tool_calls: vec![ToolCall::function(
                    "toolu_1",
                    "get_current_weather",
                    r#"{"location":"New York"}"#,
                )],
                reasoning_content: None,
            },
            Message {
                role: MessageRole::Tool,
                content: vec![MessageContent::text("72F and sunny")],
                name: None,
                tool_call_id: Some("toolu_1".to_string()),
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
        ],
        options: ChatRequestOptions {
            max_tokens: Some(128),
            ..Default::default()
        },
        tools: vec![ChatTool::function(
            "get_current_weather",
            "Get the current weather in a given location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        )],
        tool_choice: Some("auto".to_string()),
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_anthropic_json(&request).unwrap();

    assert_eq!(json["tools"][0]["name"], "get_current_weather");
    assert_eq!(json["tool_choice"]["type"], "auto");
    assert_eq!(json["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(json["messages"][1]["content"][0]["id"], "toolu_1");
    assert_eq!(
        json["messages"][1]["content"][0]["input"]["location"],
        "New York"
    );
    assert_eq!(json["messages"][2]["role"], "user");
    assert_eq!(json["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(json["messages"][2]["content"][0]["tool_use_id"], "toolu_1");
}

#[test]
fn anthropic_adapter_preserves_cache_control_extensions() {
    let client =
        AnthropicChatClient::new("claude-sonnet-4-6", "https://api.anthropic.com", "sk-test");
    let request = ChatRequest {
        model: "claude-sonnet-4-6".to_string(),
        messages: vec![
            Message {
                role: MessageRole::System,
                content: vec![MessageContent::Text {
                    text: "stable system block".to_string(),
                    cache_control: Some(serde_json::json!({"type": "ephemeral"})),
                }],
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
            Message {
                role: MessageRole::User,
                content: vec![MessageContent::text("hello")],
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
        ],
        options: ChatRequestOptions {
            max_tokens: Some(128),
            ..Default::default()
        },
        tools: vec![ChatTool {
            name: "stable_tool".to_string(),
            description: Some("Stable tool schema".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                }
            }),
            cache_control: Some(serde_json::json!({"type": "ephemeral"})),
        }],
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_anthropic_json(&request).unwrap();

    assert_eq!(
        json["system"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral"})
    );
    assert_eq!(
        json["tools"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral"})
    );
}

#[test]
fn openai_adapter_maps_image_content_for_vision_models() {
    let client =
        OpenAiCompatibleChatClient::new("qwen3-vl-flash", "https://example.com/v1", "sk-test");
    let request = ChatRequest {
        model: "qwen3-vl-flash".to_string(),
        messages: vec![Message {
            role: MessageRole::User,
            content: vec![
                MessageContent::text("describe this image"),
                MessageContent::ImageUrl {
                    url: "data:image/png;base64,AAAA".to_string(),
                },
            ],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        }],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["messages"][0]["content"][0]["type"], "text");
    assert_eq!(json["messages"][0]["content"][1]["type"], "image_url");
    assert_eq!(
        json["messages"][0]["content"][1]["image_url"]["url"],
        "data:image/png;base64,AAAA"
    );
}

#[test]
fn openai_adapter_maps_tools_and_tool_choice() {
    let client =
        OpenAiCompatibleChatClient::new("deepseek-chat", "https://example.com/v1", "sk-test");
    let request = ChatRequest {
        model: "deepseek-chat".to_string(),
        messages: vec![Message::text(
            MessageRole::User,
            "What is the weather in New York?",
        )],
        options: Default::default(),
        tools: vec![ChatTool::function(
            "get_current_weather",
            "Get the current weather in a given location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        )],
        tool_choice: Some("auto".to_string()),
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["tools"][0]["type"], "function");
    assert_eq!(json["tools"][0]["function"]["name"], "get_current_weather");
    assert_eq!(json["tool_choice"], "auto");
}

#[test]
fn openai_adapter_maps_multi_turn_tool_messages() {
    let client =
        OpenAiCompatibleChatClient::new("deepseek-chat", "https://example.com/v1", "sk-test");
    let request = ChatRequest {
        model: "deepseek-chat".to_string(),
        messages: vec![
            Message::text(MessageRole::User, "What is the weather?"),
            Message {
                role: MessageRole::Assistant,
                content: Vec::new(),
                name: None,
                tool_call_id: None,
                tool_calls: vec![ToolCall::function(
                    "call_1",
                    "get_current_weather",
                    r#"{"location":"New York"}"#,
                )],
                reasoning_content: None,
            },
            Message {
                role: MessageRole::Tool,
                content: vec![MessageContent::text("72F and sunny")],
                name: None,
                tool_call_id: Some("call_1".to_string()),
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
        ],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
        extra_body: serde_json::Value::Null,
    };

    let json = client.to_openai_json(&request).unwrap();

    assert_eq!(json["messages"][1]["role"], "assistant");
    assert_eq!(json["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(
        json["messages"][1]["tool_calls"][0]["function"]["name"],
        "get_current_weather"
    );
    assert_eq!(json["messages"][2]["role"], "tool");
    assert_eq!(json["messages"][2]["tool_call_id"], "call_1");
}

#[test]
fn openai_stream_chunk_json_normalizes_content_tools_and_usage() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "hel",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\""}
                }]
            },
            "finish_reason": null
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.content, "hel");
    assert_eq!(delta.tool_calls[0].id, "call_1");
    assert_eq!(delta.tool_calls[0].name, "lookup");
    assert_eq!(delta.tool_calls[0].arguments, "{\"q\"");
    assert_eq!(delta.tool_calls[0].index, Some(0));
    let usage = delta.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(3));
    assert_eq!(usage.completion_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(7));
}

#[test]
fn openai_stream_chunk_json_preserves_tool_call_index() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 2,
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\""}
                }]
            },
            "finish_reason": null
        }]
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.tool_calls[0].index, Some(2));
}

#[test]
fn openai_stream_chunk_json_preserves_tool_call_extra_content() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gemini-3-pro",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\""},
                    "extra_content": {
                        "google": {"thought_signature": "sig_123"}
                    }
                }]
            },
            "finish_reason": null
        }]
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.tool_calls[0].id, "call_1");
    assert_eq!(
        delta.tool_calls[0]
            .extra_content
            .as_ref()
            .expect("extra content")["google"]["thought_signature"],
        "sig_123"
    );
}

#[test]
fn openai_stream_chunk_json_extracts_reasoning_tags() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "delta": {"content": "answer <think>hidden reasoning</think> final"},
            "finish_reason": null
        }]
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.content, "answer  final");
    assert_eq!(delta.reasoning_content, "hidden reasoning");
}

#[test]
fn openai_stream_chunk_json_preserves_reasoning_content_field() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-v4-pro",
        "choices": [{
            "index": 0,
            "delta": {"reasoning_content": "step-1"},
            "finish_reason": null
        }]
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.reasoning_content, "step-1");
}

#[test]
fn gemini_stream_chunk_json_extracts_thought_tags() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gemini-3-pro",
        "choices": [{
            "index": 0,
            "delta": {"content": "<thought>hidden</thought>visible"},
            "finish_reason": null
        }]
    });

    let delta = OpenAiCompatibleChatClient::normalize_stream_chunk_json(chunk).unwrap();

    assert_eq!(delta.content, "visible");
    assert_eq!(delta.reasoning_content, "hidden");
}

#[tokio::test]
async fn openai_compatible_completion_uses_json_path_for_empty_reasoning_content() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        let body = request.split("\r\n\r\n").nth(1).expect("request body");
        let json: serde_json::Value = serde_json::from_str(body).unwrap();

        assert!(request.starts_with("POST /chat/completions "));
        assert_eq!(json["messages"][0]["role"], "assistant");
        assert_eq!(json["messages"][0]["reasoning_content"], "");

        let response = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            response.len(),
            response
        );
        use tokio::io::AsyncWriteExt;
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let client = OpenAiCompatibleChatClient::new("fallback-model", api_base, "sk-test");
    let response = client
        .create_completion(ChatRequest {
            model: "deepseek-v4-pro".to_string(),
            messages: vec![Message {
                role: MessageRole::Assistant,
                content: Vec::new(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: Some(String::new()),
            }],
            options: Default::default(),
            tools: Vec::new(),
            tool_choice: None,
            extra_body: serde_json::Value::Null,
        })
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(response.content, "ok");
}

#[tokio::test]
async fn anthropic_direct_completion_uses_json_path_for_cache_control_and_tools() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        let body = request.split("\r\n\r\n").nth(1).expect("request body");
        let json: serde_json::Value = serde_json::from_str(body).unwrap();

        assert!(request.starts_with("POST /v1/messages "));
        assert_eq!(
            json["messages"][0]["content"][0]["cache_control"],
            serde_json::json!({"type": "ephemeral"})
        );
        assert_eq!(
            json["tools"][0]["cache_control"],
            serde_json::json!({"type": "ephemeral"})
        );
        assert_eq!(
            json["thinking"],
            serde_json::json!({"type": "enabled", "budget_tokens": 16000})
        );

        let response = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "toolu_1", "name": "stable_tool", "input": {"value": "seen"}}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 5, "output_tokens": 7}
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            response.len(),
            response
        );
        use tokio::io::AsyncWriteExt;
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let client = AnthropicChatClient::new("claude-sonnet-4-6", api_base, "sk-test");
    let response = client
        .create_completion(ChatRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![Message {
                role: MessageRole::User,
                content: vec![MessageContent::text_with_cache_control(
                    "stable prompt block",
                    serde_json::json!({"type": "ephemeral"}),
                )],
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
            }],
            options: ChatRequestOptions {
                max_tokens: Some(128),
                ..Default::default()
            },
            tools: vec![ChatTool::function(
                "stable_tool",
                "Stable tool schema",
                serde_json::json!({"type": "object"}),
            )
            .with_cache_control(serde_json::json!({"type": "ephemeral"}))],
            tool_choice: None,
            extra_body: serde_json::json!({
                "thinking": {"type": "enabled", "budget_tokens": 16000}
            }),
        })
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(response.content, "ok");
    assert_eq!(response.tool_calls[0].name, "stable_tool");
    assert_eq!(response.tool_calls[0].arguments, r#"{"value":"seen"}"#);
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(5));
    assert_eq!(usage.completion_tokens, Some(7));
    assert_eq!(usage.total_tokens, Some(12));
}

#[tokio::test]
async fn chat_client_trait_supports_stream_return_type() {
    struct StaticStreamClient;

    #[async_trait::async_trait]
    impl vv_llm::ChatClient for StaticStreamClient {
        fn provider_name(&self) -> &'static str {
            "static"
        }

        async fn create_completion(
            &self,
            _request: ChatRequest,
        ) -> Result<vv_llm::ChatResponse, vv_llm::VvLlmError> {
            unreachable!("stream test only")
        }

        async fn create_stream(
            &self,
            _request: ChatRequest,
        ) -> Result<vv_llm::ChatStream, vv_llm::VvLlmError> {
            Ok(Box::pin(futures_util::stream::iter([Ok(
                ChatStreamDelta {
                    content: "pong".to_string(),
                    ..Default::default()
                },
            )])))
        }
    }

    let client = StaticStreamClient;
    let mut stream = client
        .create_stream(ChatRequest::new(
            "unit",
            vec![Message::text(MessageRole::User, "ping")],
        ))
        .await
        .unwrap();
    let delta = stream.next().await.unwrap().unwrap();
    assert_eq!(delta.content, "pong");
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = socket.read(&mut chunk).await.unwrap();
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if http_request_is_complete(&buffer) {
            break;
        }
    }
    String::from_utf8(buffer).unwrap()
}

fn http_request_is_complete(buffer: &[u8]) -> bool {
    let Some(header_end) = buffer.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let header = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = header
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .or_else(|| {
            header
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length:"))
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    buffer.len() >= header_end + 4 + content_length
}

#[tokio::test]
async fn vertex_token_provider_reuses_valid_cached_token() {
    let provider = GoogleAccessTokenProvider::new(serde_json::json!({
        "access_token": "cached-token",
        "access_token_expires_at": 4102444800.0
    }))
    .unwrap();

    let token = provider.access_token().await.unwrap();

    assert_eq!(token.token, "cached-token");
    assert_eq!(token.expires_at, 4102444800.0);
}
