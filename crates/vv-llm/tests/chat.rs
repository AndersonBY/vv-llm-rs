use vv_llm::{
    chat_clients::{create_chat_client, AnthropicChatClient, OpenAiCompatibleChatClient},
    BackendType, ChatRequest, ChatRequestOptions, Message, MessageContent, MessageRole,
};

#[test]
fn openai_compatible_adapter_builds_json_request_shape() {
    let client = OpenAiCompatibleChatClient::new("gpt-4o", "https://api.openai.com/v1", "sk-test");
    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::text(MessageRole::User, "hello")],
        options: Default::default(),
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
                content: vec![MessageContent::Text {
                    text: "tool result".to_string(),
                }],
                name: None,
                tool_call_id: Some("call-1".to_string()),
            },
        ],
        options: ChatRequestOptions {
            temperature: Some(0.2),
            max_tokens: Some(64),
            stream: Some(true),
        },
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
        },
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
