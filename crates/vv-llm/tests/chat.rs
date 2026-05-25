use vv_llm::{
    chat_clients::{create_chat_client, OpenAiCompatibleChatClient},
    BackendType, ChatRequest, Message, MessageRole,
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
