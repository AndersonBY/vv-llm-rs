use vv_llm::{BackendType, ChatRequest, ChatStreamDelta, Message, MessageRole, ToolCall};

#[test]
fn public_api_exposes_backend_and_message_types() {
    let backend = BackendType::OpenAI;
    assert_eq!(backend.as_str(), "openai");

    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::text(MessageRole::User, "hello")],
        options: Default::default(),
        tools: Vec::new(),
        tool_choice: None,
    };

    assert_eq!(request.messages[0].role, MessageRole::User);
    assert_eq!(request.messages[0].text_content().as_deref(), Some("hello"));
}

#[test]
fn public_api_exposes_normalized_stream_delta_type() {
    let delta = ChatStreamDelta {
        content: "hel".to_string(),
        reasoning_content: "thinking".to_string(),
        tool_calls: vec![ToolCall::function("call_1", "lookup", r#"{"q":"a"}"#)],
        usage: None,
        raw_content: Some(serde_json::json!({"provider":"unit"})),
        done: false,
    };

    assert_eq!(delta.content, "hel");
    assert_eq!(delta.reasoning_content, "thinking");
    assert_eq!(delta.tool_calls[0].name, "lookup");
    assert_eq!(delta.raw_content.unwrap()["provider"], "unit");
    assert!(!delta.done);
}
