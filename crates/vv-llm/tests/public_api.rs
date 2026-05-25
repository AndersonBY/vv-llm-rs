use vv_llm::{BackendType, ChatRequest, Message, MessageRole};

#[test]
fn public_api_exposes_backend_and_message_types() {
    let backend = BackendType::OpenAI;
    assert_eq!(backend.as_str(), "openai");

    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::text(MessageRole::User, "hello")],
        options: Default::default(),
    };

    assert_eq!(request.messages[0].role, MessageRole::User);
    assert_eq!(request.messages[0].text_content().as_deref(), Some("hello"));
}
