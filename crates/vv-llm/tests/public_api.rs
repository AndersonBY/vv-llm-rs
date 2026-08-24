use vv_llm::{
    BackendType, ChatRequest, ChatRequestOptions, ChatStreamDelta, ChatUsage, Message, MessageRole,
    ThinkingPreference, ToolCall,
};

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
        extra_body: serde_json::Value::Null,
        extensions: Default::default(),
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

#[test]
fn typed_thinking_preferences_preserve_legacy_wire_values() {
    let default_options = ChatRequestOptions::default().with_thinking(ThinkingPreference::Default);
    assert_eq!(default_options.thinking, None);

    let enabled =
        ChatRequestOptions::default().with_thinking(ThinkingPreference::enabled_with_budget(4096));
    assert_eq!(
        enabled.thinking,
        Some(serde_json::json!({
            "type": "enabled",
            "budget_tokens": 4096
        }))
    );

    let request = ChatRequest::new(
        "deepseek-v4-flash",
        vec![Message::text(MessageRole::User, "hello")],
    )
    .with_thinking(ThinkingPreference::Disabled);
    assert_eq!(
        request.options.thinking,
        Some(serde_json::json!({"type": "disabled"}))
    );
}

#[test]
fn chat_usage_keeps_legacy_json_compatible_and_optional_cache_values_distinct() {
    let legacy_json = serde_json::json!({
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18
    });
    let legacy: ChatUsage = serde_json::from_value(legacy_json.clone()).unwrap();

    assert_eq!(legacy.input_tokens, None);
    assert_eq!(legacy.output_tokens, None);
    assert_eq!(legacy.cache_read_input_tokens, None);
    assert_eq!(legacy.cache_creation_input_tokens, None);
    assert_eq!(legacy.raw_usage, None);
    assert_eq!(serde_json::to_value(legacy).unwrap(), legacy_json);

    let explicit_zero = ChatUsage {
        cache_read_input_tokens: Some(0),
        cache_creation_input_tokens: Some(0),
        ..Default::default()
    };
    let explicit_zero = serde_json::to_value(explicit_zero).unwrap();

    assert_eq!(explicit_zero["cache_read_input_tokens"], 0);
    assert_eq!(explicit_zero["cache_creation_input_tokens"], 0);
}
