use vv_llm::{
    utilities::{count_tokens, count_tokens_fallback, normalize_text_messages, RetryPolicy},
    Message, MessageRole,
};

#[test]
fn normalizes_adjacent_text_messages_by_role() {
    let messages = vec![
        Message::text(MessageRole::User, "hello"),
        Message::text(MessageRole::User, "world"),
    ];

    let normalized = normalize_text_messages(messages);
    assert_eq!(normalized.len(), 1);
    assert_eq!(
        normalized[0].text_content().as_deref(),
        Some("hello\nworld")
    );
}

#[test]
fn fallback_token_counter_is_deterministic() {
    assert_eq!(count_tokens_fallback("hello world"), 2);
    assert_eq!(count_tokens_fallback(""), 0);
}

#[test]
fn tiktoken_counter_matches_openai_encodings() {
    assert_eq!(count_tokens("hello world", "gpt-3.5-turbo").unwrap(), 2);
    assert_eq!(count_tokens("hello world", "gpt-4o").unwrap(), 2);
    assert_eq!(
        count_tokens("antidisestablishmentarianism", "gpt-4o").unwrap(),
        6
    );
}

#[test]
fn token_counter_falls_back_for_unknown_models() {
    assert_eq!(
        count_tokens("hello world", "unknown-provider-model").unwrap(),
        2
    );
}

#[test]
fn retry_policy_reports_attempt_count() {
    let policy = RetryPolicy::new(3);
    assert_eq!(policy.max_attempts(), 3);
}

#[test]
fn retry_policy_never_allows_zero_attempts() {
    let policy = RetryPolicy::new(0);
    assert_eq!(policy.max_attempts(), 1);
}

#[test]
fn normalization_does_not_merge_different_roles_or_images() {
    let messages = vec![
        Message::text(MessageRole::User, "hello"),
        Message::text(MessageRole::Assistant, "world"),
        Message {
            role: MessageRole::Assistant,
            content: vec![vv_llm::MessageContent::ImageUrl {
                url: "https://example.com/cat.png".to_string(),
            }],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_text_messages(messages);
    assert_eq!(normalized.len(), 3);
}
