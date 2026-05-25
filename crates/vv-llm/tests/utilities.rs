use vv_llm::{
    utilities::{count_tokens_fallback, normalize_text_messages, RetryPolicy},
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
fn retry_policy_reports_attempt_count() {
    let policy = RetryPolicy::new(3);
    assert_eq!(policy.max_attempts(), 3);
}
