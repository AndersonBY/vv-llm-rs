use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use vv_llm::{
    utilities::{
        calculate_image_tokens, count_message_tokens, count_tokens, count_tokens_fallback,
        count_tokens_with_settings, cutoff_messages, normalize_text_messages, RetryPolicy,
    },
    ChatTool, LlmSettings, Message, MessageContent, MessageRole,
};

#[derive(Debug)]
struct CapturedRequest {
    path: String,
    headers: String,
    body: String,
}

async fn spawn_json_server(
    response_body: &'static str,
) -> (String, tokio::task::JoinHandle<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0u8; 1024];

        loop {
            let read = stream.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
            if let Some((headers, body)) = split_http_request(&bytes) {
                let content_length = parse_content_length(headers);
                if body.len() >= content_length {
                    break;
                }
            }
        }

        let request = String::from_utf8(bytes).unwrap();
        let (headers, body) = request.split_once("\r\n\r\n").unwrap();
        let path = headers
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .to_string();

        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).await.unwrap();

        CapturedRequest {
            path,
            headers: headers.to_string(),
            body: body.to_string(),
        }
    });

    (format!("http://{addr}"), handle)
}

fn split_http_request(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let marker = bytes.windows(4).position(|window| window == b"\r\n\r\n")?;
    let headers = std::str::from_utf8(&bytes[..marker]).ok()?;
    Some((headers, &bytes[marker + 4..]))
}

fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0)
}

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
        count_tokens("antidisestablishmentarianism", "unknown-provider-model").unwrap(),
        6
    );
}

#[test]
fn token_counter_matches_python_provider_fallbacks() {
    assert_eq!(count_tokens("abcd", "MiniMax-M2.7").unwrap(), 3);
    assert_eq!(count_tokens("hello world", "kimi-k2.5").unwrap(), 2);
    assert_eq!(count_tokens("hello world", "gemini-3-flash").unwrap(), 2);
    assert_eq!(count_tokens("hello world", "claude-opus-4-8").unwrap(), 2);
    assert!(count_tokens("你好，世界", "qwen3.7-max").unwrap() > 1);
    assert!(count_tokens("你好，世界", "deepseek-chat").unwrap() > 1);
}

#[tokio::test]
async fn async_token_counter_prefers_configured_token_server() {
    let (base_url, request_handle) = spawn_json_server(r#"{"total_tokens":37}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "token_server": {{"host":"127.0.0.1","port":8338,"url":"{base_url}"}}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello token server", "gpt-4o")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 37);
    assert_eq!(captured.path, "/count_tokens");
    assert!(captured.body.contains(r#""text":"hello token server""#));
    assert!(captured.body.contains(r#""model":"gpt-4o""#));
}

#[tokio::test]
async fn async_token_counter_uses_minimax_tokenizer_endpoint() {
    let (base_url, request_handle) = spawn_json_server(r#"{"segments_num":11}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"minimax-default","api_base":"{base_url}/v1","api_key":"minimax-key"}}],
          "backends": {{
            "minimax": {{
              "models": {{"MiniMax-M2.7": {{"id":"MiniMax-M2.7","endpoints":["minimax-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello minimax", "MiniMax-M2.7")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 11);
    assert_eq!(captured.path, "/v1/tokenize");
    assert!(captured
        .headers
        .contains("authorization: Bearer minimax-key"));
    assert!(captured.body.contains(r#""sender_type":"USER""#));
    assert!(captured.body.contains(r#""text":"hello minimax""#));
}

#[tokio::test]
async fn async_token_counter_uses_moonshot_tokenizer_endpoint() {
    let (base_url, request_handle) = spawn_json_server(r#"{"data":{"total_tokens":23}}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"moonshot-default","api_base":"{base_url}/v1","api_key":"moonshot-key"}}],
          "backends": {{
            "moonshot": {{
              "models": {{"kimi-k2.6": {{"id":"kimi-k2.6","endpoints":["moonshot-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello moonshot", "kimi-k2.6")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 23);
    assert_eq!(captured.path, "/v1/tokenizers/estimate-token-count");
    assert!(captured
        .headers
        .contains("authorization: Bearer moonshot-key"));
    assert!(captured.body.contains(r#""model":"kimi-k2.6""#));
    assert!(captured.body.contains(r#""content":"hello moonshot""#));
}

#[tokio::test]
async fn async_token_counter_uses_gemini_count_tokens_endpoint() {
    let (base_url, request_handle) = spawn_json_server(r#"{"totalTokens":29}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"gemini-default","api_base":"{base_url}/v1beta/openai/","api_key":"gemini-key"}}],
          "backends": {{
            "gemini": {{
              "models": {{"gemini-3.5-flash": {{"id":"gemini-3.5-flash","endpoints":["gemini-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello gemini", "gemini-3.5-flash")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 29);
    assert_eq!(
        captured.path,
        "/v1beta/models/gemini-2.5-pro:countTokens?key=gemini-key"
    );
    assert!(captured.body.contains(r#""role":"USER""#));
    assert!(captured.body.contains(r#""text":"hello gemini""#));
}

#[tokio::test]
async fn async_token_counter_uses_stepfun_tokenizer_endpoint() {
    let (base_url, request_handle) = spawn_json_server(r#"{"data":{"total_tokens":31}}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"stepfun-default","api_base":"{base_url}/api","api_key":"stepfun-key"}}],
          "backends": {{
            "stepfun": {{
              "models": {{"step-3.5-flash": {{"id":"step-3.5-flash","endpoints":["stepfun-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello stepfun", "step-3.5-flash")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 31);
    assert_eq!(captured.path, "/api/token/count");
    assert!(captured
        .headers
        .contains("authorization: Bearer stepfun-key"));
    assert!(captured.body.contains(r#""model":"step-3.5-flash""#));
}

#[tokio::test]
async fn async_token_counter_uses_zhipu_tokenizer_endpoint_and_supported_model_alias() {
    let (base_url, request_handle) = spawn_json_server(r#"{"usage":{"prompt_tokens":41}}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"zhipuai-default","api_base":"{base_url}/api/paas/v4","api_key":"zhipu-key"}}],
          "backends": {{
            "zhipuai": {{
              "models": {{"glm-5.1": {{"id":"glm-5.1","endpoints":["zhipuai-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello glm", "glm-5.1")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 41);
    assert_eq!(captured.path, "/api/paas/v4/tokenizer");
    assert!(captured.headers.contains("authorization: Bearer zhipu-key"));
    assert!(captured.body.contains(r#""model":"glm-4-plus""#));
}

#[tokio::test]
async fn async_token_counter_uses_anthropic_count_tokens_endpoint() {
    let (base_url, request_handle) = spawn_json_server(r#"{"input_tokens":47}"#).await;
    let settings = LlmSettings::from_json_str(&format!(
        r#"{{
          "endpoints": [{{"id":"anthropic-default","api_base":"{base_url}","api_key":"anthropic-key","endpoint_type":"default"}}],
          "backends": {{
            "anthropic": {{
              "models": {{"claude-opus-4-8": {{"id":"claude-opus-4-8","endpoints":["anthropic-default"]}}}}
            }}
          }}
        }}"#
    ))
    .unwrap();

    let tokens = count_tokens_with_settings(&settings, "hello anthropic", "claude-opus-4-8")
        .await
        .unwrap();
    let captured = request_handle.await.unwrap();

    assert_eq!(tokens, 47);
    assert_eq!(captured.path, "/v1/messages/count_tokens");
    assert!(captured.headers.contains("x-api-key: anthropic-key"));
    assert!(captured.headers.contains("anthropic-version: 2023-06-01"));
    assert!(captured.body.contains(r#""model":"claude-opus-4-8""#));
    assert!(captured.body.contains(r#""content":"hello anthropic""#));
}

#[test]
fn image_token_counter_matches_python_formula() {
    assert_eq!(calculate_image_tokens(2048, 2048, "gpt-4o"), 765);
    assert_eq!(calculate_image_tokens(4096, 1024, "gpt-4o"), 765);
    assert_eq!(calculate_image_tokens(640, 480, "gpt-4o"), 425);
    assert_eq!(calculate_image_tokens(2048, 2048, "moonshot-v1-8k"), 1024);
}

#[test]
fn message_token_counter_counts_text_images_and_tools() {
    let messages = vec![
        Message::text(MessageRole::System, "system prompt"),
        Message {
            role: MessageRole::User,
            content: vec![
                MessageContent::text("describe"),
                MessageContent::ImageUrl {
                    url: "data:image/png;base64,AAAA".to_string(),
                },
            ],
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            reasoning_content: None,
        },
    ];
    let tools = vec![ChatTool::function(
        "lookup",
        "Lookup a value",
        serde_json::json!({"type": "object"}),
    )];

    let text_only = count_tokens("system prompt\ndescribe", "gpt-4o").unwrap();
    let tool_tokens = count_tokens(&serde_json::to_string(&tools).unwrap(), "gpt-4o").unwrap();

    assert_eq!(
        count_message_tokens(&messages, &tools, "gpt-4o", true).unwrap(),
        text_only + calculate_image_tokens(2048, 2048, "gpt-4o") + tool_tokens
    );
    assert_eq!(
        count_message_tokens(&messages, &[], "gpt-4o", false).unwrap(),
        text_only + 1
    );
}

#[test]
fn cutoff_messages_preserves_system_and_keeps_recent_messages() {
    let messages = vec![
        Message::text(MessageRole::System, "system"),
        Message::text(MessageRole::User, "old old old old old"),
        Message::text(MessageRole::Assistant, "recent"),
        Message::text(MessageRole::User, "latest"),
    ];

    let truncated = cutoff_messages(messages, 3, "gpt-4o").unwrap();

    assert_eq!(truncated.first().unwrap().role, MessageRole::System);
    assert_eq!(truncated.len(), 2);
    assert_eq!(truncated[1].text_content().as_deref(), Some("latest"));
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
fn retry_policy_classifies_legacy_and_structured_errors() {
    use vv_llm::{ErrorDetails, ErrorKind, VvLlmError};

    let policy = RetryPolicy::new(3);
    assert!(policy.should_retry(&VvLlmError::Http("offline".to_string()), 1));
    assert!(!policy.should_retry(&VvLlmError::Configuration("bad".to_string()), 1));
    assert!(!policy.should_retry(
        &VvLlmError::Classified(Box::new(ErrorDetails::new(
            ErrorKind::Authentication,
            "unauthorized",
        ))),
        1,
    ));
}

#[test]
fn legacy_provider_errors_keep_auth_and_rate_limits_distinct() {
    use vv_llm::{ErrorKind, VvLlmError};

    assert_eq!(
        VvLlmError::Provider("401 Unauthorized: invalid API key".to_string()).kind(),
        ErrorKind::Authentication,
    );
    assert_eq!(
        VvLlmError::Provider("status 429 Too Many Requests".to_string()).kind(),
        ErrorKind::RateLimited,
    );
    assert_eq!(
        VvLlmError::Http("request timed out".to_string()).kind(),
        ErrorKind::Timeout,
    );
    assert_eq!(
        VvLlmError::from_status(400, "maximum context length exceeded").kind(),
        ErrorKind::ContextLength,
    );
}

#[tokio::test]
async fn retry_executor_retries_transient_errors() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;
    use vv_llm::{execute_with_retry, ErrorDetails, ErrorKind, VvLlmError};

    let attempts = AtomicU32::new(0);
    let result = execute_with_retry(
        || async {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
            if attempt < 3 {
                Err(VvLlmError::Classified(Box::new(
                    ErrorDetails::new(ErrorKind::RateLimited, "slow down").with_retry_after(0.0),
                )))
            } else {
                Ok("ok")
            }
        },
        RetryPolicy::new(3)
            .with_max_delay(Duration::ZERO)
            .with_jitter_ratio(0.0),
    )
    .await;

    assert_eq!(result.unwrap(), "ok");
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_executor_does_not_retry_authentication_errors() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use vv_llm::{execute_with_retry, ErrorKind, VvLlmError};

    let attempts = AtomicU32::new(0);
    let error = execute_with_retry(
        || async {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err::<(), _>(VvLlmError::classified(
                ErrorKind::Authentication,
                "unauthorized",
            ))
        },
        RetryPolicy::new(3),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
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
