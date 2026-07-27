use async_trait::async_trait;
use futures_util::stream;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use vv_llm::{
    ChatClient, ChatMiddlewareV1, ChatRequest, ChatResponse, ChatStream, ErrorKind, Message,
    MessageRole, MiddlewareChatClient, MiddlewareContext, RetryPolicy, VvLlmError,
};

struct FlakyClient {
    attempts: Arc<Mutex<u32>>,
}

#[async_trait]
impl ChatClient for FlakyClient {
    fn provider_name(&self) -> &'static str {
        "test"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        let mut attempts = self.attempts.lock().unwrap();
        *attempts += 1;
        if *attempts == 1 {
            return Err(VvLlmError::classified(
                ErrorKind::ProviderInternal,
                "temporary",
            ));
        }
        Ok(ChatResponse {
            id: "response".to_string(),
            model: request.model,
            content: "ok".to_string(),
            tool_calls: Vec::new(),
            reasoning_content: None,
            usage: None,
        })
    }

    async fn create_stream(&self, _request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        Ok(Box::pin(stream::empty()))
    }
}

struct RecordingMiddleware {
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ChatMiddlewareV1 for RecordingMiddleware {
    async fn on_request(
        &self,
        context: &mut MiddlewareContext,
        mut request: ChatRequest,
    ) -> Result<ChatRequest, VvLlmError> {
        context
            .attributes
            .insert("trace_id".to_string(), serde_json::json!("trace-1"));
        self.events.lock().unwrap().push("request:0".to_string());
        request.model = "rewritten".to_string();
        Ok(request)
    }

    async fn on_response(
        &self,
        context: &MiddlewareContext,
        mut response: ChatResponse,
    ) -> Result<ChatResponse, VvLlmError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("response:{}", context.attempt));
        response.content = format!("{}:{}", response.content, context.attributes["trace_id"]);
        Ok(response)
    }

    async fn on_error(&self, context: &MiddlewareContext, error: &VvLlmError) {
        self.events
            .lock()
            .unwrap()
            .push(format!("{:?}:{}", error.kind(), context.attempt));
    }
}

#[tokio::test]
async fn middleware_wraps_request_retry_and_response() {
    let attempts = Arc::new(Mutex::new(0));
    let events = Arc::new(Mutex::new(Vec::new()));
    let client = MiddlewareChatClient::new(
        Box::new(FlakyClient {
            attempts: attempts.clone(),
        }),
        vec![Arc::new(RecordingMiddleware {
            events: events.clone(),
        })],
    )
    .unwrap()
    .with_retry_policy(
        RetryPolicy::new(2)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO),
    );

    let response = client
        .create_completion(ChatRequest::new(
            "original",
            vec![Message::text(MessageRole::User, "hello")],
        ))
        .await
        .unwrap();

    assert_eq!(response.content, "ok:\"trace-1\"");
    assert_eq!(*attempts.lock().unwrap(), 2);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["request:0", "ProviderInternal:1", "response:2"]
    );
}

#[tokio::test]
async fn middleware_returns_metadata_without_changing_chat_response() {
    let client = MiddlewareChatClient::new(
        Box::new(FlakyClient {
            attempts: Arc::new(Mutex::new(1)),
        }),
        Vec::new(),
    )
    .unwrap();

    let result = client
        .create_with_metadata(ChatRequest::new(
            "original",
            vec![Message::text(MessageRole::User, "hello")],
        ))
        .await
        .unwrap();

    assert_eq!(result.response.content, "ok");
    assert_eq!(result.metadata.provider.as_deref(), Some("test"));
    assert_eq!(result.metadata.model.as_deref(), Some("original"));
    assert_eq!(result.metadata.response_id.as_deref(), Some("response"));
    assert_eq!(result.metadata.attempts, 1);
    assert!(result.metadata.latency_ms.is_some());
}

#[test]
fn middleware_rejects_unknown_api_versions() {
    struct V2;

    #[async_trait]
    impl ChatMiddlewareV1 for V2 {
        fn api_version(&self) -> &'static str {
            "v2"
        }
    }

    let result = MiddlewareChatClient::new(
        Box::new(FlakyClient {
            attempts: Arc::new(Mutex::new(0)),
        }),
        vec![Arc::new(V2)],
    );
    assert!(matches!(result, Err(VvLlmError::Configuration(_))));
}
