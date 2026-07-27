use futures_util::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use vv_llm::{
    ChatClient, ChatRequest, ChatResponse, ChatStreamDelta, ErrorKind, Message, MessageRole,
    MiddlewareChatClient, RetryPolicy, ScriptedChatClient, ScriptedStep, ScriptedStream,
    VvLlmError,
};

fn request() -> ChatRequest {
    ChatRequest::new(
        "unit-model",
        vec![Message::text(MessageRole::User, "hello")],
    )
}

fn response() -> ChatResponse {
    ChatResponse {
        id: "response".to_string(),
        model: "unit-model".to_string(),
        content: "ok".to_string(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        usage: None,
    }
}

#[tokio::test]
async fn scripted_client_drives_retry_contract_and_records_requests() {
    let scripted = Arc::new(ScriptedChatClient::new(
        "unit",
        vec![
            ScriptedStep::error(VvLlmError::classified(
                ErrorKind::ProviderInternal,
                "temporary",
            )),
            ScriptedStep::response(response()),
        ],
    ));

    struct SharedClient(Arc<ScriptedChatClient>);

    #[async_trait::async_trait]
    impl ChatClient for SharedClient {
        fn provider_name(&self) -> &'static str {
            self.0.provider_name()
        }

        async fn create_completion(
            &self,
            request: ChatRequest,
        ) -> Result<ChatResponse, VvLlmError> {
            self.0.create_completion(request).await
        }

        async fn create_stream(
            &self,
            request: ChatRequest,
        ) -> Result<vv_llm::ChatStream, VvLlmError> {
            self.0.create_stream(request).await
        }
    }

    let client = MiddlewareChatClient::new(Box::new(SharedClient(scripted.clone())), Vec::new())
        .unwrap()
        .with_retry_policy(
            RetryPolicy::new(2)
                .with_base_delay(Duration::ZERO)
                .with_max_delay(Duration::ZERO),
        );

    let result = client.create_with_metadata(request()).await.unwrap();
    assert_eq!(result.response.content, "ok");
    assert_eq!(result.metadata.attempts, 2);
    assert_eq!(scripted.requests().len(), 2);
}

#[tokio::test]
async fn scripted_stream_fails_after_visible_chunk_without_replay() {
    let client =
        ScriptedChatClient::new("unit", Vec::new()).with_streams(vec![ScriptedStream::new(vec![
            Ok(ChatStreamDelta {
                content: "first".to_string(),
                ..Default::default()
            }),
            Err(VvLlmError::classified(ErrorKind::Network, "late failure")),
        ])]);

    let mut stream = client.create_stream(request()).await.unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "first");
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().kind(),
        ErrorKind::Network
    );
    assert_eq!(client.requests().len(), 1);
}
