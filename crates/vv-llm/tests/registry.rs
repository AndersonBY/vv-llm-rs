use futures_util::StreamExt;
use std::sync::Arc;
use vv_llm::{
    ChatClient, ChatRequest, ChatResponse, ErrorKind, FallbackChatClient, FallbackRoute, Message,
    MessageRole, ModelCapabilities, ProviderRegistry, ScriptedChatClient, ScriptedStep,
    ScriptedStream, ThinkingCapability, VvLlmError,
};

fn request() -> ChatRequest {
    ChatRequest::new(
        "logical-model",
        vec![Message::text(MessageRole::User, "hello")],
    )
}

fn response(model: &str, content: &str) -> ChatResponse {
    ChatResponse {
        id: format!("{model}-response"),
        model: model.to_string(),
        content: content.to_string(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        usage: None,
    }
}

struct SharedClient(Arc<ScriptedChatClient>);

#[async_trait::async_trait]
impl ChatClient for SharedClient {
    fn provider_name(&self) -> &'static str {
        self.0.provider_name()
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        self.0.create_completion(request).await
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<vv_llm::ChatStream, VvLlmError> {
        self.0.create_stream(request).await
    }
}

fn register_shared(
    registry: &mut ProviderRegistry,
    name: &str,
    client: Arc<ScriptedChatClient>,
    capabilities: ModelCapabilities,
) {
    registry
        .register(
            name,
            move || Box::new(SharedClient(client.clone())),
            capabilities,
        )
        .unwrap();
}

#[tokio::test]
async fn fallback_skips_incompatible_candidates_without_calling_them() {
    let incapable = Arc::new(ScriptedChatClient::new(
        "incapable",
        vec![ScriptedStep::response(response("model-a", "must not run"))],
    ));
    let capable = Arc::new(ScriptedChatClient::new(
        "capable",
        vec![ScriptedStep::response(response("model-b", "ok"))],
    ));
    let mut registry = ProviderRegistry::new();
    register_shared(
        &mut registry,
        "incapable",
        incapable.clone(),
        ModelCapabilities::default(),
    );
    register_shared(
        &mut registry,
        "capable",
        capable.clone(),
        ModelCapabilities {
            tools: true,
            ..Default::default()
        },
    );
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("incapable", "model-a"),
            FallbackRoute::new("capable", "model-b"),
        ],
    )
    .unwrap();
    let mut routed = request();
    routed.tools.push(vv_llm::ChatTool::function(
        "lookup",
        "lookup",
        serde_json::json!({"type": "object"}),
    ));

    let result = client.create_with_metadata(routed).await.unwrap();
    assert_eq!(result.response.content, "ok");
    assert_eq!(result.metadata.fallback_index, 1);
    assert!(incapable.requests().is_empty());
    assert_eq!(capable.requests()[0].model, "model-b");
}

#[tokio::test]
async fn fallback_does_not_hide_authentication_errors() {
    let primary = Arc::new(ScriptedChatClient::new(
        "primary",
        vec![ScriptedStep::error(VvLlmError::classified(
            ErrorKind::Authentication,
            "unauthorized",
        ))],
    ));
    let secondary = Arc::new(ScriptedChatClient::new(
        "secondary",
        vec![ScriptedStep::response(response("model-b", "must not run"))],
    ));
    let mut registry = ProviderRegistry::new();
    register_shared(
        &mut registry,
        "primary",
        primary,
        ModelCapabilities::default(),
    );
    register_shared(
        &mut registry,
        "secondary",
        secondary.clone(),
        ModelCapabilities::default(),
    );
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("primary", "model-a"),
            FallbackRoute::new("secondary", "model-b"),
        ],
    )
    .unwrap();

    let error = client.create_completion(request()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Authentication);
    assert!(secondary.requests().is_empty());
}

#[tokio::test]
async fn stream_does_not_fallback_after_first_visible_chunk() {
    let primary = Arc::new(
        ScriptedChatClient::new("primary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![
                Ok(vv_llm::ChatStreamDelta {
                    content: "first".to_string(),
                    ..Default::default()
                }),
                Err(VvLlmError::classified(ErrorKind::Network, "late failure")),
            ],
        )]),
    );
    let secondary = Arc::new(
        ScriptedChatClient::new("secondary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![Ok(vv_llm::ChatStreamDelta {
                content: "secondary".to_string(),
                ..Default::default()
            })],
        )]),
    );
    let capabilities = ModelCapabilities {
        thinking: ThinkingCapability::Configurable,
        ..Default::default()
    };
    let mut registry = ProviderRegistry::new();
    register_shared(&mut registry, "primary", primary, capabilities.clone());
    register_shared(&mut registry, "secondary", secondary.clone(), capabilities);
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("primary", "model-a"),
            FallbackRoute::new("secondary", "model-b"),
        ],
    )
    .unwrap();

    let mut stream = client.create_stream(request()).await.unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "first");
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().kind(),
        ErrorKind::Network
    );
    assert!(secondary.requests().is_empty());
}

#[tokio::test]
async fn stream_can_fallback_when_first_chunk_is_an_error() {
    let primary = Arc::new(
        ScriptedChatClient::new("primary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![Err(VvLlmError::classified(
                ErrorKind::Network,
                "early failure",
            ))],
        )]),
    );
    let secondary = Arc::new(
        ScriptedChatClient::new("secondary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![Ok(vv_llm::ChatStreamDelta {
                content: "secondary".to_string(),
                ..Default::default()
            })],
        )]),
    );
    let mut registry = ProviderRegistry::new();
    register_shared(
        &mut registry,
        "primary",
        primary.clone(),
        ModelCapabilities::default(),
    );
    register_shared(
        &mut registry,
        "secondary",
        secondary.clone(),
        ModelCapabilities::default(),
    );
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("primary", "model-a"),
            FallbackRoute::new("secondary", "model-b"),
        ],
    )
    .unwrap();

    let mut stream = client.create_stream(request()).await.unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "secondary");
    assert_eq!(primary.requests().len(), 1);
    assert_eq!(secondary.requests().len(), 1);
}

#[tokio::test]
async fn stream_can_fallback_after_non_visible_prelude_before_error() {
    let primary = Arc::new(
        ScriptedChatClient::new("primary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![
                Ok(vv_llm::ChatStreamDelta {
                    usage: Some(vv_llm::ChatUsage {
                        prompt_tokens: Some(1),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                Err(VvLlmError::classified(ErrorKind::Network, "early failure")),
            ],
        )]),
    );
    let secondary = Arc::new(
        ScriptedChatClient::new("secondary", Vec::new()).with_streams(vec![ScriptedStream::new(
            vec![Ok(vv_llm::ChatStreamDelta {
                content: "secondary".to_string(),
                ..Default::default()
            })],
        )]),
    );
    let mut registry = ProviderRegistry::new();
    register_shared(
        &mut registry,
        "primary",
        primary.clone(),
        ModelCapabilities::default(),
    );
    register_shared(
        &mut registry,
        "secondary",
        secondary.clone(),
        ModelCapabilities::default(),
    );
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("primary", "model-a"),
            FallbackRoute::new("secondary", "model-b"),
        ],
    )
    .unwrap();

    let mut stream = client.create_stream(request()).await.unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().content, "secondary");
    assert_eq!(primary.requests().len(), 1);
    assert_eq!(secondary.requests().len(), 1);
}

#[tokio::test]
async fn fallback_metadata_rejects_stream_requests_explicitly() {
    let capable = Arc::new(ScriptedChatClient::new(
        "capable",
        vec![ScriptedStep::response(response("model-a", "ok"))],
    ));
    let mut registry = ProviderRegistry::new();
    register_shared(
        &mut registry,
        "capable",
        capable,
        ModelCapabilities::default(),
    );
    let client = FallbackChatClient::new(
        Arc::new(registry),
        vec![FallbackRoute::new("capable", "model-a")],
    )
    .unwrap();
    let mut request = request();
    request.options.stream = Some(true);
    let error = client.create_with_metadata(request).await.unwrap_err();
    assert!(matches!(error, VvLlmError::Configuration(_)));
}
