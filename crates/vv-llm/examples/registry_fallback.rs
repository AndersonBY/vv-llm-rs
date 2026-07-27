use std::sync::Arc;
use vv_llm::{
    ChatRequest, ChatResponse, ErrorKind, FallbackChatClient, FallbackRoute, Message, MessageRole,
    ModelCapabilities, ProviderRegistry, ScriptedChatClient, ScriptedStep, VvLlmError,
};

fn response() -> ChatResponse {
    ChatResponse {
        id: "fallback-response".to_string(),
        model: "secondary-model".to_string(),
        content: "fallback response".to_string(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        usage: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let mut registry = ProviderRegistry::new();
    registry.register(
        "primary",
        || {
            Box::new(ScriptedChatClient::new(
                "primary",
                vec![ScriptedStep::error(VvLlmError::classified(
                    ErrorKind::ProviderInternal,
                    "temporary failure",
                ))],
            ))
        },
        ModelCapabilities::default(),
    )?;
    registry.register(
        "secondary",
        || {
            Box::new(ScriptedChatClient::new(
                "secondary",
                vec![ScriptedStep::response(response())],
            ))
        },
        ModelCapabilities::default(),
    )?;

    let runtime = FallbackChatClient::new(
        Arc::new(registry),
        vec![
            FallbackRoute::new("primary", "primary-model"),
            FallbackRoute::new("secondary", "secondary-model"),
        ],
    )?;
    let result = runtime
        .create_with_metadata(ChatRequest::new(
            "logical-model",
            vec![Message::text(MessageRole::User, "hello")],
        ))
        .await?;

    println!("{}", result.response.content);
    println!("provider: {:?}", result.metadata.provider);
    println!("fallback index: {}", result.metadata.fallback_index);
    Ok(())
}
