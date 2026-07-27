mod common;

use std::time::Duration;
use vv_llm::{ChatRequest, Message, MessageRole, MiddlewareChatClient, RetryPolicy, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_deepseek_client()?;
    let runtime = MiddlewareChatClient::new(client, Vec::new())?.with_retry_policy(
        RetryPolicy::new(3)
            .with_base_delay(Duration::from_millis(250))
            .with_total_timeout(Duration::from_secs(30)),
    );
    let result = runtime
        .create_with_metadata(ChatRequest::new(
            model,
            vec![Message::text(MessageRole::User, "Reply with exactly OK.")],
        ))
        .await?;

    println!("{}", result.response.content);
    println!("provider: {:?}", result.metadata.provider);
    println!("attempts: {}", result.metadata.attempts);
    println!("latency_ms: {:?}", result.metadata.latency_ms);
    Ok(())
}
