mod common;

use futures_util::StreamExt;
use std::time::Duration;
use vv_llm::{
    ChatClient, ChatRequest, Message, MessageRole, MiddlewareChatClient, RetryPolicy,
    ThinkingPreference, VvLlmError,
};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_deepseek_client()?;
    let runtime = MiddlewareChatClient::new(client, Vec::new())?.with_retry_policy(
        RetryPolicy::new(3)
            .with_base_delay(Duration::from_millis(250))
            .with_total_timeout(Duration::from_secs(30)),
    );
    let mut request = ChatRequest::new(
        model,
        vec![Message::text(
            MessageRole::User,
            "Explain retry backoff briefly.",
        )],
    )
    .with_thinking(ThinkingPreference::Disabled);
    request.options.stream = Some(true);

    let mut stream = runtime.create_stream(request).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if !chunk.reasoning_content.is_empty() {
            print!("{}", chunk.reasoning_content);
        }
        if !chunk.content.is_empty() {
            print!("{}", chunk.content);
        }
    }
    println!();
    Ok(())
}
