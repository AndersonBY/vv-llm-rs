mod common;

use vv_llm::{ChatRequest, Message, MessageRole, ThinkingPreference, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_deepseek_client()?;
    let response = client
        .create(
            ChatRequest::new(
                model,
                vec![Message::text(
                    MessageRole::User,
                    "Compute 37 * 19 and answer briefly.",
                )],
            )
            .with_thinking(ThinkingPreference::enabled()),
        )
        .await?;

    println!("reasoning: {:?}", response.reasoning_content);
    println!("answer: {}", response.content);
    Ok(())
}
