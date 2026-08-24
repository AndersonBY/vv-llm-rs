mod common;

use vv_llm::{ChatRequest, Message, MessageRole, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_chat_client()?;
    let response = client
        .create(ChatRequest::new(
            model,
            vec![Message::text(
                MessageRole::User,
                "Explain vector search in one sentence.",
            )],
        ))
        .await?;

    println!("{}", response.content);
    Ok(())
}
