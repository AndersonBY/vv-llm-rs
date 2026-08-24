mod common;

use vv_llm::{ChatRequest, Message, MessageContent, MessageRole, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_chat_client()?;
    let image_url = std::env::var("VV_LLM_IMAGE_URL")
        .unwrap_or_else(|_| "https://example.com/image.png".to_string());
    let response = client
        .create(ChatRequest::new(
            model,
            vec![Message {
                role: MessageRole::User,
                content: vec![
                    MessageContent::Text {
                        text: "What is in this image?".to_string(),
                        cache_control: None,
                        extensions: Default::default(),
                    },
                    MessageContent::ImageUrl {
                        url: image_url.clone(),
                        detail: Some("low".to_string()),
                        cache_control: Some(serde_json::json!({"type": "ephemeral"})),
                        extensions: Default::default(),
                        nested_extensions: Default::default(),
                        nested_image: false,
                    },
                    MessageContent::ImageUrl {
                        url: image_url,
                        detail: Some("high".to_string()),
                        cache_control: Some(serde_json::json!({"type": "ephemeral"})),
                        extensions: Default::default(),
                        nested_extensions: Default::default(),
                        nested_image: true,
                    },
                ],
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
                extensions: Default::default(),
            }],
        ))
        .await?;

    println!("{}", response.content);
    Ok(())
}
