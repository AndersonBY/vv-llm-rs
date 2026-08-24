mod common;

use vv_llm::{ChatRequest, ChatTool, Message, MessageRole, ToolChoice, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let (client, model) = common::load_chat_client()?;
    let mut request = ChatRequest::new(
        model,
        vec![Message::text(
            MessageRole::User,
            "Look up the weather for New York.",
        )],
    );
    request.tools = vec![ChatTool::function(
        "get_current_weather",
        "Get the current weather in a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    )];
    request.tool_choice = Some(
        ToolChoice::object(serde_json::json!({
            "type": "function",
            "function": {"name": "get_current_weather"}
        }))
        .map_err(VvLlmError::Configuration)?,
    );

    let response = client.create(request).await?;
    for call in response.tool_calls {
        println!("{} {}", call.name, call.arguments);
    }
    Ok(())
}
