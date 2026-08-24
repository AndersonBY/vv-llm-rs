use vv_llm::{ChatRequest, VvLlmError};

#[tokio::main]
async fn main() -> Result<(), VvLlmError> {
    let canonical = serde_json::json!({
        "model": "example-model",
        "messages": [{
            "role": "user",
            "content": "Use the weather tool."
        }],
        "options": {
            "max_tokens": 128,
            "stop": ["done"]
        },
        "tools": [{
            "name": "get_current_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_current_weather"}
        },
        "x_example": {"source": "rust-example"}
    });
    let request = ChatRequest::from_contract(&canonical)?;
    let encoded = request.to_contract()?;
    println!("{}", serde_json::to_string_pretty(&encoded)?);
    Ok(())
}
