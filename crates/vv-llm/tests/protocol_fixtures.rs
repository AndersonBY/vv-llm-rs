use std::fs;
use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::{json, Value};
use vv_llm::chat_clients::OpenAiCompatibleChatClient;
use vv_llm::{
    parse_retry_after, ChatRequest, ChatRequestOptions, ChatTool, Message, MessageRole, ToolCall,
};

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("protocol")
        .join("openai_compatible_v1.json");
    let content = fs::read_to_string(path).expect("protocol fixture");
    serde_json::from_str(&content).expect("valid protocol fixture")
}

fn request_from_fixture(value: &Value) -> ChatRequest {
    let role = match value["messages"][0]["role"].as_str().unwrap() {
        "user" => MessageRole::User,
        role => panic!("unsupported fixture role: {role}"),
    };
    let tools = value["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| ChatTool {
            name: tool["name"].as_str().unwrap().to_string(),
            description: tool["description"].as_str().map(ToOwned::to_owned),
            parameters: tool["parameters"].clone(),
            cache_control: None,
        })
        .collect();

    ChatRequest {
        model: value["model"].as_str().unwrap().to_string(),
        messages: vec![Message::text(
            role,
            value["messages"][0]["content"].as_str().unwrap(),
        )],
        options: serde_json::from_value::<ChatRequestOptions>(value["options"].clone()).unwrap(),
        tools,
        tool_choice: value["tool_choice"].as_str().map(ToOwned::to_owned),
        extra_body: value["extra_body"].clone(),
    }
}

fn usage(value: Option<vv_llm::ChatUsage>) -> Option<Value> {
    value.map(|usage| {
        json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        })
    })
}

fn tool_calls(value: Vec<ToolCall>) -> Value {
    Value::Array(
        value
            .into_iter()
            .map(|tool_call| {
                let mut item = json!({
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                });
                if let Some(index) = tool_call.index {
                    item["index"] = json!(index);
                }
                item
            })
            .collect(),
    )
}

#[test]
fn openai_compatible_fixture_covers_request_completion_and_stream() {
    let fixture = fixture();
    let canonical = &fixture["request_case"]["canonical_request"];
    let client = OpenAiCompatibleChatClient::new(
        canonical["model"].as_str().unwrap(),
        "https://example.invalid/v1",
        "test-key",
    );

    let request = request_from_fixture(canonical);
    let wire = client.to_openai_json(&request).unwrap();
    let expected = &fixture["request_case"]["expected_wire_request"];
    for (key, expected_value) in expected.as_object().unwrap() {
        assert_eq!(&wire[key], expected_value, "wire field {key}");
    }

    let response = OpenAiCompatibleChatClient::normalize_completion_json(
        fixture["completion_case"]["raw_response"].clone(),
    )
    .unwrap();
    let actual_response = json!({
        "content": response.content,
        "reasoning_content": response.reasoning_content,
        "tool_calls": tool_calls(response.tool_calls),
        "usage": usage(response.usage),
    });
    assert_eq!(
        actual_response,
        fixture["completion_case"]["expected_response"]
    );

    let expected_deltas = fixture["stream_case"]["expected_deltas"]
        .as_array()
        .unwrap();
    for (index, raw_chunk) in fixture["stream_case"]["raw_chunks"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let delta =
            OpenAiCompatibleChatClient::normalize_stream_chunk_json(raw_chunk.clone()).unwrap();
        let mut actual = json!({
            "content": delta.content,
            "reasoning_content": delta.reasoning_content,
            "tool_calls": tool_calls(delta.tool_calls),
        });
        if delta.usage.is_some() {
            actual["usage"] = usage(delta.usage).unwrap();
        }
        assert_eq!(actual, expected_deltas[index], "stream delta {index}");
    }
}

#[test]
fn retry_after_fixture_cases() {
    for case in fixture()["retry_after_cases"].as_array().unwrap() {
        let headers = &case["headers"];
        let parsed = parse_retry_after(
            headers.get("retry-after-ms").and_then(Value::as_str),
            headers.get("retry-after").and_then(Value::as_str),
            UNIX_EPOCH + Duration::from_secs(case["now_unix_seconds"].as_u64().unwrap()),
        )
        .map(|duration| duration.as_secs_f64());
        assert_eq!(
            parsed,
            case["expected_seconds"].as_f64(),
            "{}",
            case["name"]
        );
    }
}
