use std::fs;
use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::{json, Value};
use vv_llm::chat_clients::OpenAiCompatibleChatClient;
use vv_llm::{parse_retry_after, ChatRequest, ToolCall};

fn openai_fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("contract")
        .join("v1.0.0")
        .join("fixtures")
        .join("openai-compatible.v2.json");
    let content = fs::read_to_string(path).expect("protocol fixture");
    serde_json::from_str(&content).expect("valid protocol fixture")
}

fn retry_fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("contract")
        .join("v1.0.0")
        .join("fixtures")
        .join("retry-after.v1.json");
    let content = fs::read_to_string(path).expect("retry fixture");
    serde_json::from_str(&content).expect("valid retry fixture")
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
    let fixture = openai_fixture();
    let canonical = &fixture["request_case"]["canonical_request"];
    let client = OpenAiCompatibleChatClient::new(
        canonical["model"].as_str().unwrap(),
        "https://example.invalid/v1",
        "test-key",
    );

    let request = ChatRequest::from_contract(canonical).unwrap();
    let encoded = request.to_contract().unwrap();
    assert_eq!(encoded, *canonical);
    assert_eq!(
        request.messages[0].text_content().as_deref(),
        Some("look at both images")
    );
    assert_eq!(request.messages[0].content.len(), 3);
    assert_eq!(request.messages[1].tool_calls[0].name, "lookup");
    assert_eq!(
        request.options.max_tokens_details.as_ref().unwrap()["reasoning_tokens"],
        8
    );
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
fn canonical_chat_request_codec_accepts_object_tool_choice() {
    let canonical = json!({
        "model": "tool-model",
        "messages": [{"role": "user", "content": "hello"}],
        "options": {"stop": "DONE", "max_tokens_details": {"reason": "test"}},
        "tools": [],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "extra_body": {"trace_id": "codec"}
    });
    let request = ChatRequest::from_contract(&canonical).unwrap();
    assert_eq!(
        request.to_contract().unwrap()["tool_choice"],
        canonical["tool_choice"]
    );
    assert_eq!(request.options.stop, vec!["DONE"]);

    let client =
        OpenAiCompatibleChatClient::new("tool-model", "https://example.invalid/v1", "test-key");
    let wire = client.to_openai_json(&request).unwrap();
    assert_eq!(wire["tool_choice"], canonical["tool_choice"]);
    assert_eq!(
        wire["max_tokens_details"],
        canonical["options"]["max_tokens_details"]
    );
}

#[test]
fn canonical_codec_preserves_x_extensions_and_nested_image_locations() {
    let canonical = json!({
        "model": "extension-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "url": "https://example.com/flat.png", "detail": "low", "x_flat": {"source": "outer"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/nested.png", "detail": "high", "x_nested": {"source": "inner"}}, "x_outer": true}
            ],
            "x_message": "kept"
        }],
        "options": {"max_tokens_details": {"reasoning_tokens": 8}, "x_options": {"trace": true}},
        "tools": [{"name": "lookup", "parameters": {"type": "object"}, "x_tool": "kept"}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "extra_body": {"trace_id": "extensions"},
        "x_request": {"source": "canonical"}
    });

    let request = ChatRequest::from_contract(&canonical).unwrap();
    assert_eq!(request.to_contract().unwrap(), canonical);

    let client = OpenAiCompatibleChatClient::new(
        "extension-model",
        "https://example.invalid/v1",
        "test-key",
    );
    let wire = client.to_openai_json(&request).unwrap();
    assert_eq!(
        wire["messages"][0]["content"][0]["image_url"]["url"],
        "https://example.com/flat.png"
    );
    assert_eq!(
        wire["messages"][0]["content"][0]["image_url"]["detail"],
        "low"
    );
    assert_eq!(
        wire["messages"][0]["content"][0]["cache_control"],
        Value::Null
    );
    assert_eq!(
        wire["messages"][0]["content"][1]["image_url"]["detail"],
        "high"
    );
    assert_eq!(
        wire["messages"][0]["content"][1]["image_url"]["x_nested"]["source"],
        "inner"
    );
    assert_eq!(wire["messages"][0]["content"][1]["x_outer"], true);
    assert_eq!(wire["messages"][0]["x_message"], "kept");
    assert_eq!(wire["tools"][0]["x_tool"], "kept");
    assert_eq!(wire["x_request"]["source"], "canonical");
    assert_eq!(wire["x_options"]["trace"], true);
    assert_eq!(wire["max_tokens_details"]["reasoning_tokens"], 8);
}

fn assert_contract_rejects(value: Value) {
    let error = ChatRequest::from_contract(&value).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("canonical chat request schema validation failed"),
        "{error}"
    );
}

#[test]
fn canonical_codec_rejects_unknown_non_x_fields() {
    for value in [
        json!({
            "model": "unknown-root",
            "messages": [{"role": "user", "content": "hello"}],
            "unknown": true
        }),
        json!({
            "model": "unknown-message",
            "messages": [{"role": "user", "content": "hello", "unknown": true}]
        }),
        json!({
            "model": "unknown-content",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello", "unknown": true}]
            }]
        }),
        json!({
            "model": "unknown-options",
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"unknown": true}
        }),
    ] {
        assert_contract_rejects(value);
    }
}

#[test]
fn canonical_codec_rejects_invalid_image_detail() {
    assert_contract_rejects(json!({
        "model": "invalid-detail",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "url": "https://example.com/image.png",
                "detail": "ultra"
            }]
        }]
    }));
}

#[test]
fn canonical_codec_rejects_negative_temperature_and_top_p() {
    for options in [json!({"temperature": -0.1}), json!({"top_p": -0.1})] {
        assert_contract_rejects(json!({
            "model": "negative-option",
            "messages": [{"role": "user", "content": "hello"}],
            "options": options
        }));
    }
}

#[test]
fn canonical_codec_rejects_zero_n() {
    assert_contract_rejects(json!({
        "model": "zero-n",
        "messages": [{"role": "user", "content": "hello"}],
        "options": {"n": 0}
    }));
}

#[test]
fn canonical_codec_requires_non_blank_model_while_direct_serde_accepts_runtime_defaults() {
    let empty = json!({"model": "", "messages": []});
    let runtime: ChatRequest = serde_json::from_value(empty.clone()).unwrap();
    assert!(runtime.model.is_empty());
    assert_contract_rejects(empty);
    assert_contract_rejects(json!({"model": "   ", "messages": []}));
    assert!(ChatRequest::new("model", Vec::new()).to_contract().is_ok());
}

#[test]
fn retry_after_fixture_cases() {
    for case in retry_fixture()["cases"].as_array().unwrap() {
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
