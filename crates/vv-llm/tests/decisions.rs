use serde_json::{json, Value};
use vv_llm::{
    create_decision_client_from_resolved, DecisionAnswer, DecisionClient, DecisionRequest,
    DecisionResponse, LlmSettings, OpenAiDecisionClient,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../contract/v1.3.0/fixtures/decisions.v1.json"
    ))
    .unwrap()
}

#[test]
fn shared_decision_contract_and_invalid_payloads() {
    let fixture = fixture();
    for valid in fixture["valid_requests"].as_array().unwrap() {
        assert_eq!(
            DecisionRequest::from_contract(valid)
                .unwrap()
                .to_contract()
                .unwrap(),
            *valid
        );
    }
    let request = DecisionRequest::from_contract(&fixture["request"]).unwrap();
    for valid in fixture["valid_responses"].as_array().unwrap() {
        assert_eq!(
            DecisionResponse::from_contract(valid)
                .unwrap()
                .to_contract()
                .unwrap(),
            *valid
        );
    }
    let response = DecisionResponse::from_contract(&fixture["response"]).unwrap();
    response.validate_for(&request).unwrap();
    assert_eq!(request.to_contract().unwrap(), fixture["request"]);
    assert_eq!(response.to_contract().unwrap(), fixture["response"]);
    assert!(matches!(
        DecisionResponse::from_contract(&fixture["refusal_response"])
            .unwrap()
            .answers[0],
        DecisionAnswer::Refusal { .. }
    ));
    for raw in fixture["invalid_requests"].as_array().unwrap() {
        assert!(DecisionRequest::from_contract(raw).is_err());
    }
    for raw in fixture["invalid_responses"].as_array().unwrap() {
        assert!(DecisionResponse::from_contract(raw).is_err());
    }
    let missing = DecisionResponse::from_contract(&json!({"model": "gpt-6-luna", "answers": [{"type": "predicate", "name": null, "probability": 0}], "usage": {"output_tokens": 0}})).unwrap().to_contract().unwrap();
    assert_eq!(missing["usage"]["output_tokens"], 0);
    assert!(missing["usage"].get("input_tokens").is_none());
}

#[test]
fn decision_settings_are_independent_and_apply_binding_capabilities() {
    let raw = json!({"endpoints": [{"id": "disabled", "enabled": false}, {"id": "active", "api_base": "https://example.invalid/v1", "api_key": "test-key", "proxy": ""}], "decision_backends": {"openai": {"models": {"gpt-6-luna": {"id": "gpt-6-luna", "endpoints": [{"endpoint_id": "disabled"}, {"endpoint_id": "active", "priority": 2, "model_id": "luna-deployment"}]}}}}});
    let settings = LlmSettings::from_json_str(&raw.to_string()).unwrap();
    let resolved = settings
        .resolve_decision_model("openai", "gpt-6-luna")
        .unwrap();
    assert_eq!(resolved.endpoint.id, "active");
    assert_eq!(resolved.model_id, "luna-deployment");
    assert!(settings.backends["openai"].models["gpt-6-luna"]
        .endpoints
        .is_empty());
    create_decision_client_from_resolved(&resolved).unwrap();
    let mut request = DecisionRequest::from_contract(&fixture()["request"]).unwrap();
    request.model.clear();
    let client =
        OpenAiDecisionClient::new(&resolved.model_id, "https://example.invalid", "test-key")
            .with_capabilities(resolved.model.capabilities());
    assert_eq!(
        client.to_openai_json(&request).unwrap()["model"],
        "luna-deployment"
    );
    let mut capabilities = resolved.model.capabilities();
    capabilities.decision_types = Some(vec![]);
    let disabled = client.with_capabilities(capabilities);
    assert!(disabled.to_openai_json(&request).is_err());
}

async fn serve(status: u16, body: Value) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut received = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            let size = socket.read(&mut buffer).await.unwrap();
            assert!(size > 0);
            received.extend_from_slice(&buffer[..size]);
            if let Some(end) = received.windows(4).position(|window| window == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&received[..end]);
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(str::trim)
                            .and_then(|value| value.parse().ok())
                    })
                    .unwrap();
                if received.len() >= end + 4 + length {
                    break;
                }
            }
        }
        assert!(received.starts_with(b"POST /decisions HTTP/1.1"));
        let end = received
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4;
        let request: Value = serde_json::from_slice(&received[end..]).unwrap();
        assert_eq!(
            request["questions"][1]["choices"],
            json!(["replacement", "refund"])
        );
        assert_eq!(
            request["questions"][2]["rubric"],
            json!([{"label": "low"}, {"label": "medium"}, {"label": "high"}])
        );
        let body = body.to_string();
        let response = format!("HTTP/1.1 {status} Result\r\ncontent-type: application/json\r\nretry-after: 2\r\nx-request-id: fixture-id\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    (base, server)
}

#[tokio::test]
async fn decision_transport_and_rate_limit_metadata() {
    let fixture = fixture();
    let (base, server) = serve(200, fixture["response"].clone()).await;
    let client = OpenAiDecisionClient::new("gpt-6-luna", base, "test-key");
    let response = client
        .create(DecisionRequest::from_contract(&fixture["request"]).unwrap())
        .await
        .unwrap();
    assert_eq!(response.to_contract().unwrap(), fixture["response"]);
    server.await.unwrap();
    let (base, server) = serve(429, json!({"error": {"message": "slow down"}})).await;
    let client = OpenAiDecisionClient::new("gpt-6-luna", base, "test-key");
    let error = client
        .create(DecisionRequest::from_contract(&fixture["request"]).unwrap())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), vv_llm::ErrorKind::RateLimited);
    assert_eq!(error.retry_after_seconds(), Some(2.0));
    server.await.unwrap();
}

#[test]
fn decision_model_aliases_inherit_catalog_capabilities() {
    let raw = json!({"endpoints": [{"id": "active", "api_base": "https://example.invalid/v1"}], "decision_backends": {"openai": {"models": {"friendly": {"id": "gpt-6-luna", "endpoints": ["active"]}}}}});
    let settings = LlmSettings::from_json_str(&raw.to_string()).unwrap();
    let resolved = settings
        .resolve_decision_model("openai", "friendly")
        .unwrap();
    assert_eq!(
        resolved
            .model
            .capabilities()
            .decision_types
            .as_ref()
            .unwrap()
            .len(),
        3
    );
}
