use vv_llm::{settings::LlmSettings, BackendType};

#[test]
fn loads_v2_settings_and_resolves_chat_model_endpoint() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"openai-default","api_base":"https://api.openai.com/v1","api_key":"sk-test"}],
      "backends": {"openai": {"models": {"gpt-4o": {"id":"gpt-4o","endpoints":["openai-default"],"context_length":128000}}}},
      "embedding_backends": {},
      "rerank_backends": {}
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::OpenAI, "gpt-4o")
        .unwrap();

    assert_eq!(resolved.model.id, "gpt-4o");
    assert_eq!(resolved.endpoint.id, "openai-default");
    assert_eq!(
        resolved.endpoint.api_base.as_deref(),
        Some("https://api.openai.com/v1")
    );
}

#[test]
fn resolves_models_by_public_key_or_provider_id() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"openai-default","api_base":"https://api.openai.com/v1","api_key":"sk-test"}],
      "backends": {
        "openai": {
          "models": {
            "display-name": {"id":"provider-model-id","endpoints":["openai-default"]}
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    assert_eq!(
        settings
            .resolve_chat_model(BackendType::OpenAI, "display-name")
            .unwrap()
            .model
            .id,
        "provider-model-id"
    );
    assert_eq!(
        settings
            .resolve_chat_model(BackendType::OpenAI, "provider-model-id")
            .unwrap()
            .model
            .id,
        "provider-model-id"
    );
}

#[test]
fn resolves_embedding_and_rerank_models() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"retrieval","api_base":"https://example.com/v1","api_key":"sk-test"}],
      "embedding_backends": {
        "siliconflow": {
          "models": {
            "Qwen/Qwen3-Embedding-4B": {
              "id":"Qwen/Qwen3-Embedding-4B",
              "endpoints":["retrieval"],
              "protocol":"siliconflow"
            }
          }
        }
      },
      "rerank_backends": {
        "siliconflow": {
          "models": {
            "BAAI/bge-reranker-v2-m3": {
              "id":"BAAI/bge-reranker-v2-m3",
              "endpoints":["retrieval"],
              "protocol":"siliconflow"
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();

    let embedding = settings
        .resolve_embedding_model("siliconflow", "Qwen/Qwen3-Embedding-4B")
        .unwrap();
    let rerank = settings
        .resolve_rerank_model("siliconflow", "BAAI/bge-reranker-v2-m3")
        .unwrap();

    assert_eq!(embedding.model.protocol.as_deref(), Some("siliconflow"));
    assert_eq!(rerank.model.protocol.as_deref(), Some("siliconflow"));
}

#[test]
fn missing_backend_and_endpoint_return_specific_errors() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [],
      "backends": {
        "openai": {
          "models": {
            "gpt-4o": {"id":"gpt-4o","endpoints":["missing-endpoint"]}
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();

    let missing_model = settings
        .resolve_chat_model(BackendType::OpenAI, "not-present")
        .unwrap_err()
        .to_string();
    let missing_endpoint = settings
        .resolve_chat_model(BackendType::OpenAI, "gpt-4o")
        .unwrap_err()
        .to_string();

    assert!(missing_model.contains("model not found"));
    assert!(missing_endpoint.contains("endpoint not found: missing-endpoint"));
}
