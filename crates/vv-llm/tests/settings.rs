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

#[test]
fn endpoint_bindings_accept_string_and_object_forms() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [
        {"id":"disabled","api_base":"https://disabled.example.com/v1","api_key":"sk-disabled"},
        {"id":"enabled","api_base":"https://enabled.example.com/v1","api_key":"sk-enabled"}
      ],
      "backends": {
        "qwen": {
          "models": {
            "qwq-32b": {
              "id":"qwq-32b",
              "endpoints":[
                {"endpoint_id":"disabled","model_id":"provider-disabled","enabled":false},
                {"endpoint_id":"enabled","model_id":"Qwen/QwQ-32B","enabled":true}
              ]
            },
            "qwen-max": {
              "id":"qwen-max",
              "endpoints":["enabled"]
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let object_binding = settings
        .resolve_chat_model(BackendType::Qwen, "qwq-32b")
        .unwrap();
    let string_binding = settings
        .resolve_chat_model(BackendType::Qwen, "qwen-max")
        .unwrap();

    assert_eq!(object_binding.endpoint.id, "enabled");
    assert_eq!(object_binding.model_id, "Qwen/QwQ-32B");
    assert_eq!(string_binding.endpoint.id, "enabled");
    assert_eq!(string_binding.model_id, "qwen-max");
}

#[test]
fn endpoint_preserves_anthropic_bedrock_transport_fields() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [
        {
          "id":"bedrock-anthropic",
          "api_base":"https://bedrock-runtime.us-east-1.amazonaws.com",
          "endpoint_type":"anthropic_bedrock",
          "is_bedrock":true,
          "region":"us-east-1",
          "credentials":{"access_key":"AKIA_TEST","secret_key":"SECRET_TEST"}
        }
      ],
      "backends": {
        "anthropic": {
          "models": {
            "claude-sonnet": {
              "id":"claude-sonnet",
              "endpoints":[{"endpoint_id":"bedrock-anthropic","model_id":"global.anthropic.claude-sonnet"}]
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet")
        .unwrap();

    assert_eq!(
        resolved.endpoint.endpoint_type.as_deref(),
        Some("anthropic_bedrock")
    );
    assert_eq!(resolved.endpoint.region.as_deref(), Some("us-east-1"));
    assert_eq!(resolved.endpoint.is_bedrock, Some(true));
    assert_eq!(
        resolved.endpoint.credentials["access_key"].as_str(),
        Some("AKIA_TEST")
    );
    assert_eq!(resolved.model_id, "global.anthropic.claude-sonnet");
}

#[test]
fn endpoint_preserves_openai_vertex_transport_fields() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [
        {
          "id":"vertex-openai",
          "api_base":"https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
          "endpoint_type":"openai_vertex",
          "is_vertex":true,
          "region":"global",
          "credentials":{"refresh_token":"refresh","client_id":"client","client_secret":"secret"}
        }
      ],
      "backends": {
        "gemini": {
          "models": {
            "gemini-3-pro": {
              "id":"gemini-3-pro",
              "endpoints":[{"endpoint_id":"vertex-openai","model_id":"gemini-3-pro-preview"}]
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Gemini, "gemini-3-pro")
        .unwrap();

    assert_eq!(
        resolved.endpoint.endpoint_type.as_deref(),
        Some("openai_vertex")
    );
    assert_eq!(resolved.endpoint.region.as_deref(), Some("global"));
    assert_eq!(resolved.endpoint.is_vertex, Some(true));
    assert_eq!(
        resolved.endpoint.credentials["refresh_token"].as_str(),
        Some("refresh")
    );
    assert_eq!(resolved.model_id, "gemini-3-pro-preview");
}
