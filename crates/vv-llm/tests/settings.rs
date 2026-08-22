use vv_llm::{default_chat_model, settings::LlmSettings, BackendType};

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
fn settings_do_not_upgrade_v1_top_level_backends_or_mark_version() {
    let raw = r#"{
      "VERSION": "1",
      "endpoints": [{"id":"openai-default","api_base":"https://api.openai.com/v1","api_key":"sk-test"}],
      "openai": {
        "default_endpoint": "openai-default",
        "models": {
          "gpt-4o": {"id":"gpt-4o","context_length":128000}
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();

    assert_eq!(settings.version.as_deref(), Some("1"));
    assert!(settings.extra.contains_key("openai"));
    assert!(settings
        .backends
        .get("openai")
        .and_then(|backend| backend.default_endpoint.as_deref())
        .is_none());
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
fn preserves_python_retrieval_model_metadata() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"retrieval","api_base":"https://example.com/v1","api_key":"sk-test"}],
      "embedding_backends": {
        "custom": {
          "models": {
            "embedding-model": {
              "id":"embedding-model",
              "endpoints":["retrieval"],
              "protocol":"custom_json_http",
              "dimensions": 1024
            }
          }
        }
      },
      "rerank_backends": {
        "custom": {
          "models": {
            "rerank-model": {
              "id":"rerank-model",
              "endpoints":["retrieval"],
              "protocol":"custom_json_http",
              "default_top_n": 12
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let embedding = settings
        .resolve_embedding_model("custom", "embedding-model")
        .unwrap();
    let rerank = settings
        .resolve_rerank_model("custom", "rerank-model")
        .unwrap();

    assert_eq!(embedding.model.dimensions, Some(1024));
    assert_eq!(rerank.model.default_top_n, Some(12));
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
    assert!(resolved.endpoint.is_bedrock);
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
    assert!(resolved.endpoint.is_vertex);
    assert_eq!(
        resolved.endpoint.credentials["refresh_token"].as_str(),
        Some("refresh")
    );
    assert_eq!(resolved.model_id, "gemini-3-pro-preview");
}

#[test]
fn loads_python_default_chat_catalog_for_empty_settings() {
    let settings = LlmSettings::from_json_str("{}").unwrap();
    assert_eq!(settings.version.as_deref(), None);
    let qwen = settings
        .backends
        .get("qwen")
        .and_then(|backend| backend.models.get("qwen3.7-max"))
        .expect("qwen3.7-max should come from the Python default catalog");
    let moonshot = settings
        .backends
        .get("moonshot")
        .and_then(|backend| backend.models.get("kimi-k2.7-code"))
        .expect("kimi-k2.7-code should come from the Python default catalog");
    let moonshot_kimi_k3 = settings
        .backends
        .get("moonshot")
        .and_then(|backend| backend.models.get("kimi-k3"))
        .expect("kimi-k3 should come from the Python default catalog");
    let zhipuai = settings
        .backends
        .get("zhipuai")
        .and_then(|backend| backend.models.get("glm-5.2"))
        .expect("glm-5.2 should come from the Python default catalog");
    let anthropic = settings
        .backends
        .get("anthropic")
        .and_then(|backend| backend.models.get("claude-opus-4-8"))
        .expect("claude-opus-4-8 should come from the Python default catalog");
    let anthropic_fable = settings
        .backends
        .get("anthropic")
        .and_then(|backend| backend.models.get("claude-fable-5"))
        .expect("claude-fable-5 should come from the Python default catalog");
    let anthropic_opus_46 = settings
        .backends
        .get("anthropic")
        .and_then(|backend| backend.models.get("claude-opus-4-6"))
        .expect("claude-opus-4-6 should come from the Python default catalog");
    let anthropic_sonnet_5 = settings
        .backends
        .get("anthropic")
        .and_then(|backend| backend.models.get("claude-sonnet-5"))
        .expect("claude-sonnet-5 should come from the Python default catalog");
    let anthropic_opus_5 = settings
        .backends
        .get("anthropic")
        .and_then(|backend| backend.models.get("claude-opus-5"))
        .expect("claude-opus-5 should come from the Python default catalog");
    let minimax = settings
        .backends
        .get("minimax")
        .and_then(|backend| backend.models.get("MiniMax-M3"))
        .expect("MiniMax-M3 should come from the Python default catalog");
    let ernie_51 = settings
        .backends
        .get("ernie")
        .and_then(|backend| backend.models.get("ernie-5.1"))
        .expect("ernie-5.1 should come from the Python default catalog");
    let openai_gpt_56_sol = settings
        .backends
        .get("openai")
        .and_then(|backend| backend.models.get("gpt-5.6-sol"))
        .expect("gpt-5.6-sol should come from the Python default catalog");
    let openai_gpt_56_terra = settings
        .backends
        .get("openai")
        .and_then(|backend| backend.models.get("gpt-5.6-terra"))
        .expect("gpt-5.6-terra should come from the Python default catalog");
    let openai_gpt_56_luna = settings
        .backends
        .get("openai")
        .and_then(|backend| backend.models.get("gpt-5.6-luna"))
        .expect("gpt-5.6-luna should come from the Python default catalog");
    let gemini_37_flash = settings
        .backends
        .get("gemini")
        .and_then(|backend| backend.models.get("gemini-3.7-flash"))
        .expect("gemini-3.7-flash should come from the Python default catalog");
    let grok_46 = settings
        .backends
        .get("xai")
        .and_then(|backend| backend.models.get("grok-4.6"))
        .expect("grok-4.6 should come from the Python default catalog");

    assert_eq!(default_chat_model(BackendType::Moonshot), Some("kimi-k2.6"));
    assert_eq!(
        default_chat_model(BackendType::DeepSeek),
        Some("deepseek-v4-pro")
    );
    assert_eq!(
        default_chat_model(BackendType::Qwen),
        Some("qwen3.5-397b-a17b")
    );
    assert_eq!(default_chat_model(BackendType::ZhiPuAI), Some("glm-5.2"));
    assert_eq!(default_chat_model(BackendType::OpenAI), Some("gpt-5.5"));
    assert_eq!(
        default_chat_model(BackendType::Anthropic),
        Some("claude-opus-4-8")
    );
    assert_eq!(default_chat_model(BackendType::MiniMax), Some("MiniMax-M3"));
    assert_eq!(
        default_chat_model(BackendType::Gemini),
        Some("gemini-3.5-flash")
    );
    assert_eq!(moonshot.context_length, Some(256_000));
    assert_eq!(moonshot.function_call_available, Some(true));
    assert_eq!(moonshot.response_format_available, Some(true));
    assert_eq!(moonshot.native_multimodal, Some(true));
    assert_eq!(moonshot_kimi_k3.context_length, Some(1_048_576));
    assert_eq!(moonshot_kimi_k3.max_output_tokens, Some(1_048_576));
    assert_eq!(moonshot_kimi_k3.function_call_available, Some(true));
    assert_eq!(moonshot_kimi_k3.response_format_available, Some(true));
    assert_eq!(moonshot_kimi_k3.native_multimodal, Some(true));
    assert_eq!(qwen.context_length, Some(1_000_000));
    assert_eq!(qwen.max_output_tokens, Some(65_536));
    assert_eq!(qwen.function_call_available, Some(true));
    assert_eq!(qwen.response_format_available, Some(true));
    assert_eq!(qwen.native_multimodal, Some(false));
    assert_eq!(zhipuai.context_length, Some(1_000_000));
    assert_eq!(zhipuai.max_output_tokens, Some(128_000));
    assert_eq!(zhipuai.function_call_available, Some(true));
    assert_eq!(zhipuai.response_format_available, Some(true));
    assert_eq!(zhipuai.native_multimodal, Some(false));
    assert_eq!(anthropic_opus_46.context_length, Some(1_000_000));
    assert_eq!(anthropic.context_length, Some(1_000_000));
    assert_eq!(anthropic.max_output_tokens, Some(128_000));
    assert_eq!(anthropic.native_multimodal, Some(true));
    assert_eq!(anthropic_fable.context_length, Some(1_000_000));
    assert_eq!(anthropic_fable.max_output_tokens, Some(128_000));
    assert_eq!(anthropic_fable.native_multimodal, Some(true));
    assert_eq!(anthropic_sonnet_5.context_length, Some(1_000_000));
    assert_eq!(anthropic_sonnet_5.max_output_tokens, Some(128_000));
    assert_eq!(anthropic_sonnet_5.native_multimodal, Some(true));
    assert_eq!(anthropic_opus_5.context_length, Some(1_000_000));
    assert_eq!(anthropic_opus_5.max_output_tokens, Some(128_000));
    assert_eq!(anthropic_opus_5.function_call_available, Some(true));
    assert_eq!(anthropic_opus_5.response_format_available, Some(false));
    assert_eq!(anthropic_opus_5.native_multimodal, Some(true));
    assert_eq!(minimax.context_length, Some(1_000_000));
    assert_eq!(minimax.max_output_tokens, Some(10_240));
    assert_eq!(minimax.function_call_available, Some(true));
    assert_eq!(minimax.response_format_available, Some(true));
    assert_eq!(minimax.native_multimodal, Some(true));
    assert_eq!(ernie_51.context_length, Some(128_000));
    assert_eq!(ernie_51.max_output_tokens, Some(65_536));
    assert_eq!(ernie_51.response_format_available, Some(true));
    assert_eq!(ernie_51.native_multimodal, Some(false));
    for model in [openai_gpt_56_sol, openai_gpt_56_terra, openai_gpt_56_luna] {
        assert_eq!(model.context_length, Some(1_050_000));
        assert_eq!(model.max_output_tokens, Some(128_000));
        assert_eq!(model.function_call_available, Some(true));
        assert_eq!(model.response_format_available, Some(true));
        assert_eq!(model.native_multimodal, Some(true));
    }
    assert_eq!(gemini_37_flash.context_length, Some(1_048_576));
    assert_eq!(gemini_37_flash.max_output_tokens, Some(65_536));
    assert_eq!(gemini_37_flash.function_call_available, Some(true));
    assert_eq!(gemini_37_flash.response_format_available, Some(true));
    assert_eq!(gemini_37_flash.native_multimodal, Some(true));
    let gemini_37_capabilities = gemini_37_flash.capabilities();
    assert!(gemini_37_capabilities.tools);
    assert_eq!(
        gemini_37_capabilities.structured_output,
        vv_llm::StructuredOutputCapability::JsonSchema
    );
    assert_eq!(
        gemini_37_capabilities.thinking,
        vv_llm::ThinkingCapability::Configurable
    );
    assert!(gemini_37_capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));
    assert!(gemini_37_capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Video));
    assert!(gemini_37_capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Audio));
    assert_eq!(grok_46.context_length, Some(500_000));
    assert_eq!(grok_46.max_output_tokens, None);
    assert_eq!(grok_46.function_call_available, Some(true));
    assert_eq!(grok_46.response_format_available, Some(true));
    assert_eq!(grok_46.native_multimodal, Some(true));
    let grok_46_capabilities = grok_46.capabilities();
    assert!(grok_46_capabilities.tools);
    assert_eq!(
        grok_46_capabilities.structured_output,
        vv_llm::StructuredOutputCapability::JsonSchema
    );
    assert_eq!(
        grok_46_capabilities.thinking,
        vv_llm::ThinkingCapability::Configurable
    );
    assert!(grok_46_capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));
}

#[test]
fn default_deepseek_catalog_exposes_typed_thinking_capabilities() {
    let settings = LlmSettings::from_json_str("{}").unwrap();
    let backend = settings
        .backends
        .get("deepseek")
        .expect("deepseek defaults should exist");

    let flash = backend
        .models
        .get("deepseek-v4-flash")
        .expect("deepseek-v4-flash should exist")
        .capabilities();
    assert!(flash.tools);
    assert_eq!(
        flash.structured_output,
        vv_llm::StructuredOutputCapability::JsonSchema
    );
    assert_eq!(flash.thinking, vv_llm::ThinkingCapability::Configurable);

    let vision = backend
        .models
        .get("deepseek-v4-flash-vision-exp")
        .expect("deepseek-v4-flash-vision-exp should exist");
    assert_eq!(vision.context_length, Some(1_000_000));
    assert_eq!(vision.max_output_tokens, Some(384_000));
    assert_eq!(vision.native_multimodal, Some(true));
    assert_eq!(vision.max_image_dimension, Some(8192));
    let vision_capabilities = vision.capabilities();
    assert!(vision_capabilities.tools);
    assert_eq!(
        vision_capabilities.structured_output,
        vv_llm::StructuredOutputCapability::JsonSchema
    );
    assert_eq!(
        vision_capabilities.thinking,
        vv_llm::ThinkingCapability::Configurable
    );
    assert!(vision_capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));

    let reasoner = backend
        .models
        .get("deepseek-reasoner")
        .expect("deepseek-reasoner should exist")
        .capabilities();
    assert_eq!(reasoner.thinking, vv_llm::ThinkingCapability::AlwaysEnabled);
}

#[test]
fn default_zhipuai_catalog_exposes_glm_53_capabilities() {
    let settings = LlmSettings::from_json_str("{}").unwrap();
    let model = settings
        .backends
        .get("zhipuai")
        .and_then(|backend| backend.models.get("glm-5.3"))
        .expect("glm-5.3 should exist");

    assert_eq!(model.context_length, Some(1_000_000));
    assert_eq!(model.max_output_tokens, Some(128_000));
    assert_eq!(model.function_call_available, Some(true));
    assert_eq!(model.response_format_available, Some(true));
    assert_eq!(model.native_multimodal, Some(false));

    let capabilities = model.capabilities();
    assert!(capabilities.tools);
    assert_eq!(
        capabilities.structured_output,
        vv_llm::StructuredOutputCapability::JsonSchema
    );
    assert_eq!(
        capabilities.thinking,
        vv_llm::ThinkingCapability::AlwaysEnabled
    );
    assert!(!capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));
}

#[test]
fn default_qwen_38_catalog_exposes_hosted_api_capabilities() {
    let settings = LlmSettings::from_json_str("{}").unwrap();
    let backend = settings
        .backends
        .get("qwen")
        .expect("qwen defaults should exist");

    for model_name in ["qwen3.8-max", "qwen3.8-27b"] {
        let model = backend
            .models
            .get(model_name)
            .expect("Qwen 3.8 model should exist");
        assert_eq!(model.context_length, Some(1_000_000));
        assert_eq!(model.max_output_tokens, Some(131_072));
        assert_eq!(model.function_call_available, Some(true));
        assert_eq!(model.response_format_available, Some(true));
        assert_eq!(model.native_multimodal, Some(true));

        let capabilities = model.capabilities();
        assert!(capabilities.tools);
        assert_eq!(
            capabilities.structured_output,
            vv_llm::StructuredOutputCapability::JsonSchema
        );
        assert_eq!(
            capabilities.thinking,
            vv_llm::ThinkingCapability::Configurable
        );
        assert!(capabilities
            .input_modalities
            .contains(&vv_llm::Modality::Image));
        assert!(capabilities
            .input_modalities
            .contains(&vv_llm::Modality::Video));
    }

    assert!(
        backend
            .models
            .get("qwen3.8-max")
            .expect("qwen3.8-max should exist")
            .capabilities()
            .parallel_tool_calls
    );
    assert!(
        !backend
            .models
            .get("qwen3.8-27b")
            .expect("qwen3.8-27b should exist")
            .capabilities()
            .parallel_tool_calls
    );
}

#[test]
fn legacy_overrides_update_default_capabilities_without_losing_thinking_metadata() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"deepseek-default","api_base":"https://api.deepseek.com","api_key":"sk-test"}],
      "backends": {
        "deepseek": {
          "models": {
            "deepseek-v4-flash": {
              "id":"deepseek-v4-flash",
              "endpoints":["deepseek-default"],
              "function_call_available": false,
              "response_format_available": false,
              "native_multimodal": true
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let capabilities = settings
        .resolve_chat_model(BackendType::DeepSeek, "deepseek-v4-flash")
        .unwrap()
        .model
        .capabilities();

    assert!(!capabilities.tools);
    assert_eq!(
        capabilities.structured_output,
        vv_llm::StructuredOutputCapability::None
    );
    assert!(capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));
    assert_eq!(
        capabilities.thinking,
        vv_llm::ThinkingCapability::Configurable
    );
}

#[test]
fn merges_user_chat_model_overrides_with_python_defaults() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"dashscope-default","api_base":"https://dashscope.aliyuncs.com/compatible-mode/v1","api_key":"sk-test"}],
      "backends": {
        "qwen": {
          "default_endpoint": "dashscope-default",
          "models": {
            "qwen3.7-max": {
              "id":"qwen3.7-max",
              "max_output_tokens": 2048,
              "native_multimodal": true
            }
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Qwen, "qwen3.7-max")
        .unwrap();

    assert_eq!(resolved.endpoint.id, "dashscope-default");
    assert_eq!(resolved.model.context_length, Some(1_000_000));
    assert_eq!(resolved.model.max_output_tokens, Some(2048));
    assert_eq!(resolved.model.native_multimodal, Some(true));
}

#[test]
fn applies_python_defaults_to_user_defined_chat_models() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"openai-default","api_base":"https://api.openai.com/v1","api_key":"sk-test"}],
      "backends": {
        "openai": {
          "models": {
            "custom-model": {"id":"custom-model","endpoints":["openai-default"]}
          }
        }
      }
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::OpenAI, "custom-model")
        .unwrap();

    assert_eq!(resolved.model.context_length, Some(32_768));
    assert_eq!(resolved.model.function_call_available, Some(false));
    assert_eq!(resolved.model.response_format_available, Some(false));
    assert_eq!(resolved.model.native_multimodal, Some(false));
}

#[test]
fn settings_preserve_python_endpoint_metadata_fields() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{
        "id":"azure-openai",
        "enabled": false,
        "api_base":"https://example.openai.azure.com",
        "api_key":"sk-test",
        "response_api": true,
        "endpoint_type": "openai_azure",
        "is_azure": true,
        "rpm": 120,
        "tpm": 600000,
        "concurrent_requests": 8,
        "proxy": "http://proxy.local:8080",
        "headers": {"X-Test": "value"}
      }],
      "backends": {}
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let endpoint = &settings.endpoints[0];

    assert!(!endpoint.enabled);
    assert!(endpoint.response_api);
    assert!(endpoint.is_azure);
    assert_eq!(endpoint.rpm, 120);
    assert_eq!(endpoint.tpm, 600_000);
    assert_eq!(endpoint.concurrent_requests, 8);
    assert_eq!(endpoint.proxy.as_deref(), Some("http://proxy.local:8080"));
    assert_eq!(
        endpoint
            .headers
            .as_ref()
            .and_then(|headers| headers.get("X-Test"))
            .map(String::as_str),
        Some("value")
    );
}
