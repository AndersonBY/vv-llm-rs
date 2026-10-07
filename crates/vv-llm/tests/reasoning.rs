use serde_json::{json, Value};
use vv_llm::{
    chat_clients::{AnthropicChatClient, OpenAiCompatibleChatClient},
    BackendType, CapabilityPolicy, ChatRequest, LlmSettings, ModelCapabilities,
};

#[test]
fn shared_reasoning_effort_cases() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../contract/v1.3.0/fixtures/reasoning-effort.v1.json"
    ))
    .unwrap();
    for case in fixture["capability_cases"].as_array().unwrap() {
        let capabilities: ModelCapabilities =
            serde_json::from_value(json!({"reasoning_efforts": case["reasoning_efforts"], "reasoning_effort_aliases": case["reasoning_effort_aliases"]}))
                .unwrap();
        let mut request = ChatRequest::new(case["model"].as_str().unwrap(), vec![]);
        request.options.reasoning_effort = case["reasoning_effort"].as_str().map(str::to_string);
        assert_eq!(
            capabilities.validate_request(&request).is_ok(),
            case["valid"].as_bool().unwrap()
        );
        assert!(capabilities
            .validate_reasoning_effort(
                &request.model,
                request.options.reasoning_effort.as_deref(),
                CapabilityPolicy::Passthrough
            )
            .is_ok());
    }
}

#[test]
fn adapters_preserve_defaults_and_map_explicit_effort() {
    let capabilities = ModelCapabilities {
        reasoning_efforts: Some(vec!["none".into(), "high".into()]),
        ..Default::default()
    };
    let openai = OpenAiCompatibleChatClient::new("model", "https://example.invalid", "test-key")
        .with_capabilities(capabilities.clone())
        .with_capability_policy(CapabilityPolicy::Strict);
    let anthropic = AnthropicChatClient::new("model", "https://example.invalid", "test-key")
        .with_capabilities(capabilities)
        .with_capability_policy(CapabilityPolicy::Strict);
    let mut request = ChatRequest::new("model", vec![]);
    assert!(openai
        .to_openai_json(&request)
        .unwrap()
        .get("reasoning_effort")
        .is_none());
    assert!(anthropic
        .to_anthropic_json(&request)
        .unwrap()
        .get("output_config")
        .is_none());
    request.options.reasoning_effort = Some("none".into());
    assert_eq!(
        openai.to_openai_json(&request).unwrap()["reasoning_effort"],
        "none"
    );
    request.options.reasoning_effort = Some("high".into());
    request.extra_body = json!({"output_config": {"format": {"type": "json_schema"}}});
    let body = anthropic.to_anthropic_json(&request).unwrap();
    assert_eq!(body["output_config"]["effort"], "high");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert!(body.get("reasoning_effort").is_none());
    request.extra_body = json!({"output_config": {"effort": "low"}});
    assert!(anthropic
        .to_anthropic_json(&request)
        .unwrap_err()
        .to_string()
        .contains("Conflicting"));
    request.extra_body = json!({"reasoning_effort": "low"});
    assert!(openai
        .to_openai_json(&request)
        .unwrap_err()
        .to_string()
        .contains("Conflicting"));
    request.extra_body = json!({"google": {"thinking_config": {"thinking_budget": 0}}});
    assert!(openai
        .to_openai_json(&request)
        .unwrap_err()
        .to_string()
        .contains("thinking_level/thinking_budget"));
}

#[test]
fn adapters_validate_selected_model_and_support_passthrough() {
    let client = OpenAiCompatibleChatClient::new("model", "https://example.invalid", "test-key")
        .with_capabilities(ModelCapabilities {
            reasoning_efforts: Some(vec!["low".into()]),
            ..Default::default()
        })
        .with_capability_policy(CapabilityPolicy::Strict);
    let mut request = ChatRequest::new("model", vec![]);
    request.options.reasoning_effort = Some("high".into());
    assert!(client
        .to_openai_json(&request)
        .unwrap_err()
        .to_string()
        .contains("Supported values: low"));
    request.model = "gpt-5.5".into();
    assert_eq!(
        client.to_openai_json(&request).unwrap()["reasoning_effort"],
        "high"
    );
    request.model = "unknown".into();
    request.options.reasoning_effort = Some("ultra".into());
    assert!(client
        .to_openai_json(&request)
        .unwrap_err()
        .to_string()
        .contains("support is unknown"));
    let passthrough = client.with_capability_policy(CapabilityPolicy::Passthrough);
    assert_eq!(
        passthrough.to_openai_json(&request).unwrap()["reasoning_effort"],
        "ultra"
    );
}

#[test]
fn settings_binding_overrides_do_not_mutate_model_metadata() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../contract/v1.3.0/fixtures/settings-resolution.v1.json"
    ))
    .unwrap();
    let settings = LlmSettings::from_json_str(&fixture["settings"].to_string()).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::DeepSeek, "chat-alias")
        .unwrap();
    assert_eq!(
        resolved.model.capabilities().reasoning_efforts,
        Some(vec!["low".into(), "high".into()])
    );
    let capabilities = resolved.model.capabilities();
    assert_eq!(
        capabilities
            .reasoning_effort_aliases
            .as_ref()
            .unwrap()
            .get("ultra")
            .map(String::as_str),
        Some("low")
    );
    assert!(capabilities
        .validate_reasoning_effort("chat-alias", Some("ultra"), CapabilityPolicy::Strict)
        .is_ok());
    assert!(capabilities
        .validate_reasoning_effort("chat-alias", Some("max"), CapabilityPolicy::Strict)
        .is_err());
    assert_eq!(
        settings.backends["deepseek"].models["chat-alias"]
            .capabilities()
            .reasoning_efforts,
        Some(vec!["low".into(), "medium".into(), "high".into()])
    );
}

#[test]
fn request_validation_checks_native_and_extra_efforts() {
    let capabilities = ModelCapabilities {
        reasoning_efforts: Some(vec!["low".into()]),
        ..Default::default()
    };
    for body in [
        json!({"reasoning_effort": "high"}),
        json!({"output_config": {"effort": "high"}}),
        json!({"reasoning": {"effort": "high"}}),
    ] {
        let mut request = ChatRequest::new("model", vec![]);
        request.extra_body = body;
        assert!(capabilities.validate_request(&request).is_err());
    }
}

#[test]
fn adapters_reject_model_overrides_and_merge_thinking_fields() {
    let openai = OpenAiCompatibleChatClient::new("model", "https://example.invalid", "test-key")
        .with_capability_policy(CapabilityPolicy::Passthrough);
    let anthropic = AnthropicChatClient::new("model", "https://example.invalid", "test-key")
        .with_capability_policy(CapabilityPolicy::Passthrough);
    let mut request = ChatRequest::new("model", vec![]);
    request.extra_body = json!({"model": "other"});
    assert!(openai
        .to_openai_json(&request)
        .unwrap_err()
        .to_string()
        .contains("model"));
    assert!(anthropic
        .to_anthropic_json(&request)
        .unwrap_err()
        .to_string()
        .contains("model"));
    request.options.thinking = Some(json!({"type": "enabled"}));
    request.extra_body = json!({"thinking": {"budget_tokens": 1024}});
    let expected = json!({"type": "enabled", "budget_tokens": 1024});
    assert_eq!(
        openai.to_openai_json(&request).unwrap()["thinking"],
        expected
    );
    assert_eq!(
        anthropic.to_anthropic_json(&request).unwrap()["thinking"],
        expected
    );
    request.extra_body = json!({"thinking": {"type": "disabled"}});
    assert!(openai.to_openai_json(&request).is_err());
}

#[test]
fn partial_effort_override_preserves_legacy_flags_and_rejects_invalid_metadata() {
    let mut raw = json!({"backends": {"deepseek": {"models": {"deepseek-flash": {"id": "deepseek-flash", "function_call_available": false, "response_format_available": false, "native_multimodal": false, "capabilities": {"reasoning_efforts": ["high"]}}}}}});
    let settings = LlmSettings::from_json_str(&raw.to_string()).unwrap();
    let capabilities = settings.backends["deepseek"].models["deepseek-flash"].capabilities();
    assert!(!capabilities.tools);
    assert_eq!(
        capabilities.structured_output,
        vv_llm::StructuredOutputCapability::None
    );
    assert!(!capabilities
        .input_modalities
        .contains(&vv_llm::Modality::Image));
    raw["backends"]["deepseek"]["models"]["deepseek-flash"]["capabilities"]["tools"] = json!(true);
    assert!(
        LlmSettings::from_json_str(&raw.to_string())
            .unwrap()
            .backends["deepseek"]
            .models["deepseek-flash"]
            .capabilities()
            .tools
    );
    raw["backends"]["deepseek"]["models"]["deepseek-flash"]["capabilities"]["reasoning_efforts"] =
        json!([" "]);
    assert!(LlmSettings::from_json_str(&raw.to_string()).is_err());
}

#[test]
fn gemini_wire_parameters_preserve_legacy_models_and_inputs() {
    for model in [
        "gemini-3.8-flash",
        "google/gemini-4-flash",
        "gemini-2.5-flash",
        "gpt-5.5",
    ] {
        for stream in [false, true] {
            for nested in [false, true] {
                let client =
                    OpenAiCompatibleChatClient::new(model, "https://example.invalid", "test-key");
                let mut request = ChatRequest::new(model, vec![]);
                request.options.temperature = Some(0.3);
                request.options.top_p = Some(0.7);
                request.options.stream = Some(stream);
                let google = json!({"google": {"thinking_config": {"thinkingBudget": 1024, "thinkingLevel": "high", "include_thoughts": true}}});
                request.extra_body = if nested {
                    json!({"extra_body": google})
                } else {
                    google.clone()
                };
                for (key, value) in [
                    ("temperature", json!(0.4)),
                    ("top_p", json!(0.8)),
                    ("top_k", json!(20)),
                    ("topP", json!(0.8)),
                    ("topK", json!(20)),
                ] {
                    request.extra_body[key] = value;
                }
                let original = request.extra_body.clone();
                let sent = client.to_openai_json(&request).unwrap();
                let modern = model.contains("gemini-3") || model.contains("gemini-4");
                for key in ["temperature", "top_p", "top_k", "topP", "topK"] {
                    assert_eq!(sent.get(key).is_none(), modern);
                }
                let thinking = &(if nested { &sent["extra_body"] } else { &sent })["google"]
                    ["thinking_config"];
                assert_eq!(
                    thinking,
                    &if modern {
                        json!({"thinking_level": "high", "include_thoughts": true})
                    } else {
                        google["google"]["thinking_config"].clone()
                    }
                );
                assert_eq!(request.extra_body, original);
                request.extra_body = json!({"google": {"thinking_config": {"thinking_budget": -1, "include_thoughts": true}}});
                let sent = client.to_openai_json(&request).unwrap();
                assert_eq!(
                    sent["google"]["thinking_config"]
                        .get("thinking_budget")
                        .is_none(),
                    modern
                );
            }
        }
    }
}

#[test]
fn gemini_level_conflicts_and_minimal_are_rejected() {
    for model in ["gemini-3.7-flash", "gemini-3.8-flash"] {
        let client = OpenAiCompatibleChatClient::new(model, "https://example.invalid", "test-key")
            .with_capability_policy(CapabilityPolicy::Strict);
        let mut request = ChatRequest::new(model, vec![]);
        request.options.reasoning_effort = Some("minimal".into());
        assert!(client.to_openai_json(&request).is_err());
        for effort in ["low", "medium", "high"] {
            request.options.reasoning_effort = Some(effort.into());
            assert_eq!(
                client.to_openai_json(&request).unwrap()["reasoning_effort"],
                effort
            );
        }
        request.extra_body = json!({"google": {"thinking_config": {"thinkingLevel": "high"}}});
        assert!(client
            .to_openai_json(&request)
            .unwrap_err()
            .to_string()
            .contains("conflicts"));
        request.options.reasoning_effort = None;
        request.extra_body = json!({"google": {"thinking_config": {"thinkingLevel": "high", "thinking_level": "low"}}});
        assert!(client
            .to_openai_json(&request)
            .unwrap_err()
            .to_string()
            .contains("Conflicting Gemini"));
    }
}
