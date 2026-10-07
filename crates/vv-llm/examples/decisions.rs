//! Opt-in Decisions smoke test using an explicit local settings file.
use serde_json::{json, Value};
use vv_llm::{
    create_decision_client_from_resolved, DecisionAnswer, DecisionRequest, LlmSettings, VvLlmError,
};

async fn run() -> Result<Value, VvLlmError> {
    let source = std::env::var("VV_LLM_SETTINGS_JSON")
        .map_err(|_| VvLlmError::Configuration("set VV_LLM_SETTINGS_JSON".into()))?;
    let mut raw: Value = serde_json::from_str(
        &std::fs::read_to_string(source)
            .map_err(|_| VvLlmError::Configuration("cannot read live settings".into()))?,
    )?;
    if raw.get("decision_backends").is_none() {
        let mut backend = raw["backends"]["openai"].clone();
        if backend
            .get("default_endpoint")
            .and_then(Value::as_str)
            .is_none()
        {
            let endpoints = raw["endpoints"]
                .as_array()
                .ok_or_else(|| VvLlmError::Configuration("missing live endpoints".into()))?;
            let models = backend["models"]
                .as_object()
                .ok_or_else(|| VvLlmError::Configuration("missing OpenAI bindings".into()))?;
            let selected = models
                .values()
                .flat_map(|model| model["endpoints"].as_array().into_iter().flatten())
                .find_map(|binding| {
                    if binding.get("enabled") == Some(&Value::Bool(false)) {
                        return None;
                    }
                    let id = binding
                        .as_str()
                        .or_else(|| binding["endpoint_id"].as_str())?;
                    let endpoint = endpoints.iter().find(|endpoint| endpoint["id"] == id)?;
                    let kind = endpoint["endpoint_type"].as_str().unwrap_or("default");
                    (endpoint.get("enabled") != Some(&Value::Bool(false))
                        && endpoint["api_key"]
                            .as_str()
                            .is_some_and(|key| !key.is_empty())
                        && ["default", "openai"].contains(&kind)
                        && !["is_azure", "is_vertex", "is_bedrock"]
                            .iter()
                            .any(|flag| endpoint[*flag] == true))
                    .then(|| id.to_string())
                });
            if let Some(id) = selected {
                backend["default_endpoint"] = Value::String(id);
            }
        }
        raw["decision_backends"] = json!({"openai": backend});
    }
    let model = std::env::var("VV_LLM_MODEL").unwrap_or_else(|_| "gpt-6-luna".to_string());
    let backend = &raw["decision_backends"]["openai"];
    let selected = backend["models"].get(&model).cloned();
    let mut ids = std::collections::HashSet::new();
    if let Some(id) = backend["default_endpoint"].as_str() {
        ids.insert(id.to_string());
    }
    if let Some(bindings) = selected
        .as_ref()
        .and_then(|model| model["endpoints"].as_array())
    {
        for binding in bindings {
            if let Some(id) = binding.as_str().or_else(|| binding["endpoint_id"].as_str()) {
                ids.insert(id.to_string());
            }
        }
    }
    let endpoints: Vec<_> = raw["endpoints"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|endpoint| endpoint["id"].as_str().is_some_and(|id| ids.contains(id)))
        .cloned()
        .collect();
    let models = selected
        .map(|entry| json!({model.clone(): entry}))
        .unwrap_or_else(|| json!({}));
    let isolated = json!({"endpoints": endpoints, "decision_backends": {"openai": {"default_endpoint": backend["default_endpoint"].clone(), "models": models}}});
    let settings = LlmSettings::from_json_str(&isolated.to_string())?;
    let resolved = settings.resolve_decision_model("openai", &model)?;
    let client = create_decision_client_from_resolved(&resolved)?;
    let fixture: Value = serde_json::from_str(include_str!(
        "../contract/v1.3.0/fixtures/decisions.v1.json"
    ))?;
    let mut request = DecisionRequest::from_contract(&fixture["request"])?;
    request.model.clear();
    let response = client.create(request).await?;
    let valid = match response.answers.as_slice() {
        [DecisionAnswer::Predicate { probability, .. }, DecisionAnswer::Choice { choice, .. }, DecisionAnswer::Score { score, .. }] => {
            *probability > 0.5
                && choice == &vv_llm::ChoiceValue::String("replacement".into())
                && (0.0..=2.0).contains(score)
        }
        _ => false,
    };
    if !valid {
        return Err(VvLlmError::Configuration(
            "decision semantic check failed".into(),
        ));
    }
    Ok(
        json!({"decision_live_smoke": "passed", "answer_types": ["predicate", "choice", "score"], "semantic_checks_passed": true, "usage": response.usage.map(|usage| json!({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.total_tokens}))}),
    )
}

#[tokio::main]
async fn main() {
    if std::env::var("VV_LLM_RUN_LIVE_TESTS").as_deref() != Ok("1") {
        println!("decision live smoke skipped; set VV_LLM_RUN_LIVE_TESTS=1");
        return;
    }
    match run().await {
        Ok(result) => println!("{result}"),
        Err(error) => {
            let status = match &error {
                VvLlmError::Classified(details) => details.status_code,
                _ => None,
            };
            let diagnostic = match &error {
                VvLlmError::Configuration(message) if message.contains("proxy") => {
                    "proxy_configuration"
                }
                VvLlmError::Configuration(message) if message.contains("header") => {
                    "header_configuration"
                }
                VvLlmError::Configuration(message) if message.contains("OpenAI endpoint") => {
                    "endpoint_transport"
                }
                VvLlmError::Configuration(message) if message.contains("missing") => {
                    "missing_configuration"
                }
                VvLlmError::Configuration(message) if message.contains("top-level") => {
                    "legacy_configuration"
                }
                VvLlmError::Configuration(message) if message.contains("declare support") => {
                    "model_capabilities"
                }
                VvLlmError::EndpointNotFound(_) => "endpoint_binding",
                VvLlmError::ModelNotFound { .. } => "model_binding",
                VvLlmError::Classified(_) => "classified_error",
                _ => "other_error",
            };
            println!(
                "{}",
                json!({"decision_live_smoke": "failed", "kind": error.kind(), "status": status, "diagnostic": diagnostic})
            );
            std::process::exit(1);
        }
    }
}
