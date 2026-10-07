use crate::{CapabilityPolicy, ChatRequest, ModelCapabilities, VvLlmError};
use serde_json::{json, Map, Value};

pub(super) fn model_capabilities(
    model: &str,
    configured_model: &str,
    configured: Option<&ModelCapabilities>,
) -> ModelCapabilities {
    if model == configured_model {
        if let Some(capabilities) = configured {
            return capabilities.clone();
        }
    }
    crate::defaults::default_model_capabilities(model)
}

pub(crate) fn resolve_body(request: &ChatRequest, protocol: &str) -> Result<Value, VvLlmError> {
    let mut body = match &request.extra_body {
        Value::Null => Map::new(),
        Value::Object(object) => object.clone(),
        _ => {
            return Err(VvLlmError::Configuration(
                "extra_body must be an object".to_string(),
            ))
        }
    };
    let container = if protocol == "anthropic" {
        Some("output_config")
    } else {
        None
    };
    let mut efforts = Vec::new();
    if let Some(effort) = &request.options.reasoning_effort {
        efforts.push(json!(effort));
    }
    if let Some(effort) = body.remove("reasoning_effort") {
        efforts.push(effort);
    }
    let containers: &[&str] = match protocol {
        "anthropic" => &["output_config"],
        "validation" => &["reasoning", "output_config"],
        _ => &[],
    };
    for container in containers {
        if let Some(effort) = body.get(*container).and_then(|config| config.get("effort")) {
            efforts.push(effort.clone());
        }
    }
    if efforts
        .iter()
        .any(|value| value.as_str().is_none_or(|value| value.trim().is_empty()))
    {
        return Err(VvLlmError::Configuration(
            "reasoning_effort must be a non-empty string".to_string(),
        ));
    }
    if efforts.iter().any(|value| Some(value) != efforts.first()) {
        return Err(VvLlmError::Configuration(
            "Conflicting reasoning_effort values".to_string(),
        ));
    }
    if let Some(effort) = efforts.first() {
        for config in [
            body.get("google"),
            body.get("extra_body").and_then(|value| value.get("google")),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(thinking) = config.get("thinking_config") {
                if thinking.get("thinking_level").is_some()
                    || thinking.get("thinking_budget").is_some()
                    || thinking.get("thinkingLevel").is_some()
                    || thinking.get("thinkingBudget").is_some()
                {
                    return Err(VvLlmError::Configuration(
                        "reasoning_effort conflicts with Gemini thinking_level/thinking_budget"
                            .to_string(),
                    ));
                }
            }
        }
        if let Some(container) = container {
            let config = body
                .entry(container)
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| {
                    VvLlmError::Configuration(format!("{container} must be an object"))
                })?;
            config.insert("effort".to_string(), effort.clone());
        } else {
            body.insert("reasoning_effort".to_string(), effort.clone());
        }
    }
    if let Some(thinking) = &request.options.thinking {
        if let Some(existing) = body.get_mut("thinking") {
            merge_thinking(existing, thinking)?;
        } else {
            body.insert("thinking".to_string(), thinking.clone());
        }
    }
    Ok(Value::Object(body))
}

pub(crate) fn validate_body(
    body: &Value,
    model: &str,
    protocol: &str,
    capabilities: &ModelCapabilities,
    policy: CapabilityPolicy,
) -> Result<(), VvLlmError> {
    if body
        .get("model")
        .is_some_and(|value| value.as_str() != Some(model))
    {
        return Err(VvLlmError::Configuration(
            "Conflicting model values".to_string(),
        ));
    }
    let effort = if protocol == "anthropic" {
        body.get("output_config")
            .and_then(|value| value.get("effort"))
    } else {
        body.get("reasoning_effort")
    };
    capabilities.validate_reasoning_effort(model, effort.and_then(Value::as_str), policy)
}

fn merge_thinking(target: &mut Value, addition: &Value) -> Result<(), VvLlmError> {
    match (target, addition) {
        (Value::Object(target), Value::Object(addition)) => {
            for (key, value) in addition {
                if let Some(existing) = target.get_mut(key) {
                    merge_thinking(existing, value)?;
                } else {
                    target.insert(key.clone(), value.clone());
                }
            }
            Ok(())
        }
        (target, addition) if target == addition => Ok(()),
        _ => Err(VvLlmError::Configuration(
            "Conflicting reasoning control: thinking".to_string(),
        )),
    }
}

/// Gemini 3+ uses thinking levels and the provider's default sampling profile.
pub(super) fn normalize_gemini_body(model: &str, body: &mut Value) -> Result<(), VvLlmError> {
    let model = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    let modern = model
        .strip_prefix("gemini-")
        .and_then(|suffix| suffix.split(['.', '-']).next())
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 3);
    if !modern {
        return Ok(());
    }
    let Some(object) = body.as_object_mut() else {
        return Ok(());
    };
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "topP",
        "topK",
        "thinking_budget",
        "thinkingBudget",
    ] {
        object.remove(key);
    }
    if let Some(level) = object.remove("thinkingLevel") {
        if object
            .get("thinking_level")
            .is_some_and(|existing| existing != &level)
        {
            return Err(VvLlmError::Configuration(
                "Conflicting Gemini thinking_level values".into(),
            ));
        }
        object.insert("thinking_level".into(), level);
    }
    for key in ["extra_body", "google", "thinking_config"] {
        if let Some(value) = object.get_mut(key) {
            normalize_gemini_body(&model, value)?;
        }
    }
    Ok(())
}
