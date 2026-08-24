use std::path::PathBuf;
use vv_llm::{create_chat_client_from_resolved, BackendType, ChatClient, LlmSettings, VvLlmError};

pub fn load_chat_client() -> Result<(Box<dyn ChatClient>, String), VvLlmError> {
    let settings_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("VV_LLM_SETTINGS_JSON").map(PathBuf::from))
        .ok_or_else(|| {
            VvLlmError::Configuration(
                "pass a settings JSON path or set VV_LLM_SETTINGS_JSON".to_string(),
            )
        })?;
    let settings = LlmSettings::from_json_file(settings_path)?;
    let backend_name = std::env::var("VV_LLM_BACKEND").map_err(|_| {
        VvLlmError::Configuration("set VV_LLM_BACKEND for the selected settings file".to_string())
    })?;
    let backend: BackendType =
        serde_json::from_value(serde_json::Value::String(backend_name.clone())).map_err(|_| {
            VvLlmError::Configuration(format!("unsupported VV_LLM_BACKEND: {backend_name}"))
        })?;
    let model = std::env::var("VV_LLM_MODEL").map_err(|_| {
        VvLlmError::Configuration("set VV_LLM_MODEL for the selected settings file".to_string())
    })?;
    let resolved = settings.resolve_chat_model(backend, &model)?;
    let model = resolved.model_id.clone();
    Ok((create_chat_client_from_resolved(resolved)?, model))
}
