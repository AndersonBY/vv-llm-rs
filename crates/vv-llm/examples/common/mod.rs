use std::path::PathBuf;
use vv_llm::{create_chat_client_from_resolved, BackendType, ChatClient, LlmSettings, VvLlmError};

pub fn load_deepseek_client() -> Result<(Box<dyn ChatClient>, String), VvLlmError> {
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
    let resolved = settings.resolve_chat_model(BackendType::DeepSeek, "deepseek-v4-flash")?;
    let model = resolved.model_id.clone();
    Ok((create_chat_client_from_resolved(resolved)?, model))
}
