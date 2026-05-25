use std::{env, path::PathBuf, time::Instant};
use vv_llm::{EndpointConfig, LlmSettings, ResolvedModelConfig, VvLlmError};

const TRUTHY: &[&str] = &["1", "true", "yes", "on"];

pub fn live_enabled() -> bool {
    env_bool("VV_LLM_RUN_LIVE_TESTS", false)
}

pub fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| TRUTHY.contains(&value.trim().to_ascii_lowercase().as_str()))
        .unwrap_or(default)
}

pub fn load_live_settings(require_credentials: bool) -> Result<LlmSettings, VvLlmError> {
    let path = live_settings_path();
    let settings = LlmSettings::from_json_file(&path)?;
    let allow_empty = env_bool("VV_LLM_ALLOW_EMPTY_KEYS", false);
    if require_credentials && !allow_empty && !has_live_credentials(&settings) {
        return Err(VvLlmError::Configuration(format!(
            "live settings at {} have no usable credentials; create crates/vv-llm/tests/fixtures/dev_settings.json or set VV_LLM_ALLOW_EMPTY_KEYS=1",
            path.display()
        )));
    }
    Ok(settings)
}

pub fn live_settings_path() -> PathBuf {
    env::var("VV_LLM_SETTINGS_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join("dev_settings.json");
            if dev.exists() {
                dev
            } else {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests")
                    .join("fixtures")
                    .join("sample_settings.json")
            }
        })
}

pub fn has_live_credentials(settings: &LlmSettings) -> bool {
    settings.endpoints.iter().any(endpoint_has_live_credentials)
}

pub fn endpoint_has_live_credentials(endpoint: &EndpointConfig) -> bool {
    endpoint
        .api_key
        .as_deref()
        .map(|key| {
            let key = key.trim();
            !key.is_empty() && !key.starts_with("YOUR_")
        })
        .unwrap_or(false)
}

pub fn run_with_timer<T>(label: &str, operation: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = operation();
    eprintln!(
        "[live] {label} finished in {:.2}s",
        start.elapsed().as_secs_f32()
    );
    result
}

pub fn require_live() {
    if !live_enabled() {
        eprintln!("Live tests are disabled. Set VV_LLM_RUN_LIVE_TESTS=1 to run.");
    }
}

pub fn resolved_parts(resolved: ResolvedModelConfig) -> (String, String, String) {
    (
        resolved.model.id,
        resolved.endpoint.api_base.unwrap_or_default(),
        resolved.endpoint.api_key.unwrap_or_default(),
    )
}
