use crate::{BackendConfig, BackendType};
use serde::Deserialize;
use std::collections::HashMap;

const DEFAULT_CHAT_CATALOG_JSON: &str = include_str!("default_chat_catalog.json");

#[derive(Debug, Deserialize)]
struct DefaultChatCatalog {
    backends: HashMap<String, BackendConfig>,
}

pub fn default_chat_backends() -> HashMap<String, BackendConfig> {
    serde_json::from_str::<DefaultChatCatalog>(DEFAULT_CHAT_CATALOG_JSON)
        .expect("default chat catalog JSON must be valid")
        .backends
}

pub fn default_chat_model(backend: BackendType) -> Option<&'static str> {
    match backend {
        BackendType::Moonshot => Some("kimi-k2.6"),
        BackendType::DeepSeek => Some("deepseek-v4-pro"),
        BackendType::Baichuan => Some("Baichuan4"),
        BackendType::Groq => Some("llama3-70b-8192"),
        BackendType::Qwen => Some("qwen3.5-397b-a17b"),
        BackendType::Yi => Some("yi-lightning"),
        BackendType::ZhiPuAI => Some("glm-5.1"),
        BackendType::Mistral => Some("mistral-small"),
        BackendType::OpenAI => Some("gpt-5.5"),
        BackendType::Anthropic => Some("claude-opus-4-8"),
        BackendType::MiniMax => Some("MiniMax-M2.7"),
        BackendType::Gemini => Some("gemini-3.5-flash"),
        BackendType::Ernie => Some("ernie-4.5-8k-preview"),
        BackendType::StepFun => Some("step-3.5-flash"),
        BackendType::XAI => Some("grok-4.20-0309-reasoning"),
        BackendType::Xiaomi => Some("mimo-v2-pro"),
        BackendType::Local => None,
    }
}
