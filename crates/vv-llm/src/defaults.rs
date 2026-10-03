use crate::{BackendConfig, BackendType, ModelCapabilities};
use serde::Deserialize;
use std::{collections::HashMap, sync::OnceLock};

const DEFAULT_CHAT_CATALOG_JSON: &str =
    include_str!("../contract/v1.2.1/catalog/default-chat-catalog.json");

#[derive(Debug, Deserialize)]
struct DefaultChatCatalog {
    #[serde(default)]
    default_models: HashMap<String, String>,
    backends: HashMap<String, BackendConfig>,
}

fn default_chat_catalog() -> &'static DefaultChatCatalog {
    static CATALOG: OnceLock<DefaultChatCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str(DEFAULT_CHAT_CATALOG_JSON)
            .expect("default chat catalog JSON must be valid")
    })
}

pub(crate) fn default_model_capabilities(model: &str) -> ModelCapabilities {
    default_chat_catalog()
        .backends
        .values()
        .flat_map(|backend| backend.models.values())
        .find(|entry| entry.id == model)
        .map(|entry| entry.capabilities())
        .unwrap_or_default()
}

pub fn default_chat_backends() -> HashMap<String, BackendConfig> {
    default_chat_catalog().backends.clone()
}

pub fn default_chat_model(backend: BackendType) -> Option<&'static str> {
    default_chat_catalog()
        .default_models
        .get(backend.as_str())
        .map(String::as_str)
}
