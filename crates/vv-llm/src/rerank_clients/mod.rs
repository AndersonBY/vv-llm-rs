mod custom_json_http;

use crate::{RerankResponse, VvLlmError};
use async_trait::async_trait;

pub use custom_json_http::{CustomJsonHttpRerankClient, RerankMapping};

#[async_trait]
pub trait RerankClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn rerank(&self, query: &str, documents: &[&str]) -> Result<RerankResponse, VvLlmError>;
}

pub fn create_rerank_client(
    model: impl Into<String>,
    api_base: impl Into<String>,
    api_key: impl Into<String>,
    mapping: RerankMapping,
) -> Box<dyn RerankClient> {
    Box::new(CustomJsonHttpRerankClient::new(
        model, api_base, api_key, mapping,
    ))
}
