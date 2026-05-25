mod openai_compatible;

use crate::{EmbeddingResponse, VvLlmError};
use async_trait::async_trait;

pub use openai_compatible::OpenAiCompatibleEmbeddingClient;

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn create_embeddings(&self, input: &[&str]) -> Result<EmbeddingResponse, VvLlmError>;
}

pub fn create_embedding_client(
    _backend: impl Into<String>,
    model: impl Into<String>,
    api_base: impl Into<String>,
    api_key: impl Into<String>,
) -> Box<dyn EmbeddingClient> {
    Box::new(OpenAiCompatibleEmbeddingClient::new(
        model, api_base, api_key,
    ))
}
