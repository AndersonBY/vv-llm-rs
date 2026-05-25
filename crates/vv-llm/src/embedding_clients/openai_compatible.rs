use crate::{EmbeddingData, EmbeddingResponse, VvLlmError};
use async_openai::{
    config::OpenAIConfig,
    types::embeddings::{CreateEmbeddingRequest, CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

use super::EmbeddingClient;

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbeddingClient {
    model: String,
    api_base: String,
    api_key: String,
}

impl OpenAiCompatibleEmbeddingClient {
    pub fn new(
        model: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_base: api_base.into(),
            api_key: api_key.into(),
        }
    }

    pub fn to_openai_json(&self, input: &[&str]) -> Result<serde_json::Value, VvLlmError> {
        Ok(serde_json::to_value(self.to_openai_request(input)?)?)
    }

    fn to_openai_request(&self, input: &[&str]) -> Result<CreateEmbeddingRequest, VvLlmError> {
        CreateEmbeddingRequestArgs::default()
            .model(self.model.clone())
            .input(EmbeddingInput::StringArray(
                input.iter().map(|value| (*value).to_string()).collect(),
            ))
            .build()
            .map_err(|error| VvLlmError::Provider(error.to_string()))
    }

    fn client(&self) -> Client<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_key(self.api_key.clone())
            .with_api_base(self.api_base.clone());
        Client::with_config(config)
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiCompatibleEmbeddingClient {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    async fn create_embeddings(&self, input: &[&str]) -> Result<EmbeddingResponse, VvLlmError> {
        let response = self
            .client()
            .embeddings()
            .create(self.to_openai_request(input)?)
            .await
            .map_err(|error| VvLlmError::Provider(error.to_string()))?;

        Ok(EmbeddingResponse {
            model: response.model,
            data: response
                .data
                .into_iter()
                .map(|embedding| EmbeddingData {
                    index: embedding.index,
                    embedding: embedding.embedding,
                })
                .collect(),
        })
    }
}
