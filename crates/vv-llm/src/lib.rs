//! Typed Rust clients for chat, streaming, embeddings, rerank, and LLM endpoint resolution.

pub mod chat_clients;
pub mod defaults;
pub mod embedding_clients;
pub mod rerank_clients;
pub mod settings;
pub mod types;
pub mod utilities;

pub use chat_clients::{
    create_chat_client, create_chat_client_from_resolved, ChatClient, ChatStream,
    GoogleAccessTokenProvider,
};
pub use defaults::{default_chat_backends, default_chat_model};
pub use embedding_clients::{create_embedding_client, EmbeddingClient};
pub use rerank_clients::{create_rerank_client, RerankClient};
pub use settings::{
    BackendConfig, EndpointBinding, EndpointConfig, LlmSettings, ModelConfig, RateLimitConfig,
    ResolvedModelConfig, ServerConfig,
};
pub use types::{
    BackendType, ChatRequest, ChatRequestOptions, ChatResponse, ChatStreamDelta, ChatTool,
    ChatUsage, EmbeddingData, EmbeddingResponse, Message, MessageContent, MessageRole,
    RerankResponse, RerankResult, ToolCall, VvLlmError,
};
