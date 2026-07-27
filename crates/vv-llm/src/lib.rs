//! Typed Rust clients for chat, streaming, embeddings, rerank, and LLM endpoint resolution.

pub mod chat_clients;
pub mod defaults;
pub mod embedding_clients;
pub mod middleware;
pub mod registry;
pub mod rerank_clients;
pub mod settings;
pub mod testing;
pub mod types;
pub mod utilities;

pub use chat_clients::{
    create_chat_client, create_chat_client_from_resolved, ChatClient, ChatStream,
    GoogleAccessTokenProvider,
};
pub use defaults::{default_chat_backends, default_chat_model};
pub use embedding_clients::{create_embedding_client, EmbeddingClient};
pub use middleware::{ChatMiddlewareV1, MiddlewareChatClient, MiddlewareContext};
pub use registry::{FallbackChatClient, FallbackRoute, ProviderRegistration, ProviderRegistry};
pub use rerank_clients::{create_rerank_client, RerankClient};
pub use settings::{
    BackendConfig, EndpointBinding, EndpointConfig, LlmSettings, ModelConfig, RateLimitConfig,
    ResolvedModelConfig, ServerConfig,
};
pub use testing::{ScriptedChatClient, ScriptedStep, ScriptedStream};
pub use types::{
    BackendType, ChatRequest, ChatRequestOptions, ChatResponse, ChatStreamDelta, ChatTool,
    ChatUsage, CompletionResult, EmbeddingData, EmbeddingResponse, ErrorDetails, ErrorKind,
    Message, MessageContent, MessageRole, Modality, ModelCapabilities, RerankResponse,
    RerankResult, ResponseMetadata, StructuredOutputCapability, ThinkingCapability,
    ThinkingPreference, ToolCall, VvLlmError,
};
pub use utilities::{execute_with_retry, RetryPolicy};
