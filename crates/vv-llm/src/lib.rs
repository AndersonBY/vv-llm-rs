//! Typed Rust clients for chat, streaming, embeddings, rerank, and LLM endpoint resolution.

pub mod chat_clients;
pub mod contract;
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
pub use contract::{
    contract_checksums, contract_consumer_lock_json, contract_manifest_json, contract_metadata,
    ContractMetadata, CONTRACT_CATALOG_REVISION, CONTRACT_CHECKSUMS, CONTRACT_CONSUMER_LOCK_JSON,
    CONTRACT_CONSUMER_LOCK_SHA256, CONTRACT_FIXTURE_VERSION, CONTRACT_MANIFEST_JSON,
    CONTRACT_METADATA, CONTRACT_SCHEMA_VERSION, CONTRACT_VERSION,
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
    JsonExtensions, Message, MessageContent, MessageRole, Modality, ModelCapabilities,
    RerankResponse, RerankResult, ResponseMetadata, StructuredOutputCapability, ThinkingCapability,
    ThinkingPreference, ToolCall, ToolChoice, VvLlmError,
};
pub use utilities::{execute_with_retry, parse_retry_after, RetryPolicy};
