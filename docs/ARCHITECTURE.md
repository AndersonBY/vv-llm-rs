# Architecture

`vv-llm-rs` is a Rust workspace with one crate, `vv-llm`. The crate exposes a provider-neutral API for chat, streaming, embeddings, rerank, settings resolution, and utility helpers.

## Workspace Layout

```text
vv-llm-rs/
  Cargo.toml
  AGENTS.md
  docs/
  scripts/
    run_live_tests.sh
  crates/vv-llm/
    Cargo.toml
    src/
      lib.rs
      types.rs
      settings.rs
      chat_clients/
      embedding_clients/
      rerank_clients/
      utilities/
    tests/
      fixtures/
      chat.rs
      live_tests.rs
      settings.rs
      utilities.rs
```

## Public API Boundary

`src/lib.rs` re-exports the supported public surface:

- `ChatClient`, `create_chat_client`, `create_chat_client_from_resolved`, and `ChatStream`.
- `EmbeddingClient`, `create_embedding_client`.
- `RerankClient`, `create_rerank_client`.
- `LlmSettings`, `EndpointConfig`, `ModelConfig`, `EndpointBinding`, `ResolvedModelConfig`.
- Provider-neutral data types from `types.rs`.

Public request and response structs should stay provider-neutral. Provider-specific JSON, SDK request builders, stream event shapes, and auth mechanics belong in adapter modules.

## Core Types

`types.rs` defines the normalized contract used across providers:

- `BackendType` identifies chat backends known by settings resolution and factory routing.
- `Message`, `MessageRole`, and `MessageContent` represent text, image URL, assistant tool-call turns, and tool-result turns.
- `ChatRequest` carries model, messages, options, tools, and tool choice.
- `ChatResponse` carries normalized content, tool calls, and usage.
- `ChatStreamDelta` carries streaming text, reasoning text, tool-call deltas, usage, optional raw content, and a `done` marker.
- `EmbeddingResponse` and `RerankResponse` keep retrieval clients independent from provider payloads.
- `VvLlmError` separates configuration, model, endpoint, serialization, HTTP, and provider failures.

When adding fields, prefer optional fields with serde defaults unless all existing callers can construct the new field without churn.

## Settings Resolution

`settings.rs` parses the shared settings shape:

- `VERSION`
- `endpoints`
- `backends`
- `embedding_backends`
- `rerank_backends`

Each backend contains `models`. Each model has an `id` and `endpoints`. Endpoint bindings support both forms:

```json
"endpoints": ["openai-default"]
```

```json
"endpoints": [
  {
    "endpoint_id": "openai-default",
    "model_id": "provider-model-id",
    "enabled": true
  }
]
```

Resolution rules:

- A model can be resolved by its map key or by `ModelConfig.id`.
- The first enabled endpoint binding is selected.
- Object bindings can override the provider model id through `model_id`.
- Missing backends or models return `VvLlmError::ModelNotFound`.
- Missing endpoints return `VvLlmError::EndpointNotFound`.
- Transport metadata such as `endpoint_type`, `region`, `is_bedrock`, `is_vertex`, and `credentials` must survive parsing.

## Client Factories

`create_chat_client` accepts direct backend, model, base URL, and API key inputs.

`create_chat_client_from_resolved` accepts `ResolvedModelConfig` and routes by backend and endpoint metadata:

- `backend == anthropic` plus `endpoint_type == "anthropic_bedrock"` or `is_bedrock == true` routes to `AnthropicBedrockChatClient`.
- `endpoint_type == "openai_vertex"` routes to `VertexOpenAiChatClient`.
- Plain `backend == anthropic` routes to `AnthropicChatClient`.
- Other known chat backends route to `OpenAiCompatibleChatClient`.

Keep this routing explicit. Do not infer special transports from URL substrings when settings metadata is available.

## Adapter Boundaries

Provider adapters own these translations:

- `ChatRequest` to provider SDK request.
- Provider response to `ChatResponse`.
- Provider stream event to `ChatStreamDelta`.
- Provider tool schema representation.
- Provider multimodal representation.
- Provider auth transport details.

Adapters should expose small request-shape helpers for tests when useful, but provider implementation detail should not leak into public crate types.

## Streaming Model

`ChatClient::create_stream` returns `ChatStream`, a boxed stream of `Result<ChatStreamDelta, VvLlmError>`.

Normalization expectations:

- Text deltas append through `ChatStreamDelta.content`.
- Reasoning deltas append through `ChatStreamDelta.reasoning_content`.
- Tool-call deltas use `ChatStreamDelta.tool_calls`.
- Usage events use `ChatStreamDelta.usage`.
- Provider metadata that callers may need for debugging can go in `raw_content`.
- The terminal provider event should emit `done: true` when the provider gives a clear completion signal.

Avoid making downstream callers interpret provider event enums or provider-specific JSON to use normal chat streaming.

## Retrieval Clients

Embedding clients are currently OpenAI-compatible. Rerank clients use a small custom JSON HTTP adapter with an explicit `RerankMapping`.

Retrieval client changes should keep chat types independent. Do not add chat-only concepts to embedding or rerank traits.

## Utilities

Utilities are deliberately small:

- `normalize_text_messages` merges adjacent text-only messages with the same role and metadata.
- `count_tokens` uses `tiktoken-rs` for supported OpenAI-family encodings and falls back deterministically for unknown models.
- `RetryPolicy` stores retry metadata for callers that own retry loops.

Do not put provider transport behavior in utilities.
