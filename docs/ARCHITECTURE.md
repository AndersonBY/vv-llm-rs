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
    sync_contract.py
    test_sync_contract.py
  crates/vv-llm/
    Cargo.toml
    contract/v1.0.0/       # locked language-neutral schemas, fixtures, and catalog
    src/
      lib.rs
      contract.rs
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
- `ContractMetadata`, `contract_metadata`, and embedded manifest/consumer-lock accessors.
- Provider-neutral data types from `types.rs`.

Public request and response structs should stay provider-neutral. Provider-specific JSON, SDK request builders, stream event shapes, and auth mechanics belong in adapter modules.

## Core Types

`types.rs` defines the normalized contract used across providers:

- `BackendType` identifies chat backends known by settings resolution and factory routing.
- `Message`, `MessageRole`, and `MessageContent` represent text, image URL, assistant tool-call turns, and tool-result turns.
- `ChatRequest` carries model, messages, options (including explicit thinking configuration), tools, and tool choice.
- `ChatRequest::from_contract`/`to_contract` encode the canonical contract shape, require a non-empty model, and preserve string or object `tool_choice`, string-or-array stop values, `x_*` extensions, and image metadata. Direct serde accepts the runtime default model value; adapters own provider wire conversion.
- `ThinkingPreference` maps typed default/enabled/budgeted/disabled/provider-defined intent into the existing optional provider JSON field, preserving source and wire compatibility.
- `ModelCapabilities` describes provider-neutral tools, structured-output, modality, streaming, and thinking support. `ModelConfig` also carries the optional `max_image_dimension` input limit, and `ModelConfig::capabilities()` reads explicit catalog metadata and derives legacy fields when metadata is absent.
- `ChatResponse` carries normalized content, tool calls, and usage.
- `CompletionResult<Response>` adds execution metadata without changing the legacy `ChatResponse` struct.
- `ResponseMetadata` records provider, model, response/request ids, attempts, latency, fallback index, and extensible attributes.
- `ChatStreamDelta` carries streaming text, reasoning text, tool-call deltas, usage, optional raw content, and a `done` marker.
- `ChatUsage` preserves the legacy prompt/completion/total counters plus optional input/output, cache read/cache creation, and raw provider usage. Missing optional counters remain distinct from explicitly reported zeroes.
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

## Execution Pipeline

Provider adapters only translate protocols. Cross-provider execution behavior is
layered above them:

1. `ChatRequest` is normalized and checked against `ModelCapabilities`.
2. `ChatMiddlewareV1` hooks run in an explicitly versioned chain.
3. `MiddlewareChatClient` invokes the provider client through `RetryPolicy`.
4. `CompletionResult` can expose execution metadata without changing legacy responses.
5. `FallbackChatClient` may try the next explicitly registered route only for allowed error kinds.

`ProviderRegistry` never discovers providers automatically. A registration owns a
client factory and declared capabilities. `FallbackRoute` supplies an ordered
provider/model pair. Unsupported tools, structured output, streaming, thinking,
or image input disqualify a route before its factory is called.

Streaming retry and fallback stop at the first visible chunk. Establishment errors
and an error received as the first stream item may select another route; errors
after output begins are propagated without replay.

`ChatMiddlewareV1::on_stream_start` has a separate boundary: it runs when the
provider stream is established, before the first item is yielded, including any
metadata-only prelude. `create_with_metadata` returns completion metadata and
explicitly rejects stream requests because its `CompletionResult<ChatResponse>`
type cannot represent a boxed stream safely.

`ChatClient::create` is intentionally the non-streaming completion entry point.
Call `ChatClient::create_stream` for a boxed stream; the `stream` option does not
change the return type of `create`.

## Python/Rust Capability Matrix

The Rust crate targets the shared public contract, but it is not a feature-for-
feature replacement for the Python package. Keep these boundaries explicit:

| Capability | Python `vv-llm` | Rust `vv-llm-rs` | Compatibility boundary |
|---|---|---|---|
| Middleware | Sync and async clients; request, response, and error hooks | Async `MiddlewareChatClient`; request, response, stream-start, and error hooks | Hook names and async execution are Rust-specific; both expose versioned middleware intent |
| Retry | Sync and async classified retry with backoff and `Retry-After` | Async `RetryPolicy`/`execute_with_retry` with backoff, jitter, deadline, `Retry-After`, and configurable retryable kinds | No synchronous Rust client layer |
| Metadata | `CompletionResult` plus provider/model/attempt/latency metadata | `CompletionResult` plus `ResponseMetadata`, fallback index, ids, attempts, and latency | Field availability depends on adapter response data |
| Registry/fallback | Sync and async explicit `ProviderRegistry` and capability-aware routes | Explicit async `ProviderRegistry`/`FallbackChatClient`, capability filtering, and first-visible-chunk boundary | Rust does not discover providers or provide a sync facade |
| Scripted client | Sync and async scripted completion/error/stream doubles | Public async `ScriptedChatClient` with deterministic completion/error/stream steps | Rust double does not emulate provider wire protocols |
| Chat providers | Native sync/async adapters for 16 named backends, plus Azure/Vertex/Bedrock deployments | Async clients; named OpenAI-compatible backends share one adapter, with native Anthropic, Bedrock, and Vertex paths | Provider-specific Python quirks are not automatically present in generic Rust routing; Azure metadata parses but has no dedicated Rust wire adapter |
| Embedding/rerank | Sync and async OpenAI, Cohere, Jina, Voyage, SiliconFlow, Local, and custom mapping paths | Async OpenAI-compatible embeddings and custom JSON HTTP rerank | No Rust sync facade or dedicated Cohere/Voyage embedding protocol adapter |
| Rate limiting | Active memory, Redis, and DiskCache RPM/TPM limiters (optional extras) | Parses endpoint/global RPM/TPM settings but does not enforce a local/distributed limiter | Rust retry handling of 429/`Retry-After` is not rate-limit enforcement |
| Token counting | Local model tokenizers, provider/token-server fallback, and optional FastAPI token server | Local `tiktoken-rs`, configured token-server/provider-tokenizer fallback, no bundled server executable | Rust consumes a token server; it does not ship the Python FastAPI server |
| Settings compatibility | V1 upgrade and V2 `backends`/retrieval fields | V2 `backends`/retrieval fields, string/object bindings, and transport metadata; top-level V1 is intentionally not upgraded | Use the versioned contract and do not assume Python's V1 migration behavior |
| Contract artifacts | Vendored `vv-llm-contract` 1.0.0 schemas, fixtures, catalog, and lock | Vendored same release with lock SHA pin and compile-time catalog/fixture use | JSON wire semantics are shared; runtime orchestration remains language-specific |

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
- Providers that split usage across stream events should emit a cumulative usage snapshot so later output-only events do not erase earlier input or cache observations.
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
- `count_tokens_with_settings` matches the Python utility behavior by preferring a configured token server, then provider tokenizer endpoints for providers that expose one, then local fallback.
- `count_message_tokens`, `calculate_image_tokens`, and `cutoff_messages` mirror the Python chat request sizing helpers.
- `parse_retry_after` accepts provider `retry-after-ms` plus numeric or HTTP-date `Retry-After`; `RetryPolicy` and `execute_with_retry` consume the classified hint with exponential backoff, jitter, and an optional total deadline.

## Test Doubles

`ScriptedChatClient` consumes explicit completion and stream steps and records
normalized requests. It is a public deterministic test double for middleware,
retry, and fallback conformance; it does not emulate provider wire protocols.
