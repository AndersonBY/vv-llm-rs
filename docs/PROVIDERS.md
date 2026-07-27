# Provider Adapters

This document records how each provider path maps the public `vv-llm` API to SDK or HTTP behavior.

Retry, middleware, registry lookup, fallback selection, and cross-provider
capability filtering do not belong in provider adapters. Those behaviors are
implemented by the provider-neutral execution layer.

## OpenAI-Compatible Chat

Module: `crates/vv-llm/src/chat_clients/openai_compatible.rs`

Used for OpenAI-style `/v1/chat/completions` APIs including OpenAI, DeepSeek, Qwen, Gemini OpenAI-compatible endpoints, ZhiPuAI, Groq, Mistral, Moonshot, MiniMax, Yi, Baichuan, StepFun, xAI, Ernie, and local OpenAI-compatible servers.

Implementation notes:

- Uses `async-openai`.
- Maps `MessageRole::System`, `User`, `Assistant`, and `Tool` to OpenAI chat messages.
- Maps text and image URL parts for multimodal user messages.
- Serializes empty and reasoning-only assistant messages without tool calls as `content: ""`; reasoning stays in `reasoning_content`, while tool-call-only messages continue to omit `content`.
- Maps `ChatTool` into function tools.
- Supports `tool_choice` values accepted by the adapter.
- Forwards `ChatRequestOptions::thinking` as the top-level `thinking` request field when explicitly set.
- `ThinkingPreference` is normalized into that existing field before the adapter runs, so typed and legacy callers share the same request path.
- Normalizes completion content, tool calls, and usage into `ChatResponse`.
- Normalizes stream content, tool-call chunks, usage chunks, and done state into `ChatStreamDelta`.
- `create_stream` always sends `stream: true` and defaults missing `stream_options` to `{"include_usage": true}` so opt-in providers return their final usage chunk. Explicit caller-provided `stream_options` are preserved; completion requests are unchanged.
- Captures the raw `usage` object before typed response deserialization for both completion and stream responses.
- Maps `prompt_tokens_details.cached_tokens`, `input_tokens_details.cached_tokens`, the official top-level `cached_tokens` field, and compatible top-level cache fields to `ChatUsage.cache_read_input_tokens`; cache creation/write variants map to `ChatUsage.cache_creation_input_tokens`.
- Generic clients preserve a completely omitted cache-read value as `None`. Only clients created by `create_chat_client(BackendType::Moonshot, ...)` normalize complete cache-read omission to `Some(0)`, for both completion and stream responses. Present `null` or malformed cache-read fields remain `None`, and `raw_usage` is never modified with synthetic fields.
- Extracts tagged reasoning from streamed content for `<think>...</think>` and Gemini `<thought>...</thought>` style tags.

Keep OpenAI-compatible behavior generic unless a provider requires a settings-level transport distinction.

## Anthropic Direct

Module: `crates/vv-llm/src/chat_clients/anthropic.rs`

Used for direct Anthropic Messages API.

Implementation notes:

- Uses the `anthropic` Rust SDK.
- Extracts system messages into the Anthropic system prompt field.
- Maps text and image data URL content into Anthropic message content.
- Forwards `ChatRequestOptions::thinking` through the JSON request path when explicitly set.
- `ThinkingPreference` is normalized into that existing field before the adapter runs.
- Maps non-streaming text responses, input/output usage, cache reads, cache creation, and raw usage into `ChatResponse`.
- Accumulates direct JSON stream usage across `message_start` and `message_delta`, retaining initial input/cache values with the final output count.
- Direct streaming currently exposes normalized text deltas. The upstream crate does not expose all tool/thinking stream request fields, so full tool and reasoning streaming is handled through the Bedrock path where available.

Do not bypass the SDK with raw HTTP unless a required feature cannot be represented with the SDK and the gap is documented in tests and docs.

## Anthropic Bedrock

Module: `crates/vv-llm/src/chat_clients/anthropic.rs`

Selected when `ResolvedModelConfig.endpoint.endpoint_type == "anthropic_bedrock"` or `is_bedrock == true`.

Implementation notes:

- Uses `aws-sdk-bedrockruntime` Converse and ConverseStream.
- Requires `region` and AWS credentials in endpoint `credentials`.
- Maps text, data URL images, assistant tool-use turns, and tool-result turns.
- Maps `ChatTool` to Bedrock tool configuration.
- Normalizes text, tool calls, reasoning deltas, usage, and done events from Bedrock streams. Bedrock `cache_write_input_tokens` is exposed as `ChatUsage.cache_creation_input_tokens` while the provider key remains in `raw_usage`.
- Converts Bedrock `Document` values to JSON strings for normalized tool arguments.

For Bedrock image support, use data URLs in `MessageContent::ImageUrl`. The adapter decodes the data URL and maps the media type to the Bedrock image format.

## Vertex OpenAI-Compatible

Module: `crates/vv-llm/src/chat_clients/vertex.rs`

Selected when `ResolvedModelConfig.endpoint.endpoint_type == "openai_vertex"`.

Implementation notes:

- Wraps `OpenAiCompatibleChatClient`.
- Exchanges Google credentials for a bearer access token before building the inner OpenAI-compatible client.
- Supports user refresh-token credentials: `refresh_token`, `client_id`, and `client_secret`.
- Supports service-account credentials: `private_key` and `client_email`.
- Caches fresh access tokens in process and refreshes when needed.
- Delegates completion and streaming normalization to the OpenAI-compatible adapter.

Do not store refreshed access tokens in settings files. The cache is in memory only.

## Embeddings

Module: `crates/vv-llm/src/embedding_clients/`

Implementation notes:

- Uses `async-openai` for OpenAI-compatible embeddings.
- `create_embedding_client` currently returns `OpenAiCompatibleEmbeddingClient`.
- `EmbeddingResponse` normalizes model id and embedding vectors.

## Rerank

Module: `crates/vv-llm/src/rerank_clients/`

Implementation notes:

- Uses `CustomJsonHttpRerankClient`.
- `RerankMapping` currently carries HTTP method and path. The implemented request uses POST with bearer auth and a JSON body.
- The default SiliconFlow mapping uses `/rerank`.
- Normalizes `results[].index` and `results[].relevance_score`.

If adding a new rerank protocol, prefer a separate adapter when the response semantics differ instead of growing one large dynamic mapping layer.

## Provider Change Checklist

- Keep public types provider-neutral.
- Add request-shape tests for serialization and role/content/tool mapping.
- Add stream-normalization tests for content, reasoning, tool-call chunks, usage, and done markers when streaming changes.
- Add or update live tests for the real transport if credentials are available.
- Update `README.md`, `README_ZH.md`, and this docs directory when externally visible behavior changes.
