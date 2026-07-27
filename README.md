# vv-llm-rs

[中文文档](./README_ZH.md)

Universal LLM client layer for Rust. One typed API for chat, streaming, embeddings, rerank, multimodal messages, tool calls, and vendor endpoint resolution.

```toml
[dependencies]
vv-llm = "0.4.5"
```

The crate is published on crates.io as `vv-llm`; Rust code imports it as `vv_llm`. For local development in this repository, use `vv-llm = { path = "crates/vv-llm" }`.

## Supported Backends

OpenAI-compatible chat works with OpenAI, DeepSeek, Qwen, Gemini OpenAI-compatible endpoints, ZhiPuAI, Groq, Mistral, Moonshot, MiniMax, Yi, Baichuan, StepFun, xAI, Ernie, local OpenAI-compatible servers, and similar `/v1/chat/completions` APIs.

Native transports are also available for:

- Anthropic Messages API
- Anthropic on AWS Bedrock through Bedrock Converse
- OpenAI-compatible models on Google Vertex AI with automatic Google access-token exchange
- OpenAI-compatible embedding APIs
- JSON HTTP rerank APIs such as SiliconFlow rerank

## Quick Start

### Direct Client

```rust
use vv_llm::{create_chat_client, BackendType, ChatRequest, Message, MessageRole};

#[tokio::main]
async fn main() -> Result<(), vv_llm::VvLlmError> {
    let client = create_chat_client(
        BackendType::OpenAI,
        "gpt-4o",
        "https://api.openai.com/v1",
        "sk-...",
    );

    let mut request = ChatRequest::new(
        "gpt-4o",
        vec![Message::text(
            MessageRole::User,
            "Explain RAG in one sentence.",
        )],
    );
    request.options.max_tokens = Some(128);

    let response = client.create_completion(request).await?;

    println!("{}", response.content);
    Ok(())
}
```

### Settings-Based Client

Use `LlmSettings` when models and endpoints should come from a shared configuration file.

```rust
use vv_llm::{
    create_chat_client_from_resolved, BackendType, ChatRequest, LlmSettings, Message, MessageRole,
};

#[tokio::main]
async fn main() -> Result<(), vv_llm::VvLlmError> {
    let settings = LlmSettings::from_json_file("llm_settings.json")?;
    let resolved = settings.resolve_chat_model(BackendType::OpenAI, "gpt-4o")?;
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved)?;

    let response = client
        .create_completion(ChatRequest::new(
            model,
            vec![Message::text(MessageRole::User, "hello")],
        ))
        .await?;

    println!("{}", response.content);
    Ok(())
}
```

Minimal settings shape:

```json
{
  "VERSION": "2",
  "endpoints": [
    {
      "id": "openai-default",
      "api_base": "https://api.openai.com/v1",
      "api_key": "sk-..."
    }
  ],
  "backends": {
    "openai": {
      "models": {
        "gpt-4o": {
          "id": "gpt-4o",
          "endpoints": ["openai-default"],
          "context_length": 128000,
          "max_output_tokens": 16384,
          "function_call_available": true,
          "response_format_available": true
        }
      }
    }
  },
  "embedding_backends": {},
  "rerank_backends": {}
}
```

Endpoint bindings may be strings or objects. Object bindings can override the provider model id and can be disabled:

```json
{
  "endpoint_id": "openai-default",
  "model_id": "provider-model-id",
  "enabled": true
}
```

## Streaming

`create_stream` returns normalized `ChatStreamDelta` values. Text deltas, tool-call deltas, usage, completion state, and supported reasoning deltas use the same Rust type across providers.

```rust
use futures_util::StreamExt;
use vv_llm::{ChatRequest, ChatRequestOptions, Message, MessageRole};

let mut stream = client
    .create_stream({
        let mut request = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(MessageRole::User, "Write a haiku.")],
        );
        request.options.stream = Some(true);
        request
    })
    .await?;

while let Some(delta) = stream.next().await {
    let delta = delta?;
    if !delta.content.is_empty() {
        print!("{}", delta.content);
    }
}
```

OpenAI-compatible streams normalize content, tool calls, usage chunks, and tagged reasoning such as `<think>...</think>` or Gemini `<thought>...</thought>`. Anthropic Bedrock streams normalize text, tool use, reasoning, and usage events. The direct Anthropic SDK path currently exposes text streaming only because the upstream Rust crate does not expose tool/thinking stream request fields.

For OpenAI-compatible clients, `create_stream` always sends `stream: true`. When `ChatRequestOptions::stream_options` is not provided, it also sends `{"include_usage": true}` so providers that require an explicit opt-in can return the final usage chunk. Caller-provided `stream_options`, including `{"include_usage": false}`, are preserved unchanged. This default does not affect non-streaming requests or other provider adapters.

## Usage Accounting

`ChatUsage` keeps the legacy `prompt_tokens`, `completion_tokens`, and `total_tokens` fields and also exposes provider-neutral `input_tokens`, `output_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens`. All fields are optional: a missing cache value is `None`, while an explicitly reported zero is `Some(0)`.

`raw_usage` preserves the provider usage object for diagnostics and forward compatibility. OpenAI-compatible cache reads are normalized from prompt/input token details, the official top-level `cached_tokens` field, or compatible top-level cache fields. Anthropic cache reads and cache creation values are mapped directly; Bedrock `cache_write_input_tokens` maps to `cache_creation_input_tokens`. Invalid string, fractional, negative, or overflowing token values remain available in `raw_usage` but are not coerced into normalized counts.

Generic OpenAI-compatible clients keep an omitted cache-read value as `None`. Clients created through `create_chat_client(BackendType::Moonshot, ...)` apply Moonshot's provider contract for both completions and streams: when every recognized cache-read field is absent, `cache_read_input_tokens` is normalized to `Some(0)`. An explicitly present `null` or invalid cache-read value remains `None`. This policy never inserts synthetic fields into `raw_usage`.

## Tool Calls

```rust
use vv_llm::{ChatRequest, ChatTool, Message, MessageRole};

let mut request = ChatRequest::new(
    "deepseek-chat",
    vec![Message::text(
        MessageRole::User,
        "Use the weather tool for New York.",
    )],
);
request.tools = vec![ChatTool::function(
        "get_current_weather",
        "Get the current weather in a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    )];
request.tool_choice = Some("required".to_string());

let response = client.create_completion(request).await?;
for call in response.tool_calls {
    println!("{} {}", call.name, call.arguments);
}
```

Tool-result turns use `MessageRole::Tool` with `tool_call_id`, and assistant tool-call turns use `Message.tool_calls`.

## Provider Extensions

OpenAI-compatible providers sometimes expose extra request and response fields for
reasoning traces, thinking controls, or vendor-specific tool metadata. `vv-llm`
keeps these in typed, provider-neutral fields so callers do not have to
hand-roll protocol conversion:

- `ChatRequest.extra_body` merges object fields into the root request JSON.
- `ChatRequestOptions::thinking` forwards an explicit enabled, disabled, or adaptive thinking configuration to supported provider adapters.
- `Message.reasoning_content` preserves assistant reasoning content on request messages.
- `MessageContent::Text.cache_control` and `ChatTool.cache_control` preserve Anthropic prompt-cache breakpoints.
- `ToolCall.extra_content` preserves vendor tool-call metadata such as Google thought signatures.
- `ChatResponse.reasoning_content` and streamed `ChatStreamDelta.reasoning_content` expose supported reasoning output.

For OpenAI-compatible assistant history, messages without tool calls always send a
`content` key. Empty and reasoning-only messages use `content: ""`, while
`reasoning_content` remains separate and is never promoted to visible content.
Tool-call-only assistant messages keep the protocol's valid omitted-content shape.

The OpenAI-compatible adapter keeps typed request construction, uses
`async-openai` BYOT response decoding so provider usage extensions survive, and
normalizes raw JSON responses back into the public `vv-llm` types.

## Multimodal Input

Text and image parts can be mixed in a user message. Image URLs should be data URLs for providers that require inline base64 payloads.

```rust
use vv_llm::{Message, MessageContent, MessageRole};

let message = Message {
    role: MessageRole::User,
    content: vec![
        MessageContent::Text {
            text: "What is in this image?".to_string(),
        },
        MessageContent::ImageUrl {
            url: "data:image/png;base64,...".to_string(),
        },
    ],
    name: None,
    tool_call_id: None,
    tool_calls: Vec::new(),
    reasoning_content: None,
};
```

## Embeddings And Rerank

```rust
use vv_llm::{
    create_embedding_client,
    rerank_clients::{CustomJsonHttpRerankClient, RerankMapping},
    RerankClient,
};

let embedding_client = create_embedding_client(
    "siliconflow",
    "Qwen/Qwen3-Embedding-4B",
    "https://api.siliconflow.cn/v1",
    "sk-...",
);
let embeddings = embedding_client
    .create_embeddings(&["hello world", "vector search"])
    .await?;
println!("{}", embeddings.data.len());

let rerank_client = CustomJsonHttpRerankClient::new(
    "BAAI/bge-reranker-v2-m3",
    "https://api.siliconflow.cn/v1",
    "sk-...",
    RerankMapping::default_siliconflow(),
);
let rerank = rerank_client
    .rerank("Apple", &["apple", "banana", "fruit"])
    .await?;
println!("{:?}", rerank.results);
```

## Vertex AI And Bedrock

Vertex OpenAI-compatible endpoints are configured with `endpoint_type: "openai_vertex"` and Google credentials. User refresh-token credentials and service-account credentials are supported.

```json
{
  "id": "gemini-vertex",
  "api_base": "https://aiplatform.googleapis.com/v1beta1/projects/PROJECT/locations/global/endpoints/openapi",
  "endpoint_type": "openai_vertex",
  "region": "global",
  "credentials": {
    "refresh_token": "...",
    "client_id": "...",
    "client_secret": "..."
  }
}
```

Anthropic Bedrock endpoints are configured with `endpoint_type: "anthropic_bedrock"`, AWS region, and AWS credentials.

```json
{
  "id": "anthropic-bedrock",
  "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
  "endpoint_type": "anthropic_bedrock",
  "region": "us-east-1",
  "credentials": {
    "access_key": "...",
    "secret_key": "..."
  }
}
```

## Features

- **Unified chat API** — one `ChatClient` trait for completions and streaming
- **Settings resolution** — load model catalogs, endpoint bindings, provider ids, and transport metadata from JSON
- **OpenAI-compatible adapters** — chat and embeddings through `async-openai`
- **Provider extensions** — typed reasoning content, request `extra_body`, and tool-call `extra_content`
- **Anthropic support** — direct Messages API plus Bedrock Converse transport
- **Streaming normalization** — provider stream events become `ChatStreamDelta`
- **Tool calling** — normalized function/tool definitions, assistant tool calls, and tool-result turns
- **Multimodal messages** — text and image parts for supported providers
- **Vertex authentication** — Google access-token exchange with in-process cache
- **Retrieval clients** — OpenAI-compatible embeddings and custom JSON rerank
- **Token counting** — local tiktoken fallback plus settings-aware token server/provider tokenizer calls
- **Typed errors** — configuration, provider, HTTP, serialization, model, and endpoint errors

## Utilities

```rust
use vv_llm::utilities::{
    count_message_tokens, count_tokens, count_tokens_with_settings, normalize_text_messages,
    RetryPolicy,
};
```

| Function | Description |
|---|---|
| `normalize_text_messages` | Merge adjacent same-role text messages without merging images or tool data |
| `count_tokens` | Count tokens with supported model tokenizers |
| `count_tokens_with_settings` | Prefer configured token server and provider tokenizer endpoints, then fall back locally |
| `count_message_tokens` | Count formatted text, image placeholders, and tools for chat requests |
| `RetryPolicy` | Small retry metadata helper for callers that manage retries externally |

## Project Structure

```text
vv-llm-rs/
  Cargo.toml
  crates/vv-llm/
    src/
      chat_clients/       # Chat clients, stream normalization, Vertex auth
      embedding_clients/  # OpenAI-compatible embedding client
      rerank_clients/     # Custom JSON HTTP rerank client
      settings.rs         # Settings parsing and model resolution
      types.rs            # Public request/response/error types
      utilities/          # Message normalization, token counting, retry metadata
    tests/
      fixtures/           # Sample settings and live-test assets
```

## Development

Run checks from the workspace root:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Live integration tests are ignored by default. Put real credentials in `crates/vv-llm/tests/fixtures/dev_settings.json`, or set `VV_LLM_SETTINGS_JSON`, then run:

```bash
VV_LLM_RUN_LIVE_TESTS=1 ./scripts/run_live_tests.sh
```

Engineering documentation lives in [`docs/`](./docs/README.md). Start there for architecture notes, provider adapter behavior, live-test policy, security rules, and maintenance workflows.

Releases are published to crates.io by the tag workflow documented in [`docs/RELEASE.md`](./docs/RELEASE.md).

## License

MIT
