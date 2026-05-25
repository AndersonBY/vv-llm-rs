# vv-llm-rs

[中文文档](./README_ZH.md)

Universal LLM client layer for Rust. One typed API for chat, streaming, embeddings, rerank, multimodal messages, tool calls, and vendor endpoint resolution.

```toml
[dependencies]
vv-llm = "0.1.0"
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
use vv_llm::{
    create_chat_client, BackendType, ChatRequest, ChatRequestOptions, Message, MessageRole,
};

#[tokio::main]
async fn main() -> Result<(), vv_llm::VvLlmError> {
    let client = create_chat_client(
        BackendType::OpenAI,
        "gpt-4o",
        "https://api.openai.com/v1",
        "sk-...",
    );

    let response = client
        .create_completion(ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![Message::text(
                MessageRole::User,
                "Explain RAG in one sentence.",
            )],
            options: ChatRequestOptions {
                max_tokens: Some(128),
                ..Default::default()
            },
            tools: Vec::new(),
            tool_choice: None,
        })
        .await?;

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
    .create_stream(ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::text(MessageRole::User, "Write a haiku.")],
        options: ChatRequestOptions {
            stream: Some(true),
            ..Default::default()
        },
        tools: Vec::new(),
        tool_choice: None,
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

## Tool Calls

```rust
use vv_llm::{ChatRequest, ChatTool, Message, MessageRole};

let request = ChatRequest {
    model: "deepseek-chat".to_string(),
    messages: vec![Message::text(
        MessageRole::User,
        "Use the weather tool for New York.",
    )],
    options: Default::default(),
    tools: vec![ChatTool::function(
        "get_current_weather",
        "Get the current weather in a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    )],
    tool_choice: Some("required".to_string()),
};

let response = client.create_completion(request).await?;
for call in response.tool_calls {
    println!("{} {}", call.name, call.arguments);
}
```

Tool-result turns use `MessageRole::Tool` with `tool_call_id`, and assistant tool-call turns use `Message.tool_calls`.

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
- **Anthropic support** — direct Messages API plus Bedrock Converse transport
- **Streaming normalization** — provider stream events become `ChatStreamDelta`
- **Tool calling** — normalized function/tool definitions, assistant tool calls, and tool-result turns
- **Multimodal messages** — text and image parts for supported providers
- **Vertex authentication** — Google access-token exchange with in-process cache
- **Retrieval clients** — OpenAI-compatible embeddings and custom JSON rerank
- **Token counting** — tiktoken-based counts for GPT-3.5, GPT-4o, o1, and o3 families with deterministic fallback
- **Typed errors** — configuration, provider, HTTP, serialization, model, and endpoint errors

## Utilities

```rust
use vv_llm::utilities::{count_tokens, count_tokens_fallback, normalize_text_messages, RetryPolicy};
```

| Function | Description |
|---|---|
| `normalize_text_messages` | Merge adjacent same-role text messages without merging images or tool data |
| `count_tokens` | Count tokens with supported model tokenizers |
| `count_tokens_fallback` | Deterministic whitespace fallback counter |
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
