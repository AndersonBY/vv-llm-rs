# vv-llm-rs

[中文文档](./README_ZH.md)

Rust workspace for the VectorVein `vv-llm` interface layer. It mirrors the Python package's main public surface for settings, chat, embeddings, rerank, and utility helpers.

## Current Scope

- Python v2-style settings loading and model endpoint resolution.
- OpenAI-compatible chat and embedding adapters backed by `async-openai`.
- Anthropic chat adapter boundary backed by the `anthropic` Rust SDK.
- Custom JSON HTTP rerank client matching the SiliconFlow-style mapping used by Python `vv-llm`.
- Shared message, response, usage, and error types.
- Utility helpers for text message normalization, retry policy metadata, and fallback token counting.

Provider-specific multimodal formatting, complete streaming normalization, Bedrock/Vertex authentication, exact tokenizer parity, and live API-key tests are left for follow-up parity work.

## Layout

```text
vv-llm-rs/
  Cargo.toml
  crates/vv-llm/
    src/
      chat_clients/
      embedding_clients/
      rerank_clients/
      settings.rs
      types.rs
      utilities/
    tests/
```

The package is named `vv-llm`; Rust callers import it as `vv_llm`.

## Example

```rust
use vv_llm::{BackendType, ChatRequest, Message, MessageRole, create_chat_client};

let client = create_chat_client(
    BackendType::OpenAI,
    "gpt-4o",
    "https://api.openai.com/v1",
    "sk-...",
);

let request = ChatRequest {
    model: "gpt-4o".to_string(),
    messages: vec![Message::text(MessageRole::User, "hello")],
    options: Default::default(),
};
```

## Verification

Run from `vv-llm-rs/`:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## License

MIT

