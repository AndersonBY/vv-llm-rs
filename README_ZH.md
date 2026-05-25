# vv-llm-rs

[English](./README.md)

VectorVein `vv-llm` 接口层的 Rust 工作空间。当前版本对齐 Python 包的主要公开表面，包括 settings、chat、embedding、rerank 和 utility helpers。

## 当前范围

- 支持 Python v2 风格 settings 加载和模型 endpoint 解析。
- OpenAI-compatible chat / embedding adapter，底层使用 `async-openai`。
- Anthropic chat adapter 边界，底层使用 `anthropic` Rust SDK。
- 自定义 JSON HTTP rerank client，对齐 Python `vv-llm` 中 SiliconFlow 风格的 mapping。
- 共享 message、response、usage 和 error 类型。
- 文本消息归一化、retry policy 元数据、fallback token count 等工具函数。

Provider 细节级多模态格式化、完整 streaming 归一化、Bedrock/Vertex 鉴权、精确 tokenizer 对齐和需要真实 API key 的 live tests 会在后续 parity 工作中补齐。

## 目录结构

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

包名是 `vv-llm`，Rust 代码中以 `vv_llm` 导入。

## 示例

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

## 校验

在 `vv-llm-rs/` 下运行：

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## License

MIT

