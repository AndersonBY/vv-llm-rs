# vv-llm-rs

[English README](./README.md)

面向 Rust 的统一 LLM 客户端层。一套类型化 API，覆盖 chat、streaming、embedding、rerank、多模态消息、工具调用和多供应商 endpoint 解析。

```toml
[dependencies]
vv-llm = "0.2.0"
```

包已经发布到官方 crates.io，名称是 `vv-llm`，Rust 代码中以 `vv_llm` 导入。本仓库本地开发时可以使用 `vv-llm = { path = "crates/vv-llm" }`。

## 支持的后端

OpenAI-compatible chat 可用于 OpenAI、DeepSeek、Qwen、Gemini OpenAI-compatible endpoint、ZhiPuAI、Groq、Mistral、Moonshot、MiniMax、Yi、Baichuan、StepFun、xAI、Ernie、本地 OpenAI-compatible server，以及类似 `/v1/chat/completions` 的接口。

同时支持这些原生或专用传输：

- Anthropic Messages API
- AWS Bedrock 上的 Anthropic，通过 Bedrock Converse 调用
- Google Vertex AI 上的 OpenAI-compatible 模型，并自动换取 Google access token
- OpenAI-compatible embedding API
- SiliconFlow rerank 这类 JSON HTTP rerank API

## 快速开始

### 直接创建 Client

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
        vec![Message::text(MessageRole::User, "用一句话解释 RAG。")],
    );
    request.options.max_tokens = Some(128);

    let response = client.create_completion(request).await?;

    println!("{}", response.content);
    Ok(())
}
```

### 通过 Settings 创建 Client

如果模型和 endpoint 由统一配置管理，使用 `LlmSettings` 解析。

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

最小配置结构：

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

`endpoints` 绑定可以是字符串，也可以是对象。对象形式支持覆盖 provider model id，并可禁用：

```json
{
  "endpoint_id": "openai-default",
  "model_id": "provider-model-id",
  "enabled": true
}
```

## 流式调用

`create_stream` 返回统一的 `ChatStreamDelta`。文本 delta、工具调用 delta、usage、完成状态和支持的 reasoning delta 都使用同一个 Rust 类型。

```rust
use futures_util::StreamExt;
use vv_llm::{ChatRequest, ChatRequestOptions, Message, MessageRole};

let mut stream = client
    .create_stream({
        let mut request = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(MessageRole::User, "写一首四行诗。")],
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

OpenAI-compatible stream 会归一化文本、工具调用、usage chunk，以及 `<think>...</think>` 或 Gemini `<thought>...</thought>` 这类 tagged reasoning。Anthropic Bedrock stream 会归一化文本、工具调用、reasoning 和 usage 事件。直接 Anthropic SDK 路径目前只暴露文本 streaming，因为上游 Rust crate 没有暴露工具/思考流请求字段。

## 工具调用

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

工具结果轮次使用带 `tool_call_id` 的 `MessageRole::Tool`；assistant 发出的工具调用放在 `Message.tool_calls` 中。

## Provider 扩展字段

OpenAI-compatible provider 经常会暴露额外的请求 / 响应字段，用于 reasoning trace、thinking 控制或供应商专有工具元数据。`vv-llm` 把这些能力放在 provider-neutral 的类型化字段里，调用方不需要自己手写协议转换：

- `ChatRequest.extra_body` 会把对象字段合并到请求 JSON 根层。
- `Message.reasoning_content` 会保留 assistant 历史消息里的 reasoning 内容。
- `MessageContent::Text.cache_control` 和 `ChatTool.cache_control` 会保留 Anthropic prompt-cache 断点。
- `ToolCall.extra_content` 会保留供应商工具调用元数据，例如 Google thought signature。
- `ChatResponse.reasoning_content` 和流式 `ChatStreamDelta.reasoning_content` 会暴露支持的 reasoning 输出。

当这些扩展字段存在时，OpenAI-compatible adapter 会在内部使用 `async-openai` BYOT，并把原始 JSON 响应重新归一化成公开的 `vv-llm` 类型。

## 多模态输入

用户消息里可以混合文本和图片。对要求 inline base64 的供应商，图片 URL 使用 data URL。

```rust
use vv_llm::{Message, MessageContent, MessageRole};

let message = Message {
    role: MessageRole::User,
    content: vec![
        MessageContent::Text {
            text: "这张图片里有什么？".to_string(),
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

## Embedding 与 Rerank

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

## Vertex AI 与 Bedrock

Vertex OpenAI-compatible endpoint 使用 `endpoint_type: "openai_vertex"`，并配置 Google 凭据。支持 user refresh token 和 service account 两种凭据格式。

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

Anthropic Bedrock endpoint 使用 `endpoint_type: "anthropic_bedrock"`，并配置 AWS region 和 AWS 凭据。

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

## 核心特性

- **统一 Chat API** — 一个 `ChatClient` trait 覆盖 completion 和 streaming
- **配置解析** — 从 JSON 加载模型目录、endpoint 绑定、provider id 和传输元数据
- **OpenAI-compatible adapter** — 使用 `async-openai` 处理 chat 和 embedding
- **Provider 扩展字段** — 类型化 reasoning content、请求 `extra_body` 和 tool-call `extra_content`
- **Anthropic 支持** — 直接 Messages API，以及 Bedrock Converse transport
- **Streaming 归一化** — provider stream event 统一转成 `ChatStreamDelta`
- **工具调用** — 标准化 function/tool 定义、assistant tool call 和 tool-result 轮次
- **多模态消息** — 对支持的 provider 发送文本和图片消息块
- **Vertex 鉴权** — Google access token 换取和进程内缓存
- **检索客户端** — OpenAI-compatible embedding 与自定义 JSON rerank
- **Token 统计** — GPT-3.5、GPT-4o、o1、o3 系列使用 tiktoken，未知模型使用 deterministic fallback
- **类型化错误** — configuration、provider、HTTP、serialization、model、endpoint 等错误类型

## 工具函数

```rust
use vv_llm::utilities::{count_tokens, count_tokens_fallback, normalize_text_messages, RetryPolicy};
```

| 函数 | 说明 |
|---|---|
| `normalize_text_messages` | 合并相邻同角色文本消息，不合并图片或工具数据 |
| `count_tokens` | 使用支持的模型 tokenizer 统计 token |
| `count_tokens_fallback` | deterministic whitespace fallback 计数 |
| `RetryPolicy` | 给调用方使用的轻量 retry 元数据 helper |

## 目录结构

```text
vv-llm-rs/
  Cargo.toml
  crates/vv-llm/
    src/
      chat_clients/       # Chat client、stream 归一化、Vertex 鉴权
      embedding_clients/  # OpenAI-compatible embedding client
      rerank_clients/     # 自定义 JSON HTTP rerank client
      settings.rs         # 配置解析和模型 resolution
      types.rs            # 公开 request / response / error 类型
      utilities/          # 消息归一化、token 统计、retry 元数据
    tests/
      fixtures/           # 示例配置和 live test 资源
```

## 开发

在 workspace 根目录运行：

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

真实 API 集成测试默认 ignored。把真实凭据放到 `crates/vv-llm/tests/fixtures/dev_settings.json`，或设置 `VV_LLM_SETTINGS_JSON`，然后运行：

```bash
VV_LLM_RUN_LIVE_TESTS=1 ./scripts/run_live_tests.sh
```

工程文档放在 [`docs/`](./docs/README.md)。架构说明、provider adapter 行为、live test 规则、安全约束和维护流程都从这里进入。

发布到 crates.io 的流程由 tag 触发，说明见 [`docs/RELEASE.md`](./docs/RELEASE.md)。

## License

MIT
