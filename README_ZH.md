# vv-llm-rs

[English README](./README.md)

面向 Rust 的统一 LLM 客户端层。一套类型化 API，覆盖 chat、streaming、embedding、rerank、多模态消息、工具调用和多供应商 endpoint 解析。

```toml
[dependencies]
vv-llm = "0.4.8"
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
    request = request.with_thinking(vv_llm::ThinkingPreference::Disabled);

    let response = client.create(request).await?;

    println!("{}", response.content);
    Ok(())
}
```

现有的 `create_completion(request)` 调用和
`request.options.thinking = Some(json!(...))` 字段仍然兼容。
`ThinkingPreference` 在不改变 provider 线上 payload 的前提下，提供类型化的默认、开启、
带预算开启、关闭和 provider 自定义状态。

`ChatRequest` 是 provider-neutral 的 canonical request，并直接支持共享
contract 的 serde codec，包括字符串或对象形式的 `tool_choice`。
`create(request)` 始终返回非流式 completion；stream 有独立返回类型，必须显式调用
`create_stream(request)`，不会根据 `request.options.stream` 动态改变 `create` 的返回类型。

跨语言 contract 边界使用 `ChatRequest::from_contract(&value)` 和
`to_contract()`。这两个方法要求 model 非空，并保留 canonical `x_*` 扩展以及
图片的 `detail`、`cache_control`。直接 serde 接受运行时的默认 model 值。
扩展 map 和图片元数据是公开字段，穷举 struct literal 或 pattern 时必须包含。

### Middleware、重试与 Metadata

需要稳定的 `v1` hook、分类重试或响应 metadata 时，使用
`MiddlewareChatClient`：

```rust
use std::time::Duration;
use vv_llm::{MiddlewareChatClient, RetryPolicy};

let runtime = MiddlewareChatClient::new(client, vec![])?
    .with_retry_policy(
        RetryPolicy::new(3)
            .with_base_delay(Duration::from_millis(250))
            .with_total_timeout(Duration::from_secs(20)),
    );

let result = runtime.create_with_metadata(request).await?;
println!("{}", result.response.content);
println!(
    "provider={:?} attempts={} latency_ms={:?}",
    result.metadata.provider,
    result.metadata.attempts,
    result.metadata.latency_ms,
);
```

`create_with_metadata` 仅用于 completion metadata。流式请求请使用
`create_stream`；向 metadata API 传入 `stream: true` 会返回 configuration error，
不会静默按 completion 执行。可使用
`RetryPolicy::new(3).with_retryable_kinds([ErrorKind::RateLimited, ErrorKind::Network])`
自定义可重试错误集合。

`ErrorKind` 统一区分认证、限流、网络、超时、无效请求、上下文长度、内容策略、
模型不存在、provider 内部错误、序列化和配置错误。直接 HTTP adapter 会把
`retry-after-ms` 以及秒数或 HTTP-date 格式的 `Retry-After` 保留到分类错误中；
重试执行仍位于 middleware 层。

### 显式 Registry 与 Fallback

`ProviderRegistry` 不会自动发现 provider。调用方显式注册 factory 和 capabilities，
再提供有顺序的 `FallbackRoute`。不兼容 route 会在网络请求前跳过。认证和无效请求
错误默认不会切换 provider。

流式调用只能在建立 provider stream 或首个可见 chunk 之前 fallback。一旦返回 chunk，
后续 stream 错误会直接向上传播，不会重放。

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
        .create(ChatRequest::new(
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

对于 OpenAI-compatible 客户端，`create_stream` 始终发送 `stream: true`。调用方未提供 `ChatRequestOptions::stream_options` 时，还会默认发送 `{"include_usage": true}`，让需要显式 opt-in 的服务商返回最终 usage chunk；调用方显式提供的 `stream_options`（包括 `{"include_usage": false}`）会原样保留。该默认值不影响非流式请求和其他 provider adapter。

Fallback stream 会把只含 metadata、空内容的前导 delta 缓冲到首个可见的文本、reasoning
或工具调用 delta。首个可见边界之前的可重试错误可以切换 route；边界之后的错误会直接
传播且不会重放输出。Rust middleware 的 `on_stream_start` 表示 provider stream 已建立，
并且发生在第一个 stream item 被交付之前。

## Usage 统计

`ChatUsage` 保留原有的 `prompt_tokens`、`completion_tokens`、`total_tokens`，同时新增 provider-neutral 的 `input_tokens`、`output_tokens`、`cache_read_input_tokens`、`cache_creation_input_tokens`。所有字段均为可选值：provider 未上报 cache 字段时为 `None`，明确上报 0 时为 `Some(0)`。

`raw_usage` 保留 provider 原始 usage 对象，供诊断和未来兼容使用。OpenAI-compatible 的 cache read 会从 prompt/input token details、官方顶层 `cached_tokens` 或其他兼容的顶层字段归一化；Anthropic 的 cache read/cache creation 直接映射；Bedrock 的 `cache_write_input_tokens` 映射到 `cache_creation_input_tokens`。string、小数、负数或溢出的 token 值不会被强转为归一化计数，但仍保留在 `raw_usage` 中。

通用 OpenAI-compatible 客户端会把缺失的 cache-read 值保留为 `None`。通过 `create_chat_client(BackendType::Moonshot, ...)` 创建的客户端会在 completion 和 stream 中应用 Moonshot provider 契约：仅当所有已识别的 cache-read 字段都完全省略时，才把 `cache_read_input_tokens` 归一化为 `Some(0)`；字段明确存在但为 `null` 或无效值时仍为 `None`。该策略不会向 `raw_usage` 插入任何伪造字段。

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
request.tool_choice = Some("required".into());

let response = client.create(request).await?;
for call in response.tool_calls {
    println!("{} {}", call.name, call.arguments);
}
```

工具结果轮次使用带 `tool_call_id` 的 `MessageRole::Tool`；assistant 发出的工具调用放在 `Message.tool_calls` 中。

## Provider 扩展字段

OpenAI-compatible provider 经常会暴露额外的请求 / 响应字段，用于 reasoning trace、thinking 控制或供应商专有工具元数据。`vv-llm` 把这些能力放在 provider-neutral 的类型化字段里，调用方不需要自己手写协议转换：

- `ChatRequest.extra_body` 会把对象字段合并到请求 JSON 根层。
- `ChatRequestOptions::thinking` 会把显式的启用、关闭或自适应 thinking 配置传给支持该能力的 provider adapter。
- `Message.reasoning_content` 会保留 assistant 历史消息里的 reasoning 内容。
- `MessageContent::Text.cache_control` 和 `ChatTool.cache_control` 会保留 Anthropic prompt-cache 断点。
- `ToolCall.extra_content` 会保留供应商工具调用元数据，例如 Google thought signature。
- `ChatResponse.reasoning_content` 和流式 `ChatStreamDelta.reasoning_content` 会暴露支持的 reasoning 输出。

对于 OpenAI-compatible assistant 历史消息，只要消息不含工具调用，请求就一定发送
`content` 键。完全空消息和仅 reasoning 的消息会发送 `content: ""`；
`reasoning_content` 始终保持独立，不会转成可见文本。纯工具调用 assistant 消息继续使用协议允许的省略 `content` 形式。

OpenAI-compatible adapter 继续使用类型化 request 构建，通过 `async-openai` BYOT 解码响应以保留 provider usage 扩展，并把原始 JSON 响应归一化成公开的 `vv-llm` 类型。

## 多模态输入

用户消息里可以混合文本和图片。对要求 inline base64 的供应商，图片 URL 使用 data URL。

```rust
use vv_llm::{Message, MessageContent, MessageRole};

let message = Message {
    role: MessageRole::User,
    content: vec![
        MessageContent::Text {
            text: "这张图片里有什么？".to_string(),
            cache_control: None,
            extensions: Default::default(),
        },
        MessageContent::ImageUrl {
            url: "data:image/png;base64,...".to_string(),
            detail: None,
            cache_control: None,
            extensions: Default::default(),
            nested_extensions: Default::default(),
            nested_image: false,
        },
    ],
    name: None,
    tool_call_id: None,
    tool_calls: Vec::new(),
    reasoning_content: None,
    extensions: Default::default(),
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
- **OpenAI-compatible adapter** — chat 使用保留响应头的 typed HTTP transport 归一化，embedding 继续使用 `async-openai`
- **Provider 扩展字段** — 类型化 reasoning content、请求 `extra_body` 和 tool-call `extra_content`
- **Anthropic 支持** — 直接 Messages API，以及 Bedrock Converse transport
- **Streaming 归一化** — provider stream event 统一转成 `ChatStreamDelta`
- **工具调用** — 标准化 function/tool 定义、assistant tool call 和 tool-result 轮次
- **多模态消息** — 对支持的 provider 发送文本和图片消息块
- **Vertex 鉴权** — Google access token 换取和进程内缓存
- **检索客户端** — OpenAI-compatible embedding 与自定义 JSON rerank
- **Token 统计** — 本地 tiktoken fallback，以及基于 settings 的 token server/provider tokenizer 调用
- **类型化错误** — configuration、provider、HTTP、serialization、model、endpoint 等错误类型
- **版本化 middleware** — 稳定的 `v1` 请求、响应、stream-start 和错误 hook
- **Retry executor** — 分类瞬时错误、退避、抖动、`Retry-After` 和总 deadline
- **显式 fallback** — 按模型 capabilities 过滤的有序 provider registry route
- **Scripted client** — 用确定性的响应、错误和 stream 脚本做契约测试

## 使用示例

[`crates/vv-llm/examples/`](crates/vv-llm/examples/README.md) 提供基础 chat、
流式输出、工具调用、多模态、contract JSON、类型化 thinking、middleware
metadata 和显式 fallback 的 Cargo 示例。

## 工具函数

```rust
use vv_llm::utilities::{
    count_message_tokens, count_tokens, count_tokens_with_settings, normalize_text_messages,
    RetryPolicy,
};
```

| 函数 | 说明 |
|---|---|
| `normalize_text_messages` | 合并相邻同角色文本消息，不合并图片或工具数据 |
| `count_tokens` | 使用支持的模型 tokenizer 统计 token |
| `count_tokens_with_settings` | 优先使用已配置的 token server 和 provider tokenizer endpoint，然后回退到本地计数 |
| `count_message_tokens` | 统计 chat request 中的文本、图片占位和工具 token |
| `parse_retry_after` | 把 `retry-after-ms` 或秒数/HTTP-date `Retry-After` 解析为 `Duration` |
| `RetryPolicy` | 退避、抖动、可配置的可重试分类和总 deadline 策略 |
| `execute_with_retry` | 使用 `RetryPolicy` 执行异步操作 |

## 目录结构

```text
vv-llm-rs/
  Cargo.toml
  crates/vv-llm/
    contract/v1.0.0/      # 锁定的跨语言 schema、fixture 与模型目录
    src/
      chat_clients/       # Chat client、stream 归一化、Vertex 鉴权
      contract.rs         # contract metadata 与 manifest/lock accessor
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
python scripts/sync_contract.py --check
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

crate 提供 `contract_metadata()`、`contract_manifest_json()` 与
`contract_consumer_lock_json()` 供诊断和下游工具使用。Contract check
默认只离线校验 vendored lock；同步必须显式指定 source：

```bash
python scripts/sync_contract.py --source /secure/path/vv-llm-contract/dist/release-v1.0.0
VV_LLM_CONTRACT_SOURCE=/secure/path/vv-llm-contract/dist/release-v1.0.0 python scripts/sync_contract.py
```

`python scripts/sync_contract.py --check` 校验包内的 contract 副本。

[Python/Rust 能力矩阵](docs/ARCHITECTURE.md#pythonrust-capability-matrix)明确记录
provider、限流、token server、middleware、retry、fallback 与 Scripted client
之间的有意差异。

真实 API 集成测试默认 ignored。运行前必须显式设置 secret settings 文件路径：

```bash
VV_LLM_SETTINGS_JSON=/secure/path/llm_settings.json VV_LLM_RUN_LIVE_TESTS=1 \
  ./scripts/run_live_tests.sh
```

工程文档放在 [`docs/`](./docs/README.md)。架构说明、provider adapter 行为、live test 规则、安全约束和维护流程都从这里进入。

发布到 crates.io 的流程由 tag 触发，说明见 [`docs/RELEASE.md`](./docs/RELEASE.md)。

## License

MIT
