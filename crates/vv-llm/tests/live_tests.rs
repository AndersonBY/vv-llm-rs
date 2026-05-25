mod live_support;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures_util::StreamExt;
use live_support::{load_live_settings, require_live, resolved_parts, run_with_timer};
use std::fs;
use vv_llm::{
    chat_clients::{create_chat_client, create_chat_client_from_resolved},
    embedding_clients::create_embedding_client,
    rerank_clients::{CustomJsonHttpRerankClient, RerankMapping},
    BackendType, ChatRequest, ChatRequestOptions, ChatTool, Message, MessageContent, MessageRole,
};

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_deepseek_openai_compatible_chat_completion() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::DeepSeek, "deepseek-chat")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::DeepSeek, model.clone(), api_base, api_key);

    let response = run_with_timer("deepseek_chat", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Reply with the word pong.",
                )],
                options: Default::default(),
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    assert!(!response.content.trim().is_empty());
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_qwen_openai_compatible_chat_with_system_prompt() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::Qwen, "qwen-turbo")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::Qwen, model.clone(), api_base, api_key);

    let response = run_with_timer("qwen_chat_system", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![
                    Message::text(MessageRole::System, "Answer with short lowercase words."),
                    Message::text(MessageRole::User, "Say pong."),
                ],
                options: Default::default(),
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    assert!(response.content.to_ascii_lowercase().contains("pong"));
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_zhipuai_openai_compatible_chat_completion() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::ZhiPuAI, "glm-4-flash")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::ZhiPuAI, model.clone(), api_base, api_key);

    let response = run_with_timer("zhipuai_chat", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Reply with the word pong.",
                )],
                options: Default::default(),
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    assert!(!response.content.trim().is_empty());
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_chat_from_resolved() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();

    let response = run_with_timer("anthropic_resolved_chat", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![
                    Message::text(MessageRole::System, "Answer with short lowercase words."),
                    Message::text(MessageRole::User, "Say pong."),
                ],
                options: Default::default(),
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    assert!(response.content.to_ascii_lowercase().contains("pong"));
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_bedrock_image_understanding() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();
    let image_url = cat_image_data_url();

    let response = run_with_timer("anthropic_bedrock_image", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![Message {
                    role: MessageRole::User,
                    content: vec![
                        MessageContent::Text {
                            text: "What animal is in this image? Reply with one English word."
                                .to_string(),
                        },
                        MessageContent::ImageUrl { url: image_url },
                    ],
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                }],
                options: ChatRequestOptions {
                    max_tokens: Some(16),
                    ..Default::default()
                },
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    let content = response.content.to_ascii_lowercase();
    assert!(
        content.contains("cat") || content.contains("kitten"),
        "unexpected image response: {}",
        response.content
    );
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_bedrock_tool_call() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();

    let response = run_with_timer("anthropic_bedrock_tool_call", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Use the get_current_weather tool for New York. Do not answer directly.",
                )],
                options: ChatRequestOptions {
                    max_tokens: Some(512),
                    ..Default::default()
                },
                tools: vec![weather_tool()],
                tool_choice: Some("required".to_string()),
            })
            .await
    })
    .await
    .unwrap();

    let tool_call = response
        .tool_calls
        .first()
        .expect("expected Anthropic Bedrock to return a tool call");
    assert_eq!(tool_call.name, "get_current_weather");
    assert!(
        tool_call.arguments.contains("New York") || tool_call.arguments.contains("new york"),
        "unexpected tool arguments: {}",
        tool_call.arguments
    );
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_bedrock_tool_result_multi_turn() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();
    let user_message = Message::text(
        MessageRole::User,
        "Use the get_current_weather tool for New York. Do not answer directly.",
    );

    let first_response = run_with_timer("anthropic_bedrock_tool_call_for_multiturn", || async {
        client
            .create_completion(ChatRequest {
                model: model.clone(),
                messages: vec![user_message.clone()],
                options: ChatRequestOptions {
                    max_tokens: Some(512),
                    ..Default::default()
                },
                tools: vec![weather_tool()],
                tool_choice: Some("required".to_string()),
            })
            .await
    })
    .await
    .unwrap();
    let tool_call = first_response
        .tool_calls
        .first()
        .cloned()
        .expect("expected first turn to return a tool call");

    let final_response = run_with_timer("anthropic_bedrock_tool_result_multiturn", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![
                    user_message,
                    Message {
                        role: MessageRole::Assistant,
                        content: Vec::new(),
                        name: None,
                        tool_call_id: None,
                        tool_calls: vec![tool_call.clone()],
                    },
                    Message {
                        role: MessageRole::Tool,
                        content: vec![MessageContent::Text {
                            text: "72F and sunny".to_string(),
                        }],
                        name: None,
                        tool_call_id: Some(tool_call.id),
                        tool_calls: Vec::new(),
                    },
                    Message::text(
                        MessageRole::User,
                        "Use the tool result to answer in one short sentence.",
                    ),
                ],
                options: ChatRequestOptions {
                    max_tokens: Some(128),
                    ..Default::default()
                },
                tools: vec![weather_tool()],
                tool_choice: Some("auto".to_string()),
            })
            .await
    })
    .await
    .unwrap();

    let content = final_response.content.to_ascii_lowercase();
    assert!(
        content.contains("72") || content.contains("sunny"),
        "unexpected multi-turn response: {}",
        final_response.content
    );
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_bedrock_streaming_text() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();

    let mut stream = run_with_timer("anthropic_bedrock_stream_text", || async {
        client
            .create_stream(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Reply with the word pong.",
                )],
                options: ChatRequestOptions {
                    max_tokens: Some(64),
                    stream: Some(true),
                    ..Default::default()
                },
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    let mut content = String::new();
    let mut saw_done = false;
    while let Some(delta) = stream.next().await {
        let delta = delta.unwrap();
        content.push_str(&delta.content);
        saw_done |= delta.done;
    }

    assert!(
        content.to_ascii_lowercase().contains("pong"),
        "streamed content: {content}"
    );
    assert!(saw_done, "stream did not emit done delta");
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_bedrock_streaming_tool_call() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6")
        .unwrap();
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();

    let mut stream = run_with_timer("anthropic_bedrock_stream_tool_call", || async {
        client
            .create_stream(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Use the get_current_weather tool for New York. Do not answer directly.",
                )],
                options: ChatRequestOptions {
                    max_tokens: Some(512),
                    stream: Some(true),
                    ..Default::default()
                },
                tools: vec![weather_tool()],
                tool_choice: Some("required".to_string()),
            })
            .await
    })
    .await
    .unwrap();

    let mut tool_name = String::new();
    let mut tool_args = String::new();
    while let Some(delta) = stream.next().await {
        let delta = delta.unwrap();
        for tool_call in delta.tool_calls {
            if !tool_call.name.is_empty() {
                tool_name = tool_call.name;
            }
            tool_args.push_str(&tool_call.arguments);
        }
    }

    assert_eq!(tool_name, "get_current_weather");
    assert!(
        tool_args.contains("New York") || tool_args.contains("new york"),
        "unexpected streamed tool arguments: {tool_args}"
    );
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_gemini_openai_vertex_chat_completion() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::Gemini, "gemini-3-pro")
        .unwrap();
    if resolved.endpoint.endpoint_type.as_deref() != Some("openai_vertex") {
        eprintln!("Skipping Vertex live test; gemini-3-pro is not configured for openai_vertex");
        return;
    }
    let model = resolved.model_id.clone();
    let client = create_chat_client_from_resolved(resolved).unwrap();

    let response = run_with_timer("gemini_openai_vertex_chat", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![Message::text(
                    MessageRole::User,
                    "Reply with the word pong.",
                )],
                options: ChatRequestOptions {
                    max_tokens: Some(64),
                    ..Default::default()
                },
                tools: Vec::new(),
                tool_choice: None,
            })
            .await
    })
    .await
    .unwrap();

    assert!(response.content.to_ascii_lowercase().contains("pong"));
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_siliconflow_embedding() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_embedding_model("siliconflow", "Qwen/Qwen3-Embedding-4B")
            .unwrap(),
    );
    let client = create_embedding_client("siliconflow", model, api_base, api_key);

    let response = run_with_timer("siliconflow_embedding", || async {
        client
            .create_embeddings(&["hello world", "vector search"])
            .await
    })
    .await
    .unwrap();

    assert_eq!(response.data.len(), 2);
    assert!(!response.data[0].embedding.is_empty());
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_siliconflow_rerank() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = settings
        .resolve_rerank_model("siliconflow", "BAAI/bge-reranker-v2-m3")
        .unwrap();
    let model = resolved.model_id;
    let api_base = resolved.endpoint.api_base.unwrap_or_default();
    let api_key = resolved.endpoint.api_key.unwrap_or_default();
    let client = CustomJsonHttpRerankClient::new(
        model,
        api_base,
        api_key,
        RerankMapping::default_siliconflow(),
    );

    let response = run_with_timer("siliconflow_rerank", || async {
        vv_llm::RerankClient::rerank(&client, "apple", &["banana", "apple"]).await
    })
    .await
    .unwrap();

    assert!(!response.results.is_empty());
}

fn weather_tool() -> ChatTool {
    ChatTool::function(
        "get_current_weather",
        "Get the current weather in a given location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. New York, NY"
                },
                "unit": {
                    "type": "string",
                    "enum": ["fahrenheit", "celsius"]
                }
            },
            "required": ["location"]
        }),
    )
}

fn cat_image_data_url() -> String {
    let image = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cat.png"
    ))
    .unwrap();
    format!("data:image/png;base64,{}", STANDARD.encode(image))
}
