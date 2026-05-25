mod live_support;

use live_support::{load_live_settings, require_live, resolved_parts, run_with_timer};
use vv_llm::{
    chat_clients::create_chat_client,
    embedding_clients::create_embedding_client,
    rerank_clients::{CustomJsonHttpRerankClient, RerankMapping},
    BackendType, ChatRequest, Message, MessageRole,
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
            })
            .await
    })
    .await
    .unwrap();

    assert!(!response.content.trim().is_empty());
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_anthropic_direct_chat_if_configured() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let resolved = match settings.resolve_chat_model(BackendType::Anthropic, "claude-sonnet-4-6") {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!("[live] skipping direct Anthropic test: {error}");
            return;
        }
    };
    let (model, api_base, api_key) = resolved_parts(resolved);
    if !api_base.contains("api.anthropic.com") {
        eprintln!(
            "[live] skipping direct Anthropic test: configured endpoint is not api.anthropic.com"
        );
        return;
    }
    let client = create_chat_client(BackendType::Anthropic, model.clone(), api_base, api_key);

    let response = run_with_timer("anthropic_direct_chat", || async {
        client
            .create_completion(ChatRequest {
                model,
                messages: vec![
                    Message::text(MessageRole::System, "Answer with short lowercase words."),
                    Message::text(MessageRole::User, "Say pong."),
                ],
                options: Default::default(),
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
