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
async fn live_openai_compatible_chat_completion() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::OpenAI, "gpt-4o-mini")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::OpenAI, model.clone(), api_base, api_key);

    let response = run_with_timer("openai_chat", || async {
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
async fn live_openai_compatible_chat_with_system_prompt_and_stop_instruction() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::OpenAI, "gpt-4o-mini")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::OpenAI, model.clone(), api_base, api_key);

    let response = run_with_timer("openai_chat_system", || async {
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
async fn live_anthropic_chat_completion() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::Anthropic, "claude-3-5-haiku-latest")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::Anthropic, model.clone(), api_base, api_key);

    let response = run_with_timer("anthropic_chat", || async {
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
async fn live_anthropic_chat_with_system_prompt() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_chat_model(BackendType::Anthropic, "claude-3-5-haiku-latest")
            .unwrap(),
    );
    let client = create_chat_client(BackendType::Anthropic, model.clone(), api_base, api_key);

    let response = run_with_timer("anthropic_chat_system", || async {
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
async fn live_openai_embedding() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_embedding_model("openai", "text-embedding-3-small")
            .unwrap(),
    );
    let client = create_embedding_client("openai", model, api_base, api_key);

    let response = run_with_timer("openai_embedding", || async {
        client.create_embeddings(&["hello world"]).await
    })
    .await
    .unwrap();

    assert_eq!(response.data.len(), 1);
    assert!(!response.data[0].embedding.is_empty());
}

#[tokio::test]
#[ignore = "live API call; run with VV_LLM_RUN_LIVE_TESTS=1 cargo test --test live_tests -- --ignored"]
async fn live_siliconflow_embedding() {
    require_live();
    let settings = load_live_settings(true).unwrap();
    let (model, api_base, api_key) = resolved_parts(
        settings
            .resolve_embedding_model("siliconflow", "BAAI/bge-large-zh-v1.5")
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
    let model = resolved.model.id;
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
