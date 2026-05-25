use vv_llm::{
    embedding_clients::OpenAiCompatibleEmbeddingClient,
    rerank_clients::{CustomJsonHttpRerankClient, RerankMapping},
};

#[test]
fn embedding_adapter_builds_json_request_shape() {
    let client = OpenAiCompatibleEmbeddingClient::new(
        "text-embedding-3-small",
        "https://api.openai.com/v1",
        "sk-test",
    );
    let json = client.to_openai_json(&["hello", "world"]).unwrap();

    assert_eq!(json["model"], "text-embedding-3-small");
    assert_eq!(json["input"][0], "hello");
    assert_eq!(json["input"][1], "world");
}

#[test]
fn custom_rerank_mapping_builds_request_body() {
    let mapping = RerankMapping::default_siliconflow();
    let client = CustomJsonHttpRerankClient::new(
        "BAAI/bge-reranker-v2-m3",
        "https://api.siliconflow.cn/v1",
        "sk-test",
        mapping,
    );
    let body = client
        .build_request_body("Apple", &["apple", "banana"])
        .unwrap();

    assert_eq!(body["model"], "BAAI/bge-reranker-v2-m3");
    assert_eq!(body["query"], "Apple");
    assert_eq!(body["documents"][0], "apple");
}
