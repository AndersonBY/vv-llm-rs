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

#[test]
fn embedding_adapter_preserves_empty_input_list() {
    let client = OpenAiCompatibleEmbeddingClient::new(
        "text-embedding-3-small",
        "https://api.openai.com/v1",
        "sk-test",
    );
    let json = client.to_openai_json(&[]).unwrap();

    assert_eq!(json["model"], "text-embedding-3-small");
    assert_eq!(json["input"].as_array().unwrap().len(), 0);
}

#[test]
fn custom_rerank_mapping_supports_custom_path_and_top_n() {
    let mapping = RerankMapping {
        method: "POST".to_string(),
        path: "/rank".to_string(),
    };
    let client = CustomJsonHttpRerankClient::new(
        "custom-rerank-id",
        "https://example.com/v1/",
        "sk-test",
        mapping,
    );

    let body = client
        .build_request_body_with_top_n("hello", &["doc-a", "doc-b"], Some(1))
        .unwrap();

    assert_eq!(client.endpoint_url(), "https://example.com/v1/rank");
    assert_eq!(body["model"], "custom-rerank-id");
    assert_eq!(body["top_n"], 1);
}
