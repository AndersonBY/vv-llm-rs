use vv_llm::{
    embedding_clients::OpenAiCompatibleEmbeddingClient,
    rerank_clients::{CustomJsonHttpRerankClient, RerankClient, RerankMapping},
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

#[tokio::test]
async fn custom_rerank_error_preserves_retry_after_hint() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut buffer = [0_u8; 2048];
        let _ = socket.read(&mut buffer).await.unwrap();
        let body = r#"{"error":"slow down"}"#;
        let response = format!(
            "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\nretry-after: 2.25\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let client = CustomJsonHttpRerankClient::new(
        "rerank-model",
        api_base,
        "sk-test",
        RerankMapping::default_siliconflow(),
    );
    let error = client.rerank("query", &["document"]).await.unwrap_err();
    server.await.unwrap();

    assert_eq!(error.kind(), vv_llm::ErrorKind::RateLimited);
    assert_eq!(error.retry_after_seconds(), Some(2.25));
}
