use vv_llm::{settings::LlmSettings, BackendType};

#[test]
fn loads_v2_settings_and_resolves_chat_model_endpoint() {
    let raw = r#"{
      "VERSION": "2",
      "endpoints": [{"id":"openai-default","api_base":"https://api.openai.com/v1","api_key":"sk-test"}],
      "backends": {"openai": {"models": {"gpt-4o": {"id":"gpt-4o","endpoints":["openai-default"],"context_length":128000}}}},
      "embedding_backends": {},
      "rerank_backends": {}
    }"#;

    let settings = LlmSettings::from_json_str(raw).unwrap();
    let resolved = settings
        .resolve_chat_model(BackendType::OpenAI, "gpt-4o")
        .unwrap();

    assert_eq!(resolved.model.id, "gpt-4o");
    assert_eq!(resolved.endpoint.id, "openai-default");
    assert_eq!(
        resolved.endpoint.api_base.as_deref(),
        Some("https://api.openai.com/v1")
    );
}
