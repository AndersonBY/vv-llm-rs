use crate::{
    BackendType, ChatTool, EndpointConfig, LlmSettings, Message, MessageContent, MessageRole,
    ResolvedModelConfig, VvLlmError,
};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::{json, Value};

pub fn count_tokens_fallback(text: &str) -> usize {
    text.split_whitespace().count()
}

pub fn count_tokens(text: &str, model: &str) -> Result<usize, VvLlmError> {
    let normalized_model = model.to_ascii_lowercase();
    if normalized_model.starts_with("abab") || normalized_model.starts_with("minimax") {
        return Ok((text.chars().count() as f64 / 1.33) as usize);
    }

    let bpe = if normalized_model == "gpt-3.5-turbo"
        || normalized_model.starts_with("moonshot")
        || normalized_model.starts_with("kimi")
        || normalized_model.starts_with("gemini")
        || normalized_model.starts_with("stepfun")
        || normalized_model.starts_with("glm")
    {
        tiktoken_rs::cl100k_base_singleton()
    } else {
        tiktoken_rs::o200k_base_singleton()
    };

    Ok(bpe.encode_with_special_tokens(text).len())
}

pub async fn count_tokens_with_settings(
    settings: &LlmSettings,
    text: &str,
    model: &str,
) -> Result<usize, VvLlmError> {
    count_token_value_with_settings(settings, Value::String(text.to_string()), model).await
}

pub async fn count_token_value_with_settings(
    settings: &LlmSettings,
    text: Value,
    model: &str,
) -> Result<usize, VvLlmError> {
    if let Ok(Some(tokens)) = count_with_token_server(settings, &text, model).await {
        return Ok(tokens);
    }

    let text = match text {
        Value::String(text) => text,
        other => other.to_string(),
    };

    if let Ok(Some(tokens)) = count_with_provider_tokenizer(settings, &text, model).await {
        return Ok(tokens);
    }

    count_tokens(&text, model)
}

async fn count_with_token_server(
    settings: &LlmSettings,
    text: &Value,
    model: &str,
) -> Result<Option<usize>, VvLlmError> {
    let Some(server) = settings.token_server.as_ref() else {
        return Ok(None);
    };
    let base_url = server
        .url
        .clone()
        .unwrap_or_else(|| format!("http://{}:{}", server.host, server.port));
    let url = join_url(&base_url, "/count_tokens");
    let response = reqwest::Client::new()
        .post(url)
        .json(&json!({ "text": text, "model": model }))
        .send()
        .await
        .map_err(|error| VvLlmError::Http(error.to_string()))?;
    if !response.status().is_success() {
        return Ok(None);
    }
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| VvLlmError::Provider(error.to_string()))?;
    Ok(value_at_path(&body, &["total_tokens"]))
}

async fn count_with_provider_tokenizer(
    settings: &LlmSettings,
    text: &str,
    model: &str,
) -> Result<Option<usize>, VvLlmError> {
    let Some(backend) = tokenizer_backend_for_model(model) else {
        return Ok(None);
    };
    let resolved = match settings.resolve_chat_model(backend, model) {
        Ok(resolved) => resolved,
        Err(_) => return Ok(None),
    };

    match backend {
        BackendType::MiniMax => count_minimax_tokens(text, &resolved).await.map(Some),
        BackendType::Moonshot => count_moonshot_tokens(text, &resolved).await.map(Some),
        BackendType::Gemini => count_gemini_tokens(text, &resolved).await.map(Some),
        BackendType::StepFun => count_stepfun_tokens(text, &resolved).await.map(Some),
        BackendType::ZhiPuAI => count_zhipu_tokens(text, &resolved).await.map(Some),
        BackendType::Anthropic => count_anthropic_tokens(text, &resolved).await,
        _ => Ok(None),
    }
}

fn tokenizer_backend_for_model(model: &str) -> Option<BackendType> {
    let model = model.to_ascii_lowercase();
    if model.starts_with("abab") || model.starts_with("minimax") {
        Some(BackendType::MiniMax)
    } else if model.starts_with("moonshot") || model.starts_with("kimi") {
        Some(BackendType::Moonshot)
    } else if model.starts_with("gemini") {
        Some(BackendType::Gemini)
    } else if model.starts_with("stepfun") || model.starts_with("step-") {
        Some(BackendType::StepFun)
    } else if model.starts_with("glm") {
        Some(BackendType::ZhiPuAI)
    } else if model.starts_with("claude") {
        Some(BackendType::Anthropic)
    } else {
        None
    }
}

async fn count_minimax_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<usize, VvLlmError> {
    let api_base = resolved
        .endpoint
        .api_base
        .as_deref()
        .unwrap_or("https://api.minimax.chat/v1");
    let body = post_json(
        &resolved.endpoint,
        join_url(api_base, "/tokenize"),
        json!({
            "model": resolved.model_id,
            "tokens_to_generate": 128,
            "temperature": 0.2,
            "messages": [
                { "sender_type": "USER", "text": text }
            ]
        }),
    )
    .await?;
    value_at_path(&body, &["segments_num"]).ok_or_else(|| {
        VvLlmError::Provider("MiniMax token count response missing segments_num".to_string())
    })
}

async fn count_moonshot_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<usize, VvLlmError> {
    let api_base = resolved
        .endpoint
        .api_base
        .as_deref()
        .unwrap_or("https://api.moonshot.cn/v1");
    let body = post_json(
        &resolved.endpoint,
        join_url(api_base, "/tokenizers/estimate-token-count"),
        json!({
            "model": resolved.model_id,
            "messages": [
                { "role": "user", "content": text }
            ]
        }),
    )
    .await?;
    value_at_path(&body, &["data", "total_tokens"]).ok_or_else(|| {
        VvLlmError::Provider("Moonshot token count response missing data.total_tokens".to_string())
    })
}

async fn count_gemini_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<usize, VvLlmError> {
    let api_base = resolved
        .endpoint
        .api_base
        .as_deref()
        .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
    let api_base = strip_gemini_openai_suffix(api_base);
    let tokenizer_model = if resolved.model_id.starts_with("gemini-3") {
        "gemini-2.5-pro"
    } else {
        resolved.model_id.as_str()
    };
    let url = join_url(&api_base, &format!("/models/{tokenizer_model}:countTokens"));
    let client = client_for_endpoint(&resolved.endpoint)?;
    let mut request = client.post(url).json(&json!({
        "contents": {
            "role": "USER",
            "parts": [
                { "text": text }
            ]
        }
    }));
    if let Some(api_key) = &resolved.endpoint.api_key {
        request = request.query(&[("key", api_key)]);
    }
    let response = request
        .send()
        .await
        .map_err(|error| VvLlmError::Http(error.to_string()))?;
    if !response.status().is_success() {
        return Err(VvLlmError::Provider(format!(
            "Gemini token count failed with status {}",
            response.status()
        )));
    }
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| VvLlmError::Provider(error.to_string()))?;
    value_at_path(&body, &["totalTokens"]).ok_or_else(|| {
        VvLlmError::Provider("Gemini token count response missing totalTokens".to_string())
    })
}

async fn count_stepfun_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<usize, VvLlmError> {
    let api_base = resolved.endpoint.api_base.as_deref().unwrap_or_default();
    let body = post_json(
        &resolved.endpoint,
        join_url(api_base, "/token/count"),
        json!({
            "model": resolved.model_id,
            "messages": [
                { "role": "user", "content": text }
            ]
        }),
    )
    .await?;
    value_at_path(&body, &["data", "total_tokens"]).ok_or_else(|| {
        VvLlmError::Provider("StepFun token count response missing data.total_tokens".to_string())
    })
}

async fn count_zhipu_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<usize, VvLlmError> {
    let api_base = resolved
        .endpoint
        .api_base
        .as_deref()
        .unwrap_or("https://open.bigmodel.cn/api/paas/v4");
    let tokenizer_model = supported_zhipu_tokenizer_model(&resolved.model_id);
    let body = post_json(
        &resolved.endpoint,
        join_url(api_base, "/tokenizer"),
        json!({
            "model": tokenizer_model,
            "messages": [
                { "role": "user", "content": text }
            ]
        }),
    )
    .await?;
    value_at_path(&body, &["usage", "prompt_tokens"]).ok_or_else(|| {
        VvLlmError::Provider("ZhiPuAI token count response missing usage.prompt_tokens".to_string())
    })
}

async fn count_anthropic_tokens(
    text: &str,
    resolved: &ResolvedModelConfig,
) -> Result<Option<usize>, VvLlmError> {
    let endpoint_type = resolved
        .endpoint
        .endpoint_type
        .as_deref()
        .unwrap_or("default")
        .to_ascii_lowercase();
    if endpoint_type == "anthropic_bedrock"
        || endpoint_type == "anthropic_vertex"
        || resolved.endpoint.is_bedrock
        || resolved.endpoint.is_vertex
    {
        return Ok(None);
    }

    let api_base = resolved
        .endpoint
        .api_base
        .as_deref()
        .unwrap_or("https://api.anthropic.com");
    let client = client_for_endpoint(&resolved.endpoint)?;
    let mut request = client
        .post(join_url(api_base, "/v1/messages/count_tokens"))
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "application/json")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": resolved.model_id,
            "messages": [
                { "role": "user", "content": text }
            ]
        }));
    if let Some(api_key) = &resolved.endpoint.api_key {
        request = request.header("x-api-key", api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| VvLlmError::Http(error.to_string()))?;
    if !response.status().is_success() {
        return Err(VvLlmError::Provider(format!(
            "Anthropic token count failed with status {}",
            response.status()
        )));
    }
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| VvLlmError::Provider(error.to_string()))?;
    Ok(value_at_path(&body, &["input_tokens"]))
}

async fn post_json(
    endpoint: &EndpointConfig,
    url: String,
    body: Value,
) -> Result<Value, VvLlmError> {
    let client = client_for_endpoint(endpoint)?;
    let mut request = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .json(&body);
    if let Some(api_key) = &endpoint.api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| VvLlmError::Http(error.to_string()))?;
    if !response.status().is_success() {
        return Err(VvLlmError::Provider(format!(
            "token count failed with status {}",
            response.status()
        )));
    }
    response
        .json::<Value>()
        .await
        .map_err(|error| VvLlmError::Provider(error.to_string()))
}

fn client_for_endpoint(endpoint: &EndpointConfig) -> Result<reqwest::Client, VvLlmError> {
    let mut builder = reqwest::Client::builder();
    if let Some(proxy) = &endpoint.proxy {
        builder = builder.proxy(
            reqwest::Proxy::all(proxy)
                .map_err(|error| VvLlmError::Configuration(error.to_string()))?,
        );
    }
    builder
        .build()
        .map_err(|error| VvLlmError::Configuration(error.to_string()))
}

fn value_at_path(value: &Value, path: &[&str]) -> Option<usize> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_u64().map(|value| value as usize)
}

fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn strip_gemini_openai_suffix(api_base: &str) -> String {
    let trimmed = api_base.trim_end_matches('/');
    trimmed
        .strip_suffix("/openai")
        .unwrap_or(trimmed)
        .to_string()
}

fn supported_zhipu_tokenizer_model(model: &str) -> &str {
    match model {
        "glm-4-plus" | "glm-4-long" | "glm-4-0520" | "glm-4-air" | "glm-4-flash" => model,
        _ => "glm-4-plus",
    }
}

pub fn calculate_image_tokens(width: u32, height: u32, model: &str) -> usize {
    if model.to_ascii_lowercase().starts_with("moonshot") {
        return 1024;
    }
    if width == 0 || height == 0 {
        return 0;
    }

    let mut width = width as f64;
    let mut height = height as f64;

    if width > 2048.0 || height > 2048.0 {
        let aspect_ratio = width / height;
        if aspect_ratio > 1.0 {
            width = 2048.0;
            height = (2048.0 / aspect_ratio).floor();
        } else {
            width = (2048.0 * aspect_ratio).floor();
            height = 2048.0;
        }
    }

    if width >= height && height > 768.0 {
        width = ((768.0 / height) * width).floor();
        height = 768.0;
    } else if height > width && width > 768.0 {
        height = ((768.0 / width) * height).floor();
        width = 768.0;
    }

    let tiles_width = (width / 512.0).ceil() as usize;
    let tiles_height = (height / 512.0).ceil() as usize;
    85 + 170 * (tiles_width * tiles_height)
}

pub fn count_message_tokens(
    messages: &[Message],
    tools: &[ChatTool],
    model: &str,
    native_multimodal: bool,
) -> Result<usize, VvLlmError> {
    let mut text_parts = Vec::new();
    let mut image_count = 0usize;

    for message in messages {
        for content in &message.content {
            match content {
                MessageContent::Text { text, .. } => text_parts.push(text.as_str()),
                MessageContent::ImageUrl { .. } => image_count += 1,
            }
        }
    }

    let combined_text = text_parts.join("\n");
    let mut tokens = if combined_text.is_empty() {
        0
    } else {
        count_tokens(&combined_text, model)?
    };

    if image_count > 0 {
        tokens += if native_multimodal {
            image_count * calculate_image_tokens(2048, 2048, model)
        } else {
            image_count
        };
    }

    if !tools.is_empty() {
        let tools_json = serde_json::to_string(tools)?;
        tokens += count_tokens(&tools_json, model)?;
    }

    Ok(tokens)
}

pub fn cutoff_messages(
    mut messages: Vec<Message>,
    max_count: usize,
    model: &str,
) -> Result<Vec<Message>, VvLlmError> {
    if messages.is_empty() {
        return Ok(messages);
    }

    let mut total = 0usize;
    let system_message = if messages
        .first()
        .map(|message| message.role == MessageRole::System)
        .unwrap_or(false)
    {
        let system = messages.remove(0);
        let system_tokens = message_text_token_count(&system, model)?;
        if system_tokens > max_count {
            return Ok(vec![truncate_message_tail(&system, max_count)]);
        }
        total += system_tokens;
        Some(system)
    } else {
        None
    };

    for (index, message) in messages.iter().rev().enumerate() {
        total += message_text_token_count(message, model)?;
        if total < max_count {
            continue;
        }

        let mut result = system_message.into_iter().collect::<Vec<_>>();
        if index == 0 {
            result.push(truncate_message_tail(message, max_count));
        } else {
            let start = messages.len().saturating_sub(index);
            result.extend(messages[start..].iter().cloned());
        }
        return Ok(result);
    }

    let mut result = system_message.into_iter().collect::<Vec<_>>();
    result.extend(messages);
    Ok(result)
}

fn message_text_token_count(message: &Message, model: &str) -> Result<usize, VvLlmError> {
    let text = message.text_content().unwrap_or_default();
    if text.is_empty() {
        Ok(0)
    } else {
        count_tokens(&text, model)
    }
}

fn truncate_message_tail(message: &Message, max_chars: usize) -> Message {
    let text = message.text_content().unwrap_or_default();
    let truncated = if max_chars == 0 {
        String::new()
    } else {
        let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
        chars.reverse();
        chars.into_iter().collect()
    };

    Message {
        role: message.role,
        content: vec![MessageContent::text(truncated)],
        name: message.name.clone(),
        tool_call_id: message.tool_call_id.clone(),
        tool_calls: message.tool_calls.clone(),
        reasoning_content: message.reasoning_content.clone(),
    }
}
