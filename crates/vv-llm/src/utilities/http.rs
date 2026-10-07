use crate::VvLlmError;
use serde_json::Value;
use std::time::SystemTime;

pub(crate) async fn ensure_success(
    response: reqwest::Response,
) -> Result<reqwest::Response, VvLlmError> {
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(openai_http_error(response).await)
    }
}

async fn openai_http_error(response: reqwest::Response) -> VvLlmError {
    let status = response.status();
    let headers = response.headers().clone();
    let retry_after = crate::utilities::parse_retry_after_headers(&headers, SystemTime::now());
    let body = response.text().await.unwrap_or_default();
    let parsed = serde_json::from_str::<Value>(&body).unwrap_or(Value::Null);
    let error_value = parsed.get("error").unwrap_or(&parsed);
    let message = error_value
        .get("message")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("OpenAI-compatible HTTP {status}")
            } else {
                body.clone()
            }
        });
    let provider_code = error_value
        .get("code")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let request_id = headers
        .get("x-request-id")
        .or_else(|| headers.get("request-id"))
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);

    let mut error = VvLlmError::from_status_with_retry_after(status.as_u16(), message, retry_after);
    if let VvLlmError::Classified(details) = &mut error {
        details.provider_code = provider_code;
        details.request_id = request_id;
    }
    error
}
