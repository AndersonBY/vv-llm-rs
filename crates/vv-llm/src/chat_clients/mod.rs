mod anthropic;
mod openai_compatible;
mod vertex;

use crate::{
    utilities::normalize_image_inputs_async, BackendType, ChatRequest, ChatStreamDelta,
    ResolvedModelConfig, VvLlmError,
};
use async_trait::async_trait;
use futures_core::Stream;
use std::pin::Pin;

pub use anthropic::{AnthropicBedrockChatClient, AnthropicChatClient};
pub use openai_compatible::OpenAiCompatibleChatClient;
pub use vertex::{GoogleAccessToken, GoogleAccessTokenProvider, VertexOpenAiChatClient};

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamDelta, VvLlmError>> + Send>>;

#[async_trait]
pub trait ChatClient: Send + Sync {
    fn provider_name(&self) -> &'static str;
    async fn create(&self, request: ChatRequest) -> Result<crate::ChatResponse, VvLlmError> {
        self.create_completion(request).await
    }
    async fn create_completion(
        &self,
        request: ChatRequest,
    ) -> Result<crate::ChatResponse, VvLlmError>;
    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError>;
}

pub fn create_chat_client(
    backend: BackendType,
    model: impl Into<String>,
    api_base: impl Into<String>,
    api_key: impl Into<String>,
) -> Box<dyn ChatClient> {
    let model = model.into();
    let api_base = api_base.into();
    let api_key = api_key.into();

    match backend {
        BackendType::Anthropic => Box::new(AnthropicChatClient::new(model, api_base, api_key)),
        BackendType::Moonshot => Box::new(OpenAiCompatibleChatClient::for_moonshot(
            model, api_base, api_key,
        )),
        _ => Box::new(OpenAiCompatibleChatClient::new(model, api_base, api_key)),
    }
}

pub fn create_chat_client_from_resolved(
    resolved: ResolvedModelConfig,
) -> Result<Box<dyn ChatClient>, VvLlmError> {
    let max_image_dimension = resolved.model.max_image_dimension;
    let backend = resolved.backend.as_str();
    let model = resolved.model_id;
    let api_base = resolved.endpoint.api_base.unwrap_or_default();
    let api_key = resolved.endpoint.api_key.unwrap_or_default();

    if backend == BackendType::Anthropic.as_str()
        && (resolved.endpoint.is_bedrock
            || resolved.endpoint.endpoint_type.as_deref() == Some("anthropic_bedrock"))
    {
        let client = Box::new(AnthropicBedrockChatClient::new(
            model,
            api_base,
            resolved.endpoint.region,
            resolved.endpoint.credentials,
        )?) as Box<dyn ChatClient>;
        return with_model_image_limit(client, max_image_dimension);
    }

    if resolved.endpoint.endpoint_type.as_deref() == Some("openai_vertex") {
        let client = Box::new(VertexOpenAiChatClient::new(
            model,
            api_base,
            resolved.endpoint.credentials,
        )?) as Box<dyn ChatClient>;
        return with_model_image_limit(client, max_image_dimension);
    }

    let backend = match backend {
        "anthropic" => BackendType::Anthropic,
        "openai" => BackendType::OpenAI,
        "zhipuai" => BackendType::ZhiPuAI,
        "minimax" => BackendType::MiniMax,
        "moonshot" => BackendType::Moonshot,
        "mistral" => BackendType::Mistral,
        "deepseek" => BackendType::DeepSeek,
        "qwen" => BackendType::Qwen,
        "groq" => BackendType::Groq,
        "local" => BackendType::Local,
        "yi" => BackendType::Yi,
        "gemini" => BackendType::Gemini,
        "baichuan" => BackendType::Baichuan,
        "stepfun" => BackendType::StepFun,
        "xai" => BackendType::XAI,
        "xiaomi" => BackendType::Xiaomi,
        "ernie" => BackendType::Ernie,
        other => {
            return Err(VvLlmError::Configuration(format!(
                "unsupported chat backend: {other}"
            )))
        }
    };

    with_model_image_limit(
        create_chat_client(backend, model, api_base, api_key),
        max_image_dimension,
    )
}

fn with_model_image_limit(
    client: Box<dyn ChatClient>,
    max_image_dimension: Option<u32>,
) -> Result<Box<dyn ChatClient>, VvLlmError> {
    if max_image_dimension == Some(0) {
        return Err(VvLlmError::Configuration(
            "max_image_dimension must be at least 1".to_string(),
        ));
    }
    match max_image_dimension {
        Some(max_image_dimension) => Ok(Box::new(ModelImageLimitChatClient {
            inner: client,
            max_image_dimension,
        })),
        None => Ok(client),
    }
}

struct ModelImageLimitChatClient {
    inner: Box<dyn ChatClient>,
    max_image_dimension: u32,
}

#[async_trait]
impl ChatClient for ModelImageLimitChatClient {
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }

    async fn create_completion(
        &self,
        mut request: ChatRequest,
    ) -> Result<crate::ChatResponse, VvLlmError> {
        normalize_image_inputs_async(&mut request, Some(self.max_image_dimension)).await?;
        self.inner.create_completion(request).await
    }

    async fn create_stream(&self, mut request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        normalize_image_inputs_async(&mut request, Some(self.max_image_dimension)).await?;
        self.inner.create_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    struct CaptureClient {
        request: Arc<Mutex<Option<ChatRequest>>>,
    }

    #[async_trait]
    impl ChatClient for CaptureClient {
        fn provider_name(&self) -> &'static str {
            "capture"
        }

        async fn create_completion(
            &self,
            request: ChatRequest,
        ) -> Result<crate::ChatResponse, VvLlmError> {
            *self.request.lock().unwrap() = Some(request);
            Ok(crate::ChatResponse {
                id: "capture".to_string(),
                model: "capture".to_string(),
                content: String::new(),
                tool_calls: Vec::new(),
                reasoning_content: None,
                usage: None,
            })
        }

        async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
            *self.request.lock().unwrap() = Some(request);
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    #[tokio::test]
    async fn model_image_limit_wrapper_normalizes_before_forwarding() {
        let captured = Arc::new(Mutex::new(None));
        let client = with_model_image_limit(
            Box::new(CaptureClient {
                request: captured.clone(),
            }),
            Some(128),
        )
        .unwrap();
        let request = ChatRequest::new(
            "deepseek-v4-flash-vision-exp",
            vec![crate::Message {
                role: crate::MessageRole::User,
                content: vec![crate::MessageContent::ImageUrl {
                    url: test_png_data_url(400, 200),
                }],
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
            }],
        );

        client.create_completion(request).await.unwrap();
        let captured = captured.lock().unwrap().take().unwrap();
        let url = match &captured.messages[0].content[0] {
            crate::MessageContent::ImageUrl { url } => url,
            _ => panic!("expected image content"),
        };
        let bytes = STANDARD.decode(url.split_once(',').unwrap().1).unwrap();
        let image = image::load_from_memory(&bytes).unwrap();

        assert_eq!((image.width(), image.height()), (128, 64));
    }

    fn test_png_data_url(width: u32, height: u32) -> String {
        let image =
            DynamicImage::ImageRgb8(ImageBuffer::from_pixel(width, height, Rgb([35, 96, 120])));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(bytes.into_inner())
        )
    }
}
