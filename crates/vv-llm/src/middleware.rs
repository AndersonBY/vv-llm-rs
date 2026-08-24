use crate::{
    execute_with_retry, ChatClient, ChatRequest, ChatResponse, ChatStream, CompletionResult,
    ResponseMetadata, RetryPolicy, VvLlmError,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct MiddlewareContext {
    pub provider: String,
    pub model: String,
    pub attempt: u32,
    pub attributes: HashMap<String, Value>,
}

impl MiddlewareContext {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            attempt: 0,
            attributes: HashMap::new(),
        }
    }
}

#[async_trait]
pub trait ChatMiddlewareV1: Send + Sync {
    fn api_version(&self) -> &'static str {
        "v1"
    }

    async fn on_request(
        &self,
        _context: &mut MiddlewareContext,
        request: ChatRequest,
    ) -> Result<ChatRequest, VvLlmError> {
        Ok(request)
    }

    async fn on_response(
        &self,
        _context: &MiddlewareContext,
        response: ChatResponse,
    ) -> Result<ChatResponse, VvLlmError> {
        Ok(response)
    }

    /// Called after the provider stream has been established and before the
    /// first stream item is yielded. This is not a first-visible-delta hook;
    /// metadata-only prelude items may still follow it.
    async fn on_stream_start(&self, _context: &MiddlewareContext) -> Result<(), VvLlmError> {
        Ok(())
    }

    async fn on_error(&self, _context: &MiddlewareContext, _error: &VvLlmError) {}
}

pub struct MiddlewareChatClient {
    inner: Box<dyn ChatClient>,
    middleware: Vec<Arc<dyn ChatMiddlewareV1>>,
    retry_policy: RetryPolicy,
}

impl MiddlewareChatClient {
    pub fn new(
        inner: Box<dyn ChatClient>,
        middleware: Vec<Arc<dyn ChatMiddlewareV1>>,
    ) -> Result<Self, VvLlmError> {
        if let Some(item) = middleware.iter().find(|item| item.api_version() != "v1") {
            return Err(VvLlmError::Configuration(format!(
                "unsupported middleware API version: {}",
                item.api_version()
            )));
        }
        Ok(Self {
            inner,
            middleware,
            retry_policy: RetryPolicy::new(1),
        })
    }

    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    async fn prepare_request(
        &self,
        request: ChatRequest,
    ) -> Result<(MiddlewareContext, ChatRequest), VvLlmError> {
        let mut context = MiddlewareContext::new(self.inner.provider_name(), request.model.clone());
        let mut request = request;
        for middleware in &self.middleware {
            request = middleware.on_request(&mut context, request).await?;
        }
        context.model.clone_from(&request.model);
        Ok((context, request))
    }

    async fn execute_completion(
        &self,
        request: ChatRequest,
    ) -> Result<(ChatResponse, u32), VvLlmError> {
        let (base_context, request) = self.prepare_request(request).await?;
        let attempts = AtomicU32::new(0);

        let response = execute_with_retry(
            || {
                let request = request.clone();
                let mut context = base_context.clone();
                context.attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    match self.inner.create(request).await {
                        Ok(mut response) => {
                            for middleware in self.middleware.iter().rev() {
                                response = middleware.on_response(&context, response).await?;
                            }
                            Ok(response)
                        }
                        Err(error) => {
                            for middleware in self.middleware.iter().rev() {
                                middleware.on_error(&context, &error).await;
                            }
                            Err(error)
                        }
                    }
                }
            },
            self.retry_policy,
        )
        .await?;

        Ok((response, attempts.load(Ordering::SeqCst)))
    }

    pub async fn create_with_metadata(
        &self,
        request: ChatRequest,
    ) -> Result<CompletionResult<ChatResponse>, VvLlmError> {
        if request.options.stream == Some(true) {
            return Err(VvLlmError::Configuration(
                "create_with_metadata does not support streaming; use create_stream".to_string(),
            ));
        }
        let started = Instant::now();
        let (response, attempts) = self.execute_completion(request).await?;
        let metadata = ResponseMetadata {
            provider: Some(self.inner.provider_name().to_string()),
            model: Some(response.model.clone()),
            response_id: Some(response.id.clone()),
            attempts,
            latency_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
            ..Default::default()
        };
        Ok(CompletionResult { response, metadata })
    }
}

#[async_trait]
impl ChatClient for MiddlewareChatClient {
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name()
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        self.execute_completion(request)
            .await
            .map(|(response, _)| response)
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let (base_context, request) = self.prepare_request(request).await?;
        let attempts = AtomicU32::new(0);

        execute_with_retry(
            || {
                let request = request.clone();
                let mut context = base_context.clone();
                context.attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    match self.inner.create_stream(request).await {
                        Ok(stream) => {
                            for middleware in self.middleware.iter().rev() {
                                middleware.on_stream_start(&context).await?;
                            }
                            Ok(stream)
                        }
                        Err(error) => {
                            for middleware in self.middleware.iter().rev() {
                                middleware.on_error(&context, &error).await;
                            }
                            Err(error)
                        }
                    }
                }
            },
            self.retry_policy,
        )
        .await
    }
}
