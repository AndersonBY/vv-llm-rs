use crate::{
    ChatClient, ChatRequest, ChatResponse, ChatStream, CompletionResult, ErrorKind,
    ModelCapabilities, ResponseMetadata, VvLlmError,
};
use async_trait::async_trait;
use futures_util::{stream, StreamExt};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

type ClientFactory = Arc<dyn Fn() -> Box<dyn ChatClient> + Send + Sync>;

#[derive(Clone)]
pub struct ProviderRegistration {
    pub name: String,
    pub capabilities: ModelCapabilities,
    factory: ClientFactory,
}

impl ProviderRegistration {
    pub fn create(&self) -> Box<dyn ChatClient> {
        (self.factory)()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackRoute {
    pub provider: String,
    pub model: String,
}

impl FallbackRoute {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, ProviderRegistration>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<Factory>(
        &mut self,
        name: impl Into<String>,
        factory: Factory,
        capabilities: ModelCapabilities,
    ) -> Result<(), VvLlmError>
    where
        Factory: Fn() -> Box<dyn ChatClient> + Send + Sync + 'static,
    {
        let name = name.into();
        if self.providers.contains_key(&name) {
            return Err(VvLlmError::Configuration(format!(
                "provider is already registered: {name}"
            )));
        }
        self.providers.insert(
            name.clone(),
            ProviderRegistration {
                name,
                capabilities,
                factory: Arc::new(factory),
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<&ProviderRegistration, VvLlmError> {
        self.providers
            .get(name)
            .ok_or_else(|| VvLlmError::Configuration(format!("provider is not registered: {name}")))
    }

    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }
}

pub struct FallbackChatClient {
    registry: Arc<ProviderRegistry>,
    routes: Vec<FallbackRoute>,
    fallback_on: HashSet<ErrorKind>,
}

impl FallbackChatClient {
    pub fn new(
        registry: Arc<ProviderRegistry>,
        routes: Vec<FallbackRoute>,
    ) -> Result<Self, VvLlmError> {
        if routes.is_empty() {
            return Err(VvLlmError::Configuration(
                "fallback routes cannot be empty".to_string(),
            ));
        }
        Ok(Self {
            registry,
            routes,
            fallback_on: default_fallback_errors(),
        })
    }

    pub fn with_fallback_on(mut self, fallback_on: HashSet<ErrorKind>) -> Self {
        self.fallback_on = fallback_on;
        self
    }

    async fn execute_completion(
        &self,
        request: ChatRequest,
    ) -> Result<(ChatResponse, usize, FallbackRoute), VvLlmError> {
        let mut last_error = None;
        for (index, route) in self.routes.iter().enumerate() {
            let registration = self.registry.get(&route.provider)?;
            let mut routed = request.clone();
            routed.model.clone_from(&route.model);
            if let Err(error) = registration.capabilities.validate_request(&routed) {
                last_error = Some(error);
                continue;
            }
            match registration.create().create(routed).await {
                Ok(response) => return Ok((response, index, route.clone())),
                Err(error)
                    if self.fallback_on.contains(&error.kind())
                        && index + 1 < self.routes.len() =>
                {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            VvLlmError::Configuration("no fallback route was eligible".to_string())
        }))
    }

    pub async fn create_with_metadata(
        &self,
        request: ChatRequest,
    ) -> Result<CompletionResult<ChatResponse>, VvLlmError> {
        let started = Instant::now();
        let (response, fallback_index, route) = self.execute_completion(request).await?;
        let metadata = ResponseMetadata {
            provider: Some(route.provider),
            model: Some(response.model.clone()),
            response_id: Some(response.id.clone()),
            latency_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
            fallback_index,
            ..Default::default()
        };
        Ok(CompletionResult { response, metadata })
    }
}

#[async_trait]
impl ChatClient for FallbackChatClient {
    fn provider_name(&self) -> &'static str {
        "fallback"
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        self.execute_completion(request)
            .await
            .map(|(response, _, _)| response)
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        let mut last_error = None;
        for (index, route) in self.routes.iter().enumerate() {
            let registration = self.registry.get(&route.provider)?;
            let mut routed = request.clone();
            routed.model.clone_from(&route.model);
            routed.options.stream = Some(true);
            if let Err(error) = registration.capabilities.validate_request(&routed) {
                last_error = Some(error);
                continue;
            }

            let stream_result = registration.create().create_stream(routed).await;
            let mut provider_stream = match stream_result {
                Ok(provider_stream) => provider_stream,
                Err(error)
                    if self.fallback_on.contains(&error.kind())
                        && index + 1 < self.routes.len() =>
                {
                    last_error = Some(error);
                    continue;
                }
                Err(error) => return Err(error),
            };

            match provider_stream.next().await {
                Some(Ok(first)) => {
                    return Ok(Box::pin(
                        stream::once(async move { Ok(first) }).chain(provider_stream),
                    ));
                }
                None => return Ok(Box::pin(stream::empty())),
                Some(Err(error))
                    if self.fallback_on.contains(&error.kind())
                        && index + 1 < self.routes.len() =>
                {
                    last_error = Some(error);
                }
                Some(Err(error)) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            VvLlmError::Configuration("no fallback route was eligible".to_string())
        }))
    }
}

fn default_fallback_errors() -> HashSet<ErrorKind> {
    HashSet::from([
        ErrorKind::RateLimited,
        ErrorKind::Network,
        ErrorKind::Timeout,
        ErrorKind::ProviderInternal,
        ErrorKind::ModelNotFound,
    ])
}
