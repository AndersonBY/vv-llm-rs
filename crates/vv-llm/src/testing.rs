use crate::{ChatClient, ChatRequest, ChatResponse, ChatStream, ChatStreamDelta, VvLlmError};
use async_trait::async_trait;
use futures_util::stream;
use std::collections::VecDeque;
use std::sync::Mutex;

pub enum ScriptedStep {
    Response(ChatResponse),
    Error(VvLlmError),
}

impl ScriptedStep {
    pub fn response(response: ChatResponse) -> Self {
        Self::Response(response)
    }

    pub fn error(error: VvLlmError) -> Self {
        Self::Error(error)
    }
}

pub struct ScriptedStream {
    pub chunks: Vec<Result<ChatStreamDelta, VvLlmError>>,
}

impl ScriptedStream {
    pub fn new(chunks: Vec<Result<ChatStreamDelta, VvLlmError>>) -> Self {
        Self { chunks }
    }
}

pub struct ScriptedChatClient {
    provider: &'static str,
    completion_steps: Mutex<VecDeque<ScriptedStep>>,
    stream_steps: Mutex<VecDeque<ScriptedStream>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedChatClient {
    pub fn new(provider: &'static str, steps: Vec<ScriptedStep>) -> Self {
        Self {
            provider,
            completion_steps: Mutex::new(steps.into()),
            stream_steps: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn with_streams(self, streams: Vec<ScriptedStream>) -> Self {
        *self
            .stream_steps
            .lock()
            .expect("scripted stream lock poisoned") = streams.into();
        self
    }

    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("scripted request lock poisoned")
            .clone()
    }

    fn record(&self, request: &ChatRequest) {
        self.requests
            .lock()
            .expect("scripted request lock poisoned")
            .push(request.clone());
    }
}

#[async_trait]
impl ChatClient for ScriptedChatClient {
    fn provider_name(&self) -> &'static str {
        self.provider
    }

    async fn create_completion(&self, request: ChatRequest) -> Result<ChatResponse, VvLlmError> {
        self.record(&request);
        let step = self
            .completion_steps
            .lock()
            .expect("scripted completion lock poisoned")
            .pop_front()
            .ok_or_else(|| {
                VvLlmError::Configuration("scripted client has no remaining steps".to_string())
            })?;
        match step {
            ScriptedStep::Response(response) => Ok(response),
            ScriptedStep::Error(error) => Err(error),
        }
    }

    async fn create_stream(&self, request: ChatRequest) -> Result<ChatStream, VvLlmError> {
        self.record(&request);
        let step = self
            .stream_steps
            .lock()
            .expect("scripted stream lock poisoned")
            .pop_front()
            .ok_or_else(|| {
                VvLlmError::Configuration(
                    "scripted client has no remaining stream steps".to_string(),
                )
            })?;
        Ok(Box::pin(stream::iter(step.chunks)))
    }
}
