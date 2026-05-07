//! Mutate the LLM's emitted `tool_calls` before they reach the dispatcher.
//!
//! Useful for: renaming tools the model picked but you exposed under a
//! different name, coercing arg types, redacting secrets out of args,
//! retargeting calls to a different tool entirely.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result, ToolCall};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Edits `tool_calls` in place. Receives a mutable slice — patcher is
/// free to rename, drop args, append new args, etc.
pub trait ToolCallPatcher: Send + Sync {
    /// Edit the slice in place.
    fn patch(&self, calls: &mut Vec<ToolCall>);
}

/// Closure-backed patcher.
pub struct FnToolCallPatcher<F: Fn(&mut Vec<ToolCall>) + Send + Sync>(pub F);

impl<F: Fn(&mut Vec<ToolCall>) + Send + Sync> ToolCallPatcher for FnToolCallPatcher<F> {
    fn patch(&self, calls: &mut Vec<ToolCall>) {
        (self.0)(calls)
    }
}

/// Middleware applying a patcher to every `Message::Ai`'s `tool_calls`.
pub struct PatchToolCalls {
    patcher: Arc<dyn ToolCallPatcher>,
}

impl PatchToolCalls {
    /// Build with a patcher.
    pub fn new(patcher: Arc<dyn ToolCallPatcher>) -> Self {
        Self { patcher }
    }
}

#[async_trait]
impl Middleware for PatchToolCalls {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let mut resp = next.invoke(ctx).await?;
        if let Message::Ai(a) = &mut resp.message {
            self.patcher.patch(&mut a.tool_calls);
        }
        Ok(resp)
    }
    fn name(&self) -> &str {
        "PatchToolCalls"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use cognis_core::{AiMessage, Message, RunnableStream};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};
    use cognis_llm::Client;

    /// Provider that emits a fixed AiMessage with one tool call.
    struct WithToolCall;
    #[async_trait]
    impl LLMProvider for WithToolCall {
        fn name(&self) -> &str {
            "tc"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> cognis_core::Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::Ai(AiMessage {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "old_name".into(),
                        arguments: serde_json::json!({"x": 1}),
                    }],
                    parts: Vec::new(),
                }),
                usage: Some(Usage::default()),
                finish_reason: "tool_calls".into(),
                model: "tc".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> cognis_core::Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> cognis_core::Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn renames_tool_call() {
        let _ = StdMutex::new(()); // silence unused
        let _ = make_recording_provider("");
        let patcher = FnToolCallPatcher(|calls: &mut Vec<ToolCall>| {
            for c in calls {
                if c.name == "old_name" {
                    c.name = "new_name".into();
                }
            }
        });
        let pipe = MiddlewarePipeline::new()
            .push(PatchToolCalls::new(Arc::new(patcher)))
            .build(Client::new(Arc::new(WithToolCall)));
        let r = pipe
            .invoke(
                vec![Message::human("go")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.tool_calls()[0].name, "new_name");
    }

    #[tokio::test]
    async fn drops_disallowed_calls() {
        let patcher = FnToolCallPatcher(|calls: &mut Vec<ToolCall>| {
            calls.retain(|c| c.name != "old_name");
        });
        let pipe = MiddlewarePipeline::new()
            .push(PatchToolCalls::new(Arc::new(patcher)))
            .build(Client::new(Arc::new(WithToolCall)));
        let r = pipe
            .invoke(
                vec![Message::human("go")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert!(r.message.tool_calls().is_empty());
    }
}
