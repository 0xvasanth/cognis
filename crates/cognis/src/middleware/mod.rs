//! Agent middleware — hooks that wrap an LLM call.
//!
//! A middleware sits between the agent's `ThinkNode` and the actual LLM
//! invocation. Each middleware can:
//! - Inspect / mutate the outgoing request (`messages` + `tool_defs`).
//! - Short-circuit with a synthetic response.
//! - Wrap the inner call (retry, fallback, cache).
//! - Inspect / mutate the response.
//!
//! The contract is one canonical async method: [`Middleware::call`].
//! Implementations call `next.invoke(ctx)` to continue the chain.
//!
//! ```ignore
//! let pipeline = MiddlewarePipeline::new()
//!     .push(ModelRetry::new(3))
//!     .push(ModelFallback::new(backup_client))
//!     .push(PromptCaching::new())
//!     .build(client);
//! ```

pub mod approval_gate;
pub mod context_editing;
pub mod context_injection;
pub mod filesystem;
pub mod human_in_the_loop;
pub mod model_call_limit;
pub mod model_fallback;
pub mod model_retry;
pub mod patch_tool_calls;
pub mod pii;
pub mod planning;
pub mod prompt_caching;
pub mod rate_limit;
pub mod recovery;
pub mod redaction;
pub mod subagent;
pub mod summarization;
pub mod todo;
pub mod token_counter;
pub mod tool_call_limit;
pub mod tool_emulator;
pub mod tool_retry;
pub mod tool_selection;

#[cfg(test)]
pub(crate) mod tests_util;

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::{ChatOptions, ChatResponse};
use cognis_llm::tools::ToolDefinition;
use cognis_llm::Client;

pub use approval_gate::{ApprovalGate, AutoApproveAll, AutoRejectAll, ChatApproval, ChatApprover};
pub use context_editing::{CapMessageLength, ContextEditing, DropMatching, EditPolicy};
pub use context_injection::{ContextInjection, ContextProvider, FnContextProvider};
pub use filesystem::{FilesystemMiddleware, WorkspaceLister};
pub use human_in_the_loop::{AlwaysSkip, HumanDecision, HumanInTheLoop, HumanResponder};
pub use model_call_limit::ModelCallLimit;
pub use model_fallback::ModelFallback;
pub use model_retry::ModelRetry;
pub use patch_tool_calls::{FnToolCallPatcher, PatchToolCalls, ToolCallPatcher};
pub use pii::PiiRedactor;
pub use planning::Planning;
pub use prompt_caching::PromptCaching;
pub use rate_limit::{RateLimit, RateLimiter, TokenBucket};
pub use recovery::{FixedRecovery, FnRecovery, Recovery, RecoveryStrategy};
pub use redaction::RegexRedactor;
pub use subagent::{SubagentMiddleware, SubagentRouter};
pub use summarization::Summarization;
pub use todo::TodoMiddleware;
pub use token_counter::TokenCounter;
pub use tool_call_limit::ToolCallLimit;
pub use tool_emulator::{EmulatorSource, MapEmulator, ToolEmulator};
pub use tool_retry::{ToolRetry, ToolRetryClassifier};
pub use tool_selection::{LimitTools, ToolAllowList, ToolDenyList, ToolFilter, ToolSelection};

/// What flows through the middleware chain.
#[derive(Debug, Clone)]
pub struct MiddlewareCtx {
    /// Conversation messages being sent to the LLM.
    pub messages: Vec<Message>,
    /// Tool definitions being sent to the LLM.
    pub tool_defs: Vec<ToolDefinition>,
    /// Per-call options.
    pub opts: ChatOptions,
}

impl MiddlewareCtx {
    /// Build a new context.
    pub fn new(messages: Vec<Message>, tool_defs: Vec<ToolDefinition>, opts: ChatOptions) -> Self {
        Self {
            messages,
            tool_defs,
            opts,
        }
    }
}

/// The "next" hop in a middleware chain — an opaque thing that turns a
/// [`MiddlewareCtx`] into a [`ChatResponse`].
#[async_trait]
pub trait Next: Send + Sync {
    /// Invoke the next layer.
    async fn invoke(&self, ctx: MiddlewareCtx) -> Result<ChatResponse>;
}

/// One middleware layer.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Wrap a call. Implementations typically:
    /// - Mutate `ctx`.
    /// - `next.invoke(ctx).await`.
    /// - Mutate the returned `ChatResponse`.
    /// They may also short-circuit (return without calling `next`).
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse>;

    /// Friendly name for telemetry / debugging.
    fn name(&self) -> &str {
        "Middleware"
    }
}

/// Builder + entry point for a chain. Layers are pushed outermost-last:
/// `pipeline.push(retry).push(redact)` runs `redact -> retry -> client`.
pub struct MiddlewarePipeline {
    layers: Vec<Arc<dyn Middleware>>,
}

impl Default for MiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewarePipeline {
    /// Empty pipeline.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Append a layer (runs *before* layers pushed earlier in the chain
    /// when invoked, since the chain is built from the inside out).
    pub fn push(mut self, m: impl Middleware + 'static) -> Self {
        self.layers.push(Arc::new(m));
        self
    }

    /// Append a pre-boxed layer.
    pub fn push_boxed(mut self, m: Arc<dyn Middleware>) -> Self {
        self.layers.push(m);
        self
    }

    /// Wrap a `Client` and produce a [`PipelinedClient`] that goes through
    /// the chain on every chat call.
    pub fn build(self, client: Client) -> PipelinedClient {
        PipelinedClient {
            client,
            layers: self.layers,
        }
    }
}

/// `Client` wrapped with a middleware chain. Has the same surface as the
/// underlying `Client` for chat methods.
#[derive(Clone)]
pub struct PipelinedClient {
    client: Client,
    layers: Vec<Arc<dyn Middleware>>,
}

impl PipelinedClient {
    /// Run the chain. The innermost layer is the raw `Client`; outer layers
    /// run in reverse-push order so the most-recently-pushed layer is the
    /// outermost wrapper.
    pub async fn invoke(
        &self,
        messages: Vec<Message>,
        tool_defs: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let ctx = MiddlewareCtx::new(messages, tool_defs, opts);
        let next: Arc<dyn Next> = Arc::new(ClientNext {
            client: self.client.clone(),
        });
        let chained = self
            .layers
            .iter()
            .rev()
            .fold(next, |acc, layer| -> Arc<dyn Next> {
                Arc::new(LayerNext {
                    layer: layer.clone(),
                    next: acc,
                })
            });
        chained.invoke(ctx).await
    }

    /// The underlying client.
    pub fn client(&self) -> &Client {
        &self.client
    }
}

struct ClientNext {
    client: Client,
}

#[async_trait]
impl Next for ClientNext {
    async fn invoke(&self, ctx: MiddlewareCtx) -> Result<ChatResponse> {
        self.client
            .provider()
            .chat_completion_with_tools(ctx.messages, ctx.tool_defs, ctx.opts)
            .await
    }
}

struct LayerNext {
    layer: Arc<dyn Middleware>,
    next: Arc<dyn Next>,
}

#[async_trait]
impl Next for LayerNext {
    async fn invoke(&self, ctx: MiddlewareCtx) -> Result<ChatResponse> {
        self.layer.call(ctx, self.next.clone()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis_llm::chat::{HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};

    struct Echo {
        calls: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl LLMProvider for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let last = messages.last().cloned().unwrap_or(Message::ai(""));
            Ok(ChatResponse {
                message: Message::ai(format!("echo: {}", last.content())),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "echo".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<cognis_core::RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    /// Trivial counter-middleware that increments before delegating.
    struct CountingMw {
        seen: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl Middleware for CountingMw {
        async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            next.invoke(ctx).await
        }
        fn name(&self) -> &str {
            "Counting"
        }
    }

    fn client() -> (Client, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Client::new(Arc::new(Echo {
                calls: calls.clone(),
            })),
            calls,
        )
    }

    #[tokio::test]
    async fn empty_pipeline_passes_through() {
        let (c, calls) = client();
        let pipe = MiddlewarePipeline::new().build(c);
        let resp = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(resp.message.content(), "echo: hi");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn layers_run_in_reverse_push_order_and_each_sees_call() {
        let (c, calls) = client();
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));
        let pipe = MiddlewarePipeline::new()
            .push(CountingMw { seen: a.clone() })
            .push(CountingMw { seen: b.clone() })
            .build(c);
        let _ = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(b.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
