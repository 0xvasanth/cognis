//! Tool emulator — replay or mock tool-call responses without hitting
//! the real LLM.
//!
//! Use cases: deterministic tests, golden-dataset replay, offline
//! development. The middleware intercepts the chat call and consults
//! an [`EmulatorSource`] to decide whether to short-circuit with a
//! synthetic [`ChatResponse`].
//!
//! Customization: implement [`EmulatorSource`] for your replay format
//! (HAR-style recording, fixture file, in-memory map). For in-process
//! tests, [`MapEmulator`] keys responses by the last user message.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Message, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable source of synthetic responses.
#[async_trait]
pub trait EmulatorSource: Send + Sync {
    /// Decide whether to emulate. Returning `Some(resp)` short-circuits
    /// the middleware chain; returning `None` falls through to the real
    /// LLM.
    async fn lookup(&self, ctx: &MiddlewareCtx) -> Option<ChatResponse>;
}

/// Closure-based source.
#[async_trait]
impl<F> EmulatorSource for F
where
    F: Fn(&MiddlewareCtx) -> Option<ChatResponse> + Send + Sync,
{
    async fn lookup(&self, ctx: &MiddlewareCtx) -> Option<ChatResponse> {
        (self)(ctx)
    }
}

/// In-memory keyed source. The key is the last user message's content;
/// values are the canned responses to return.
#[derive(Default)]
pub struct MapEmulator {
    table: HashMap<String, ChatResponse>,
}

impl MapEmulator {
    /// New empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Map `input` → `response`.
    pub fn record(mut self, input: impl Into<String>, response: ChatResponse) -> Self {
        self.table.insert(input.into(), response);
        self
    }
}

#[async_trait]
impl EmulatorSource for MapEmulator {
    async fn lookup(&self, ctx: &MiddlewareCtx) -> Option<ChatResponse> {
        let key = ctx
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m, Message::Human(_)))
            .map(|m| m.content().to_string())?;
        self.table.get(&key).cloned()
    }
}

/// Middleware: short-circuits with the source's response if available.
pub struct ToolEmulator {
    source: Arc<dyn EmulatorSource>,
}

impl ToolEmulator {
    /// Wrap a source.
    pub fn new<S: EmulatorSource + 'static>(source: S) -> Self {
        Self {
            source: Arc::new(source),
        }
    }

    /// Wrap a pre-boxed source (use when sharing a source across pipelines).
    pub fn from_arc(source: Arc<dyn EmulatorSource>) -> Self {
        Self { source }
    }
}

#[async_trait]
impl Middleware for ToolEmulator {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        if let Some(r) = self.source.lookup(&ctx).await {
            return Ok(r);
        }
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "ToolEmulator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, FixedNext};

    #[tokio::test]
    async fn map_emulator_short_circuits_on_match() {
        let emu = MapEmulator::new().record("hello", ok_resp("emulated"));
        let mw = ToolEmulator::new(emu);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("real")));
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hello")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "emulated");
    }

    #[tokio::test]
    async fn map_emulator_passes_through_on_miss() {
        let emu = MapEmulator::new().record("x", ok_resp("nope"));
        let mw = ToolEmulator::new(emu);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("real")));
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("y")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "real");
    }

    #[tokio::test]
    async fn closure_source_works() {
        let mw = ToolEmulator::new(|ctx: &MiddlewareCtx| {
            if ctx.messages.iter().any(|m| m.content().contains("magic")) {
                Some(ok_resp("zap"))
            } else {
                None
            }
        });
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("real")));
        let r = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("magic word")],
                    vec![],
                    Default::default(),
                ),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "zap");
    }
}
