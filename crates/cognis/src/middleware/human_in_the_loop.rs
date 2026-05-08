//! Pause an LLM call to ask a human a question.
//!
//! Distinct from the approval-gate (which yes/no-gates a tool call):
//! `HumanInTheLoop` lets the middleware pause and await an arbitrary
//! human reply, optionally feeding the reply back as a system message.
//!
//! Customization: implement [`HumanResponder`] to plug in a Slack
//! prompt, a CLI prompt, a web inbox poller, etc. The default is
//! [`AlwaysSkip`] which is a no-op (handy for tests / when the
//! middleware is registered globally but only fires conditionally).

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Decision returned by a [`HumanResponder`].
#[derive(Debug, Clone)]
pub enum HumanDecision {
    /// Skip the human prompt entirely; continue with the original call.
    Skip,
    /// Inject this content as a system message before the call.
    InjectSystem(String),
    /// Replace the response with a synthetic message (short-circuits).
    Override(String),
}

/// Pluggable strategy for asking a human.
#[async_trait]
pub trait HumanResponder: Send + Sync {
    /// Receives the call's context and decides what to do.
    async fn ask(&self, ctx: &MiddlewareCtx) -> Result<HumanDecision>;
}

/// No-op responder — never pauses.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysSkip;

#[async_trait]
impl HumanResponder for AlwaysSkip {
    async fn ask(&self, _ctx: &MiddlewareCtx) -> Result<HumanDecision> {
        Ok(HumanDecision::Skip)
    }
}

/// Closure-based responder.
#[async_trait]
impl<F, Fut> HumanResponder for F
where
    F: Fn(MiddlewareCtx) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<HumanDecision>> + Send,
{
    async fn ask(&self, ctx: &MiddlewareCtx) -> Result<HumanDecision> {
        (self)(ctx.clone()).await
    }
}

/// Boxed gate predicate.
pub type HitLGate = Arc<dyn Fn(&MiddlewareCtx) -> bool + Send + Sync>;

/// Middleware that consults a [`HumanResponder`] before each call.
pub struct HumanInTheLoop {
    responder: Arc<dyn HumanResponder>,
    /// Predicate: only consult the responder when this returns true.
    /// Default `None` ⇒ consult every call.
    gate: Option<HitLGate>,
}

impl HumanInTheLoop {
    /// Build with a responder.
    pub fn new<R: HumanResponder + 'static>(responder: R) -> Self {
        Self {
            responder: Arc::new(responder),
            gate: None,
        }
    }

    /// Only consult the responder when `predicate` returns true.
    pub fn with_gate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&MiddlewareCtx) -> bool + Send + Sync + 'static,
    {
        self.gate = Some(Arc::new(predicate));
        self
    }
}

#[async_trait]
impl Middleware for HumanInTheLoop {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let gated_in = self.gate.as_ref().is_none_or(|g| g(&ctx));
        if !gated_in {
            return next.invoke(ctx).await;
        }
        match self.responder.ask(&ctx).await? {
            HumanDecision::Skip => next.invoke(ctx).await,
            HumanDecision::InjectSystem(text) => {
                ctx.messages.insert(0, Message::system(text));
                next.invoke(ctx).await
            }
            HumanDecision::Override(text) => Ok(ChatResponse {
                message: Message::ai(text),
                usage: None,
                finish_reason: "human_override".into(),
                model: "human-in-the-loop".into(),
            }),
        }
    }
    fn name(&self) -> &str {
        "HumanInTheLoop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, RecordingNext};

    #[tokio::test]
    async fn skip_passes_through() {
        let mw = HumanInTheLoop::new(AlwaysSkip);
        let recorder = Arc::new(RecordingNext::new(ok_resp("real")));
        let next: Arc<dyn Next> = recorder.clone();
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "real");
        assert_eq!(recorder.seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn override_short_circuits() {
        let mw = HumanInTheLoop::new(|_ctx: MiddlewareCtx| async {
            Ok(HumanDecision::Override("manual answer".into()))
        });
        let recorder = Arc::new(RecordingNext::new(ok_resp("never reached")));
        let next: Arc<dyn Next> = recorder.clone();
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "manual answer");
        assert_eq!(recorder.seen.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn inject_prepends_system_message() {
        let mw = HumanInTheLoop::new(|_ctx: MiddlewareCtx| async {
            Ok(HumanDecision::InjectSystem("hint from a person".into()))
        });
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        let _ = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await;
        let seen = recorder.seen.lock().unwrap();
        assert!(matches!(seen[0].messages[0], Message::System(_)));
        assert_eq!(seen[0].messages[0].content(), "hint from a person");
    }

    #[tokio::test]
    async fn gate_predicate_filters_when_to_ask() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let asked = Arc::new(AtomicUsize::new(0));
        let a2 = asked.clone();
        let mw = HumanInTheLoop::new(move |_ctx: MiddlewareCtx| {
            let a3 = a2.clone();
            async move {
                a3.fetch_add(1, Ordering::SeqCst);
                Ok(HumanDecision::Skip)
            }
        })
        .with_gate(|ctx| ctx.messages.iter().any(|m| m.content().contains("?")));
        let next: Arc<dyn Next> = Arc::new(RecordingNext::new(ok_resp("ok")));
        // No question mark → not asked.
        let _ = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("statement")],
                    vec![],
                    Default::default(),
                ),
                next.clone(),
            )
            .await;
        assert_eq!(asked.load(Ordering::SeqCst), 0);
        // Question mark → asked.
        let _ = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("really?")], vec![], Default::default()),
                next,
            )
            .await;
        assert_eq!(asked.load(Ordering::SeqCst), 1);
    }
}
