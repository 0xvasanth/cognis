//! Approval gate — gate any chat call (not just tool calls) behind a
//! pluggable [`Approver`].
//!
//! Distinct from `cognis2::tools::ApprovalGatedTool`: that gates a
//! single tool's execution. This middleware gates the **whole LLM
//! call** — useful for cost-sensitive or destructive flows where you
//! want every model call to require explicit approval.
//!
//! Customization:
//! - Implement [`ChatApprover`] for a fully custom approver.
//! - Use [`AutoApproveAll`] / [`AutoRejectAll`] in tests.
//! - Closure-based approvers are supported via blanket impl.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{CognisError, Message, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Approver decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatApproval {
    /// Allow the call.
    Allow,
    /// Reject — middleware errors with a `Configuration` error.
    Reject {
        /// Reason surfaced in the error message.
        reason: String,
    },
}

/// Pluggable approver.
#[async_trait]
pub trait ChatApprover: Send + Sync {
    /// Decide whether the call may proceed.
    async fn decide(&self, ctx: &MiddlewareCtx) -> Result<ChatApproval>;
}

/// Always allow — convenience for tests / opt-in deployments.
#[derive(Debug, Default, Clone, Copy)]
pub struct AutoApproveAll;

#[async_trait]
impl ChatApprover for AutoApproveAll {
    async fn decide(&self, _ctx: &MiddlewareCtx) -> Result<ChatApproval> {
        Ok(ChatApproval::Allow)
    }
}

/// Always reject — convenience for tests / kill-switches.
pub struct AutoRejectAll {
    /// Reason returned for every rejection.
    pub reason: String,
}

impl AutoRejectAll {
    /// Build with a reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl ChatApprover for AutoRejectAll {
    async fn decide(&self, _ctx: &MiddlewareCtx) -> Result<ChatApproval> {
        Ok(ChatApproval::Reject {
            reason: self.reason.clone(),
        })
    }
}

/// Closure-based approver.
#[async_trait]
impl<F, Fut> ChatApprover for F
where
    F: Fn(MiddlewareCtx) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<ChatApproval>> + Send,
{
    async fn decide(&self, ctx: &MiddlewareCtx) -> Result<ChatApproval> {
        (self)(ctx.clone()).await
    }
}

/// Approval-gate middleware.
pub struct ApprovalGate {
    approver: Arc<dyn ChatApprover>,
}

impl ApprovalGate {
    /// Wrap an approver.
    pub fn new<A: ChatApprover + 'static>(approver: A) -> Self {
        Self {
            approver: Arc::new(approver),
        }
    }
}

#[async_trait]
impl Middleware for ApprovalGate {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        match self.approver.decide(&ctx).await? {
            ChatApproval::Allow => next.invoke(ctx).await,
            ChatApproval::Reject { reason } => Err(CognisError::Configuration(format!(
                "approval gate rejected the call: {reason}"
            ))),
        }
    }
    fn name(&self) -> &str {
        "ApprovalGate"
    }
}

// Avoid unused-import lint when `Message` is not needed in tests.
#[allow(dead_code)]
fn _msg_marker(_m: Message) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, FixedNext};

    #[tokio::test]
    async fn allow_passes_through() {
        let mw = ApprovalGate::new(AutoApproveAll);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("ok")));
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("x")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "ok");
    }

    #[tokio::test]
    async fn reject_errors_with_reason() {
        let mw = ApprovalGate::new(AutoRejectAll::new("budget exceeded"));
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("never")));
        let err = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("x")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("budget exceeded"));
    }

    #[tokio::test]
    async fn closure_approver_works() {
        let mw = ApprovalGate::new(|ctx: MiddlewareCtx| async move {
            if ctx.messages.iter().any(|m| m.content().contains("admin")) {
                Ok(ChatApproval::Allow)
            } else {
                Ok(ChatApproval::Reject {
                    reason: "non-admin".into(),
                })
            }
        });
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("ok")));
        let allowed = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("admin: do thing")],
                    vec![],
                    Default::default(),
                ),
                next.clone(),
            )
            .await;
        assert!(allowed.is_ok());
        let denied = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("regular user")],
                    vec![],
                    Default::default(),
                ),
                next,
            )
            .await;
        assert!(denied.is_err());
    }
}
