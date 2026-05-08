//! Human-in-the-loop tool approval.
//!
//! [`ApprovalGatedTool`] wraps any [`Tool`] and asks an [`Approver`] before
//! executing. Three built-in approvers ship; users can plug in their own
//! (Slack bot, web UI, Linear ticket, etc.) by implementing the trait.
//!
//! # Why a tool wrapper, not a middleware?
//!
//! V1 had this as an agent-level middleware with an event bus + token
//! registry + oneshot resolvers. That's a lot of machinery for "ask before
//! running this thing." A wrapper is composable per-tool, has zero global
//! state, and works through the same dispatch path the agent already uses.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

/// What a human (or surrogate) decided about a pending tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Decision {
    /// Approve the call as-is.
    Approve,
    /// Reject the call. The reason is surfaced to the agent as the tool's
    /// output so the model can react.
    Reject {
        /// Why the call was rejected.
        reason: String,
    },
    /// Approve, but with edited arguments. The wrapped tool is invoked
    /// with `args` instead of the LLM-supplied input.
    Edit {
        /// Replacement arguments (any JSON shape).
        args: serde_json::Value,
    },
}

impl Decision {
    /// Convenience: `Decision::reject("nope")` instead of the verbose form.
    pub fn reject(reason: impl Into<String>) -> Self {
        Self::Reject {
            reason: reason.into(),
        }
    }
}

/// Decides whether a pending tool call should run.
#[async_trait]
pub trait Approver: Send + Sync {
    /// Inspect a pending tool call and return a [`Decision`].
    async fn approve(&self, tool_name: &str, args: &serde_json::Value) -> Result<Decision>;
}

/// Always approves. Useful when you want the wrapper structure but no actual gating.
pub struct AutoApprove;

#[async_trait]
impl Approver for AutoApprove {
    async fn approve(&self, _: &str, _: &serde_json::Value) -> Result<Decision> {
        Ok(Decision::Approve)
    }
}

/// Always rejects with a fixed reason. Safe default for production where
/// the operator must wire in a real approver.
pub struct RejectAll {
    reason: String,
}

impl RejectAll {
    /// Build with a reject reason.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl Default for RejectAll {
    fn default() -> Self {
        Self::new("approval required but no approver configured")
    }
}

#[async_trait]
impl Approver for RejectAll {
    async fn approve(&self, _: &str, _: &serde_json::Value) -> Result<Decision> {
        Ok(Decision::reject(self.reason.clone()))
    }
}

/// Approver that approves only the listed tools.
pub struct AllowList {
    allowed: Vec<String>,
}

impl AllowList {
    /// Build from an iterator of tool names.
    pub fn new<I, S>(allowed: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl Approver for AllowList {
    async fn approve(&self, tool_name: &str, _: &serde_json::Value) -> Result<Decision> {
        if self.allowed.iter().any(|n| n == tool_name) {
            Ok(Decision::Approve)
        } else {
            Ok(Decision::reject(format!(
                "tool `{tool_name}` is not on the allow list"
            )))
        }
    }
}

/// Wraps a `Tool` so it asks an [`Approver`] before each invocation.
pub struct ApprovalGatedTool {
    inner: Arc<dyn Tool>,
    approver: Arc<dyn Approver>,
}

impl ApprovalGatedTool {
    /// Build the wrapper.
    pub fn new(inner: Arc<dyn Tool>, approver: Arc<dyn Approver>) -> Self {
        Self { inner, approver }
    }
}

#[async_trait]
impl Tool for ApprovalGatedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        self.inner.args_schema()
    }
    fn return_direct(&self) -> bool {
        self.inner.return_direct()
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args_json = input.clone().into_json();
        let decision = self.approver.approve(self.inner.name(), &args_json).await?;
        match decision {
            Decision::Approve => self.inner._run(input).await,
            Decision::Reject { reason } => Err(CognisError::Tool {
                name: self.inner.name().to_string(),
                reason: format!("rejected by approver: {reason}"),
            }),
            Decision::Edit { args } => {
                let edited = if let serde_json::Value::Object(m) = args {
                    let map: std::collections::HashMap<String, serde_json::Value> =
                        m.into_iter().collect();
                    ToolInput::Structured(map)
                } else {
                    ToolInput::Text(args.to_string())
                };
                self.inner._run(edited).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            None
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Content(input.into_json()))
        }
    }

    #[tokio::test]
    async fn auto_approve_passes_through() {
        let t = ApprovalGatedTool::new(Arc::new(Echo), Arc::new(AutoApprove));
        let out = t._run(ToolInput::Text("hi".into())).await.unwrap();
        assert_eq!(out.as_string(), "\"hi\"");
    }

    #[tokio::test]
    async fn reject_all_blocks() {
        let t = ApprovalGatedTool::new(Arc::new(Echo), Arc::new(RejectAll::default()));
        let err = t._run(ToolInput::Text("hi".into())).await.unwrap_err();
        assert!(format!("{err}").contains("rejected"));
    }

    #[tokio::test]
    async fn allow_list_filters_by_tool_name() {
        let allow = ApprovalGatedTool::new(Arc::new(Echo), Arc::new(AllowList::new(["echo"])));
        assert!(allow._run(ToolInput::Text("a".into())).await.is_ok());
        let block = ApprovalGatedTool::new(Arc::new(Echo), Arc::new(AllowList::new(["other"])));
        assert!(block._run(ToolInput::Text("a".into())).await.is_err());
    }

    /// Approver that always returns Edit with new args.
    struct Editor;
    #[async_trait]
    impl Approver for Editor {
        async fn approve(&self, _: &str, _: &serde_json::Value) -> Result<Decision> {
            Ok(Decision::Edit {
                args: serde_json::json!({"replaced": true}),
            })
        }
    }

    #[tokio::test]
    async fn edit_substitutes_args() {
        let t = ApprovalGatedTool::new(Arc::new(Echo), Arc::new(Editor));
        let out = t._run(ToolInput::Text("ignored".into())).await.unwrap();
        let v: serde_json::Value = match out {
            ToolOutput::Content(v) => v,
            _ => panic!(),
        };
        assert_eq!(v["replaced"], true);
    }
}
