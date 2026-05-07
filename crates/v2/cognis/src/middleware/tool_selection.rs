//! Filter / reorder the tool definitions sent to the LLM per call.
//!
//! Use cases: hide tools the user doesn't have access to, reduce the
//! context cost for large tool sets by sending only relevant tools,
//! enforce a tool budget per request.
//!
//! Customization:
//! - [`ToolSelection::new`] takes any [`ToolFilter`] — implement the
//!   trait or pass a closure.
//! - [`ToolFilter`] sees `(messages, opts, tool_defs)` and returns the
//!   pruned/ordered Vec.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Message, Result};
use cognis2_llm::chat::{ChatOptions, ChatResponse};
use cognis2_llm::tools::ToolDefinition;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable tool filter. Receives the call's full context and returns
/// the (possibly reduced) tool list to actually send.
pub trait ToolFilter: Send + Sync {
    /// Pick which tools to send.
    fn pick(
        &self,
        messages: &[Message],
        opts: &ChatOptions,
        tools: Vec<ToolDefinition>,
    ) -> Vec<ToolDefinition>;
}

/// Closure-based filter.
impl<F> ToolFilter for F
where
    F: Fn(&[Message], &ChatOptions, Vec<ToolDefinition>) -> Vec<ToolDefinition> + Send + Sync,
{
    fn pick(
        &self,
        messages: &[Message],
        opts: &ChatOptions,
        tools: Vec<ToolDefinition>,
    ) -> Vec<ToolDefinition> {
        (self)(messages, opts, tools)
    }
}

/// Stock filter: keep tools whose name appears in the allow list.
pub struct ToolAllowList {
    allowed: Vec<String>,
}

impl ToolAllowList {
    /// Build from a list of allowed names.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: names.into_iter().map(Into::into).collect(),
        }
    }
}

impl ToolFilter for ToolAllowList {
    fn pick(
        &self,
        _messages: &[Message],
        _opts: &ChatOptions,
        tools: Vec<ToolDefinition>,
    ) -> Vec<ToolDefinition> {
        tools
            .into_iter()
            .filter(|t| self.allowed.iter().any(|n| n == &t.name))
            .collect()
    }
}

/// Stock filter: drop tools whose name appears in the deny list.
pub struct ToolDenyList {
    denied: Vec<String>,
}

impl ToolDenyList {
    /// Build from a list of denied names.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            denied: names.into_iter().map(Into::into).collect(),
        }
    }
}

impl ToolFilter for ToolDenyList {
    fn pick(
        &self,
        _messages: &[Message],
        _opts: &ChatOptions,
        tools: Vec<ToolDefinition>,
    ) -> Vec<ToolDefinition> {
        tools
            .into_iter()
            .filter(|t| !self.denied.iter().any(|n| n == &t.name))
            .collect()
    }
}

/// Cap the number of tools sent per call to `n` (preserving order).
pub struct LimitTools(pub usize);

impl ToolFilter for LimitTools {
    fn pick(
        &self,
        _messages: &[Message],
        _opts: &ChatOptions,
        tools: Vec<ToolDefinition>,
    ) -> Vec<ToolDefinition> {
        tools.into_iter().take(self.0).collect()
    }
}

/// Middleware that runs the configured filter on each request.
pub struct ToolSelection {
    filter: Arc<dyn ToolFilter>,
}

impl ToolSelection {
    /// Wrap a filter.
    pub fn new<F: ToolFilter + 'static>(filter: F) -> Self {
        Self {
            filter: Arc::new(filter),
        }
    }
}

#[async_trait]
impl Middleware for ToolSelection {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let original = std::mem::take(&mut ctx.tool_defs);
        ctx.tool_defs = self.filter.pick(&ctx.messages, &ctx.opts, original);
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "ToolSelection"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, RecordingNext};

    fn td(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: name.into(),
            parameters: None,
        }
    }

    #[tokio::test]
    async fn allow_list_keeps_only_listed_tools() {
        let mw = ToolSelection::new(ToolAllowList::new(["b", "c"]));
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        mw.call(
            MiddlewareCtx::new(
                vec![],
                vec![td("a"), td("b"), td("c"), td("d")],
                Default::default(),
            ),
            next,
        )
        .await
        .unwrap();
        let seen = recorder.seen.lock().unwrap();
        let names: Vec<String> = seen[0].tool_defs.iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, vec!["b".to_string(), "c".into()]);
    }

    #[tokio::test]
    async fn deny_list_drops_listed_tools() {
        let mw = ToolSelection::new(ToolDenyList::new(["b"]));
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        mw.call(
            MiddlewareCtx::new(vec![], vec![td("a"), td("b"), td("c")], Default::default()),
            next,
        )
        .await
        .unwrap();
        let seen = recorder.seen.lock().unwrap();
        let names: Vec<String> = seen[0].tool_defs.iter().map(|t| t.name.clone()).collect();
        assert_eq!(names, vec!["a".to_string(), "c".into()]);
    }

    #[tokio::test]
    async fn limit_tools_truncates() {
        let mw = ToolSelection::new(LimitTools(2));
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        mw.call(
            MiddlewareCtx::new(
                vec![],
                vec![td("a"), td("b"), td("c"), td("d")],
                Default::default(),
            ),
            next,
        )
        .await
        .unwrap();
        let seen = recorder.seen.lock().unwrap();
        assert_eq!(seen[0].tool_defs.len(), 2);
    }

    #[tokio::test]
    async fn closure_filter_works() {
        let mw = ToolSelection::new(
            |_msgs: &[Message], _opts: &ChatOptions, defs: Vec<ToolDefinition>| {
                defs.into_iter()
                    .filter(|t| t.name.starts_with('x'))
                    .collect()
            },
        );
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        mw.call(
            MiddlewareCtx::new(
                vec![],
                vec![td("a"), td("xa"), td("xb")],
                Default::default(),
            ),
            next,
        )
        .await
        .unwrap();
        let seen = recorder.seen.lock().unwrap();
        assert_eq!(seen[0].tool_defs.len(), 2);
    }
}
