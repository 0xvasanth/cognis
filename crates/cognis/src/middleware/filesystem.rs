//! Filesystem middleware — injects a system-prompt fragment that
//! announces filesystem capabilities to the LLM and (optionally) seeds
//! the workspace listing into the request.
//!
//! Distinct from the filesystem **tools** (`tools::filesystem`): the
//! tools execute filesystem operations; this middleware advertises
//! them via the prompt and surfaces workspace state.
//!
//! Customization:
//! - [`FilesystemMiddleware::with_prompt`] — override the announcement
//!   prompt.
//! - [`FilesystemMiddleware::with_lister`] — plug in a custom listing
//!   strategy (e.g. depth-limited, glob-filtered, async-disk).

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable workspace lister. Returns a string the middleware will
/// append to the system prompt — `""` for "no listing".
#[async_trait]
pub trait WorkspaceLister: Send + Sync {
    /// Produce a workspace summary at the time of the call.
    async fn list(&self) -> Result<String>;
}

/// Closure-based lister.
#[async_trait]
impl<F, Fut> WorkspaceLister for F
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<String>> + Send,
{
    async fn list(&self) -> Result<String> {
        (self)().await
    }
}

/// Filesystem-announcement middleware.
pub struct FilesystemMiddleware {
    prompt: String,
    lister: Option<Arc<dyn WorkspaceLister>>,
}

const DEFAULT_FS_PROMPT: &str =
    "You have read/write access to a workspace via the filesystem tools \
(read, write, edit, list, glob, grep). Prefer them over guessing file contents.";

impl Default for FilesystemMiddleware {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_FS_PROMPT.to_string(),
            lister: None,
        }
    }
}

impl FilesystemMiddleware {
    /// Default middleware (announcement only, no listing).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the announcement prompt.
    pub fn with_prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = p.into();
        self
    }

    /// Attach a workspace lister whose output is appended to the prompt.
    pub fn with_lister<L: WorkspaceLister + 'static>(mut self, l: L) -> Self {
        self.lister = Some(Arc::new(l));
        self
    }
}

#[async_trait]
impl Middleware for FilesystemMiddleware {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let mut prompt = self.prompt.clone();
        if let Some(l) = &self.lister {
            let listing = l.list().await?;
            if !listing.trim().is_empty() {
                prompt.push_str("\n\nWorkspace:\n");
                prompt.push_str(&listing);
            }
        }
        ctx.messages.insert(0, Message::system(prompt));
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "FilesystemMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, RecordingNext};

    #[tokio::test]
    async fn injects_default_prompt() {
        let mw = FilesystemMiddleware::new();
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
        assert!(seen[0].messages[0].content().contains("filesystem tools"));
    }

    #[tokio::test]
    async fn lister_appended_to_prompt() {
        let mw = FilesystemMiddleware::new()
            .with_prompt("FS available")
            .with_lister(|| async { Ok("- file1.txt\n- file2.txt".to_string()) });
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        let _ = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await;
        let seen = recorder.seen.lock().unwrap();
        assert!(seen[0].messages[0].content().contains("file1.txt"));
        assert!(seen[0].messages[0].content().contains("FS available"));
    }

    #[tokio::test]
    async fn empty_lister_output_omits_workspace_section() {
        let mw = FilesystemMiddleware::new().with_lister(|| async { Ok("".to_string()) });
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        let _ = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await;
        let seen = recorder.seen.lock().unwrap();
        assert!(!seen[0].messages[0].content().contains("Workspace"));
    }
}
