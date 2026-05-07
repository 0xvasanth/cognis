//! Cache LLM responses keyed by the (messages, tool_defs, opts) triple.
//!
//! This is identical-prompt caching, not Anthropic-style token-prefix
//! caching — that's a provider-level concept. For prefix caching, set the
//! relevant ChatOptions / per-message cache marker via your provider's
//! native API.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Memoizes responses keyed by serialized `(messages, tool_defs, opts)`.
///
/// Cache lives in-process; for cross-process caching, use the inner
/// `RunnableExt::with_memory_cache` on a wrapped Runnable instead.
pub struct PromptCaching {
    cache: Arc<RwLock<HashMap<String, ChatResponse>>>,
}

impl Default for PromptCaching {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptCaching {
    /// Empty cache.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn key(ctx: &MiddlewareCtx) -> String {
        // Best-effort stable key; we serialize into a single JSON string.
        let v = serde_json::json!({
            "messages": ctx.messages,
            "tools": ctx.tool_defs,
            "opts": ctx.opts,
        });
        v.to_string()
    }
}

#[async_trait]
impl Middleware for PromptCaching {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let key = Self::key(&ctx);
        if let Some(hit) = self.cache.read().await.get(&key).cloned() {
            return Ok(hit);
        }
        let resp = next.invoke(ctx).await?;
        self.cache.write().await.insert(key, resp.clone());
        Ok(resp)
    }

    fn name(&self) -> &str {
        "PromptCaching"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use cognis_core::Message;
    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn second_identical_call_hits_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cs = calls.clone();
        let provider = make_flaky_provider(move |_| {
            cs.fetch_add(1, Ordering::SeqCst);
            Ok("response".into())
        });
        let pipe = MiddlewarePipeline::new()
            .push(PromptCaching::new())
            .build(Client::new(provider));
        let _ = pipe
            .invoke(
                vec![Message::human("same")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let _ = pipe
            .invoke(
                vec![Message::human("same")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn different_input_skips_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cs = calls.clone();
        let provider = make_flaky_provider(move |_| {
            cs.fetch_add(1, Ordering::SeqCst);
            Ok("response".into())
        });
        let pipe = MiddlewarePipeline::new()
            .push(PromptCaching::new())
            .build(Client::new(provider));
        let _ = pipe
            .invoke(
                vec![Message::human("a")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let _ = pipe
            .invoke(
                vec![Message::human("b")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
