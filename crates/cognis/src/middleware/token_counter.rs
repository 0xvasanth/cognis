//! Token-counting middleware — observes `Usage` on every response.
//!
//! Doesn't enforce limits; pair with [`super::RateLimit`] for that. This
//! is the metering primitive — feeds an [`UsageTracker`] you read at
//! whatever cadence makes sense (per-request, per-session, per-process).

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;
use cognis_llm::usage::UsageTracker;

use super::{Middleware, MiddlewareCtx, Next};

/// Middleware that records every `Usage` returned by the LLM into a
/// shared [`UsageTracker`].
pub struct TokenCounter {
    tracker: Arc<UsageTracker>,
}

impl TokenCounter {
    /// Build with a shared tracker.
    pub fn new(tracker: Arc<UsageTracker>) -> Self {
        Self { tracker }
    }
}

#[async_trait]
impl Middleware for TokenCounter {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let resp = next.invoke(ctx).await?;
        if let Some(u) = &resp.usage {
            self.tracker.record(u);
        }
        Ok(resp)
    }
    fn name(&self) -> &str {
        "TokenCounter"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_core::Message;
    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn aggregates_usage_across_calls() {
        let tracker = Arc::new(UsageTracker::new());
        let provider = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(TokenCounter::new(tracker.clone()))
            .build(Client::new(provider));
        for _ in 0..3 {
            pipe.invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        }
        // RecordingProvider returns Usage::default() (all zero), but the
        // tracker should still register 3 requests.
        assert_eq!(tracker.requests(), 3);
    }
}
