//! Cap the cumulative number of tool calls the LLM may request.
//!
//! Counts every `tool_call` entry returned across responses; when the
//! running total would exceed `max`, the middleware strips the new
//! tool calls from the response (the LLM's text remains).
//!
//! Customization:
//! - [`ToolCallLimit::with_message`] — the synthetic message inserted
//!   when tool calls are stripped.
//! - [`ToolCallLimit::with_callback`] — fired once when the cap is hit.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{AiMessage, Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

type CapCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Hard cap on tool calls.
pub struct ToolCallLimit {
    max: u64,
    counter: AtomicU64,
    message: String,
    on_cap: Option<CapCallback>,
}

impl ToolCallLimit {
    /// Build with a maximum.
    pub fn new(max: u64) -> Self {
        Self {
            max,
            counter: AtomicU64::new(0),
            message: format!("tool call limit ({max}) reached; further tool calls suppressed"),
            on_cap: None,
        }
    }

    /// Override the message inserted when tool calls are stripped.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    /// Register a callback fired the first time the cap is hit.
    pub fn with_callback<F>(mut self, f: F) -> Self
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        self.on_cap = Some(Arc::new(f));
        self
    }

    /// Current count.
    pub fn count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Reset the counter.
    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl Middleware for ToolCallLimit {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let mut resp = next.invoke(ctx).await?;
        let new_calls = resp.message.tool_calls().len() as u64;
        if new_calls == 0 {
            return Ok(resp);
        }
        let prev = self.counter.fetch_add(new_calls, Ordering::Relaxed);
        if prev >= self.max {
            // Cap already hit — strip all new calls.
            self.strip(&mut resp);
            if let Some(cb) = &self.on_cap {
                cb(self.max);
            }
            return Ok(resp);
        }
        if prev + new_calls > self.max {
            // Partial: keep `max - prev` calls, strip the rest.
            let keep = (self.max - prev) as usize;
            if let Message::Ai(ref mut a) = resp.message {
                a.tool_calls.truncate(keep);
            }
            self.append_message(&mut resp);
            if let Some(cb) = &self.on_cap {
                cb(self.max);
            }
        }
        Ok(resp)
    }
    fn name(&self) -> &str {
        "ToolCallLimit"
    }
}

impl ToolCallLimit {
    fn strip(&self, resp: &mut ChatResponse) {
        if let Message::Ai(ref mut a) = resp.message {
            a.tool_calls.clear();
        }
        self.append_message(resp);
    }

    fn append_message(&self, resp: &mut ChatResponse) {
        let new_text = if resp.message.content().is_empty() {
            self.message.clone()
        } else {
            format!("{}\n\n{}", resp.message.content(), self.message)
        };
        resp.message = Message::Ai(AiMessage {
            content: new_text,
            tool_calls: resp.message.tool_calls().to_vec(),
            parts: Vec::new(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::FixedNext;
    use cognis_core::ToolCall;
    use cognis_llm::chat::ChatResponse;

    fn resp_with_tool_calls(n: usize) -> ChatResponse {
        ChatResponse {
            message: Message::Ai(AiMessage {
                content: String::new(),
                tool_calls: (0..n)
                    .map(|i| ToolCall {
                        id: format!("c{i}"),
                        name: "x".into(),
                        arguments: serde_json::Value::Null,
                    })
                    .collect(),
                parts: Vec::new(),
            }),
            usage: None,
            finish_reason: "tool_calls".into(),
            model: "test".into(),
        }
    }

    #[tokio::test]
    async fn under_cap_passes_through_unchanged() {
        let mw = ToolCallLimit::new(5);
        let next: Arc<dyn Next> = Arc::new(FixedNext(resp_with_tool_calls(2)));
        let r = mw
            .call(MiddlewareCtx::new(vec![], vec![], Default::default()), next)
            .await
            .unwrap();
        assert_eq!(r.message.tool_calls().len(), 2);
    }

    #[tokio::test]
    async fn partial_cap_truncates() {
        let mw = ToolCallLimit::new(3);
        let next: Arc<dyn Next> = Arc::new(FixedNext(resp_with_tool_calls(5)));
        let r = mw
            .call(MiddlewareCtx::new(vec![], vec![], Default::default()), next)
            .await
            .unwrap();
        assert_eq!(r.message.tool_calls().len(), 3);
        assert!(r.message.content().contains("limit"));
    }

    #[tokio::test]
    async fn over_cap_strips_completely() {
        let mw = ToolCallLimit::new(2);
        let next1: Arc<dyn Next> = Arc::new(FixedNext(resp_with_tool_calls(2)));
        let _ = mw
            .call(
                MiddlewareCtx::new(vec![], vec![], Default::default()),
                next1,
            )
            .await;
        // Now we're at 2/2; the next 1 call exceeds.
        let next2: Arc<dyn Next> = Arc::new(FixedNext(resp_with_tool_calls(1)));
        let r = mw
            .call(
                MiddlewareCtx::new(vec![], vec![], Default::default()),
                next2,
            )
            .await
            .unwrap();
        assert_eq!(r.message.tool_calls().len(), 0);
    }

    #[tokio::test]
    async fn callback_fires_when_capped() {
        use std::sync::atomic::AtomicUsize;
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let mw = ToolCallLimit::new(0).with_callback(move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let next: Arc<dyn Next> = Arc::new(FixedNext(resp_with_tool_calls(1)));
        let _ = mw
            .call(MiddlewareCtx::new(vec![], vec![], Default::default()), next)
            .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
