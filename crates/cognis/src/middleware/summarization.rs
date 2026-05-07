//! Summarization middleware — when conversation history exceeds a configurable
//! length, replace the older portion with a fixed system "summary placeholder"
//! note before sending to the LLM.
//!
//! This is a *defensive* middleware that prevents context blow-up. It does
//! NOT call an LLM to produce the summary; for an LLM-driven summarizer,
//! plug in a custom Middleware that uses a `Client` to compress history.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Trim conversation history to the most recent `keep_last` messages,
/// inserting a system "summary placeholder" message at the head if any
/// messages were trimmed.
pub struct Summarization {
    keep_last: usize,
    placeholder: String,
}

impl Summarization {
    /// Keep the most recent `keep_last` messages (excluding the very first
    /// system message, if any). 0 means trim aggressively.
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last,
            placeholder: "[earlier conversation truncated for length]".into(),
        }
    }

    /// Override the placeholder text.
    pub fn with_placeholder(mut self, s: impl Into<String>) -> Self {
        self.placeholder = s.into();
        self
    }
}

fn trim(messages: Vec<Message>, keep_last: usize, placeholder: &str) -> Vec<Message> {
    // Preserve a leading system prompt — common pattern is `[system, ...rest]`.
    let mut head: Vec<Message> = Vec::new();
    let mut rest = messages;
    if let Some(first) = rest.first() {
        if matches!(first, Message::System(_)) {
            head.push(rest.remove(0));
        }
    }
    if rest.len() <= keep_last {
        head.extend(rest);
        return head;
    }
    let trim_count = rest.len() - keep_last;
    let _ = rest.drain(..trim_count);
    let mut out = head;
    out.push(Message::system(format!(
        "{placeholder} ({trim_count} messages omitted)"
    )));
    out.extend(rest);
    out
}

#[async_trait]
impl Middleware for Summarization {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        ctx.messages = trim(ctx.messages, self.keep_last, &self.placeholder);
        next.invoke(ctx).await
    }

    fn name(&self) -> &str {
        "Summarization"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    fn convo(n: usize) -> Vec<Message> {
        let mut out = vec![Message::system("you are helpful")];
        for i in 0..n {
            out.push(Message::human(format!("u{i}")));
            out.push(Message::ai(format!("a{i}")));
        }
        out
    }

    #[tokio::test]
    async fn passes_through_under_keep_last() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(Summarization::new(10))
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(convo(2), Vec::new(), ChatOptions::default())
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        // 1 system + 2 turns × 2 = 5 messages, all preserved.
        assert_eq!(received[0].0.len(), 5);
    }

    #[tokio::test]
    async fn trims_when_over_threshold_and_inserts_placeholder() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(Summarization::new(2))
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(convo(5), Vec::new(), ChatOptions::default())
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        let msgs = &received[0].0;
        // [system_prompt, summary_placeholder, last 2 messages]
        assert_eq!(msgs.len(), 4);
        assert!(matches!(msgs[0], Message::System(_)));
        assert!(matches!(msgs[1], Message::System(_)));
        assert!(msgs[1].content().contains("truncated"));
    }
}
