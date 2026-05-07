//! Built-in PII redactor — common patterns (SSN, credit-card, email, phone).
//!
//! Composes `RegexRedactor` rules. Bring your own [`super::RegexRedactor`]
//! if you need custom patterns or a different replacement scheme.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next, RegexRedactor};

/// Drop-in middleware that redacts common PII before sending to the LLM.
///
/// Replaces:
/// - US SSN (`\d{3}-\d{2}-\d{4}`) → `[SSN]`
/// - Credit card-like 16-digit blocks → `[CC]`
/// - Email addresses → `[EMAIL]`
/// - US-style phone numbers → `[PHONE]`
pub struct PiiRedactor {
    inner: RegexRedactor,
}

impl Default for PiiRedactor {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiRedactor {
    /// Build with the default rule set.
    pub fn new() -> Self {
        let inner = RegexRedactor::new()
            .with_rule(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]")
            .expect("static SSN regex")
            .with_rule(r"\b(?:\d[ -]?){13,19}\b", "[CC]")
            .expect("static CC regex")
            .with_rule(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                "[EMAIL]",
            )
            .expect("static EMAIL regex")
            .with_rule(r"(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]\d{4}", "[PHONE]")
            .expect("static PHONE regex");
        Self { inner }
    }
}

#[async_trait]
impl Middleware for PiiRedactor {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        self.inner.call(ctx, next).await
    }

    fn name(&self) -> &str {
        "PiiRedactor"
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
    async fn redacts_email_and_ssn_and_phone() {
        let rec = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(PiiRedactor::new())
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(
                vec![Message::human(
                    "ssn 123-45-6789 email a@b.com phone (555) 123-4567",
                )],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        let m = &received[0].0[0];
        assert!(m.content().contains("[SSN]"));
        assert!(m.content().contains("[EMAIL]"));
        assert!(m.content().contains("[PHONE]"));
        assert!(!m.content().contains("a@b.com"));
        assert!(!m.content().contains("123-45-6789"));
    }
}
