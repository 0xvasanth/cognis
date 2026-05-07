//! Regex-based redaction. Replace matching substrings in outgoing message
//! content with a fixed token before they hit the LLM.
//!
//! For more structured PII detection, see [`super::PiiRedactor`].

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{
    AiMessage, CognisError, HumanMessage, Message, Result, SystemMessage, ToolMessage,
};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pre-compiled `(regex, replacement)` pairs applied to every outgoing
/// message's text content.
pub struct RegexRedactor {
    rules: Vec<(regex_lite::Regex, String)>,
}

impl RegexRedactor {
    /// Empty redactor.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule: any match of `pattern` is replaced with `replacement`.
    pub fn with_rule(mut self, pattern: &str, replacement: impl Into<String>) -> Result<Self> {
        let re = regex_lite::Regex::new(pattern)
            .map_err(|e| CognisError::Configuration(format!("regex compile: {e}")))?;
        self.rules.push((re, replacement.into()));
        Ok(self)
    }
}

impl Default for RegexRedactor {
    fn default() -> Self {
        Self::new()
    }
}

fn redact_string(s: &str, rules: &[(regex_lite::Regex, String)]) -> String {
    let mut out = s.to_string();
    for (re, rep) in rules {
        out = re.replace_all(&out, rep.as_str()).into_owned();
    }
    out
}

fn redact_message(m: Message, rules: &[(regex_lite::Regex, String)]) -> Message {
    match m {
        Message::Human(HumanMessage { content, parts }) => Message::Human(HumanMessage {
            content: redact_string(&content, rules),
            parts,
        }),
        Message::System(SystemMessage { content }) => Message::System(SystemMessage {
            content: redact_string(&content, rules),
        }),
        Message::Ai(AiMessage {
            content,
            tool_calls,
            parts,
        }) => Message::Ai(AiMessage {
            content: redact_string(&content, rules),
            tool_calls,
            parts,
        }),
        Message::Tool(ToolMessage {
            tool_call_id,
            content,
        }) => Message::Tool(ToolMessage {
            tool_call_id,
            content: redact_string(&content, rules),
        }),
    }
}

#[async_trait]
impl Middleware for RegexRedactor {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        if !self.rules.is_empty() {
            ctx.messages = ctx
                .messages
                .into_iter()
                .map(|m| redact_message(m, &self.rules))
                .collect();
        }
        next.invoke(ctx).await
    }

    fn name(&self) -> &str {
        "RegexRedactor"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis2_llm::chat::ChatOptions;
    use cognis2_llm::Client;

    #[tokio::test]
    async fn redacts_human_messages_before_llm_sees_them() {
        let rec = make_recording_provider("ok");
        let provider = rec.clone();
        let pipe = MiddlewarePipeline::new()
            .push(
                RegexRedactor::new()
                    .with_rule(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]")
                    .unwrap(),
            )
            .build(Client::new(provider));
        let _ = pipe
            .invoke(
                vec![Message::human("my SSN is 123-45-6789 thanks")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        assert_eq!(received.len(), 1);
        let msg = &received[0].0[0];
        assert!(msg.content().contains("[SSN]"));
        assert!(!msg.content().contains("123-45-6789"));
    }
}
