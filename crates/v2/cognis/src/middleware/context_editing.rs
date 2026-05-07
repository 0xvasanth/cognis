//! Context editing — rewrite messages mid-run via a pluggable
//! [`EditPolicy`].
//!
//! Useful for: redaction passes that go beyond regex (LLM-based),
//! summarization that compresses certain message types but not others,
//! style normalization, length capping per message.
//!
//! Customization: implement [`EditPolicy`] for a custom rewrite, or
//! pass a closure (blanket impl).

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Message, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable message editor.
pub trait EditPolicy: Send + Sync {
    /// Receive the call's full message list and return the edited
    /// version. Implementations may shorten / remove / add messages.
    fn edit(&self, messages: Vec<Message>) -> Vec<Message>;
}

/// Closure-based policy.
impl<F> EditPolicy for F
where
    F: Fn(Vec<Message>) -> Vec<Message> + Send + Sync,
{
    fn edit(&self, messages: Vec<Message>) -> Vec<Message> {
        (self)(messages)
    }
}

/// Stock policy: cap each message's content to `max_chars` characters.
pub struct CapMessageLength {
    /// Max characters per message.
    pub max_chars: usize,
}

impl EditPolicy for CapMessageLength {
    fn edit(&self, messages: Vec<Message>) -> Vec<Message> {
        messages
            .into_iter()
            .map(|m| match m {
                Message::Human(mut h) => {
                    if h.content.chars().count() > self.max_chars {
                        h.content = h.content.chars().take(self.max_chars).collect();
                    }
                    Message::Human(h)
                }
                Message::Ai(mut a) => {
                    if a.content.chars().count() > self.max_chars {
                        a.content = a.content.chars().take(self.max_chars).collect();
                    }
                    Message::Ai(a)
                }
                Message::System(mut s) => {
                    if s.content.chars().count() > self.max_chars {
                        s.content = s.content.chars().take(self.max_chars).collect();
                    }
                    Message::System(s)
                }
                Message::Tool(mut t) => {
                    if t.content.chars().count() > self.max_chars {
                        t.content = t.content.chars().take(self.max_chars).collect();
                    }
                    Message::Tool(t)
                }
            })
            .collect()
    }
}

/// Stock policy: drop messages whose content matches `predicate`.
pub struct DropMatching {
    predicate: Arc<dyn Fn(&Message) -> bool + Send + Sync>,
}

impl DropMatching {
    /// Build from a predicate.
    pub fn new<F>(predicate: F) -> Self
    where
        F: Fn(&Message) -> bool + Send + Sync + 'static,
    {
        Self {
            predicate: Arc::new(predicate),
        }
    }
}

impl EditPolicy for DropMatching {
    fn edit(&self, messages: Vec<Message>) -> Vec<Message> {
        messages
            .into_iter()
            .filter(|m| !(self.predicate)(m))
            .collect()
    }
}

/// Middleware that runs the editor on every request before the inner call.
pub struct ContextEditing {
    policy: Arc<dyn EditPolicy>,
}

impl ContextEditing {
    /// Wrap a policy.
    pub fn new<P: EditPolicy + 'static>(policy: P) -> Self {
        Self {
            policy: Arc::new(policy),
        }
    }
}

#[async_trait]
impl Middleware for ContextEditing {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let original = std::mem::take(&mut ctx.messages);
        ctx.messages = self.policy.edit(original);
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "ContextEditing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, RecordingNext};

    #[tokio::test]
    async fn cap_message_length_truncates() {
        let mw = ContextEditing::new(CapMessageLength { max_chars: 5 });
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        let _ = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("longer than five chars")],
                    vec![],
                    Default::default(),
                ),
                next,
            )
            .await;
        let seen = recorder.seen.lock().unwrap();
        assert_eq!(seen[0].messages[0].content().chars().count(), 5);
    }

    #[tokio::test]
    async fn drop_matching_removes() {
        let mw = ContextEditing::new(DropMatching::new(|m| m.content().contains("REDACT")));
        let recorder = Arc::new(RecordingNext::new(ok_resp("ok")));
        let next: Arc<dyn Next> = recorder.clone();
        let _ = mw
            .call(
                MiddlewareCtx::new(
                    vec![
                        Message::human("keep me"),
                        Message::human("REDACT THIS"),
                        Message::human("also keep"),
                    ],
                    vec![],
                    Default::default(),
                ),
                next,
            )
            .await;
        let seen = recorder.seen.lock().unwrap();
        assert_eq!(seen[0].messages.len(), 2);
        assert!(seen[0]
            .messages
            .iter()
            .all(|m| !m.content().contains("REDACT")));
    }

    #[tokio::test]
    async fn closure_policy_works() {
        let mw = ContextEditing::new(|msgs: Vec<Message>| {
            // Add a system pin in front.
            let mut out = vec![Message::system("policy injected")];
            out.extend(msgs);
            out
        });
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
    }
}
