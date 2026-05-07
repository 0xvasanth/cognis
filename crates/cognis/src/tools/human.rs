//! Human-in-the-loop tool — pause an agent run to ask a human a free-form
//! question and feed the reply back into the conversation.
//!
//! Distinct from:
//! - [`crate::middleware::HumanInTheLoop`] — middleware that pauses the
//!   chat call, not exposed as a tool.
//! - [`crate::tools::ApprovalGatedTool`] — wraps another tool with
//!   yes/no approval. This `HumanTool` is a tool the LLM itself can
//!   call when it wants to ask the operator a question.
//!
//! Customization: implement [`HumanResponder`] for terminal / Slack /
//! web-inbox prompts. The default [`StaticResponder`] returns a
//! pre-canned reply — useful for tests and dev fixtures.

use std::sync::Arc;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

/// Pluggable strategy for asking a human and getting a reply.
#[async_trait]
pub trait HumanResponder: Send + Sync {
    /// Ask `question` and await the human's text response.
    async fn ask(&self, question: &str) -> Result<String>;
}

/// Closure-based responder.
#[async_trait]
impl<F, Fut> HumanResponder for F
where
    F: Fn(String) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<String>> + Send,
{
    async fn ask(&self, question: &str) -> Result<String> {
        (self)(question.to_string()).await
    }
}

/// Always returns a pre-canned reply. Useful for tests / golden runs.
pub struct StaticResponder {
    /// The reply returned for every question.
    pub reply: String,
}

impl StaticResponder {
    /// Build with the canned reply.
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            reply: reply.into(),
        }
    }
}

#[async_trait]
impl HumanResponder for StaticResponder {
    async fn ask(&self, _question: &str) -> Result<String> {
        Ok(self.reply.clone())
    }
}

/// Tool input.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HumanToolInput {
    /// The question to ask the human.
    pub question: String,
    /// Optional context to help the human answer.
    #[serde(default)]
    pub context: Option<String>,
}

/// Tool that asks a human a question and returns the reply.
pub struct HumanTool {
    responder: Arc<dyn HumanResponder>,
    name: String,
    description: String,
}

impl HumanTool {
    /// Build with a responder and the default name `"ask_human"`.
    pub fn new<R: HumanResponder + 'static>(responder: R) -> Self {
        Self {
            responder: Arc::new(responder),
            name: "ask_human".into(),
            description: "Ask the human operator a free-form question. Use sparingly — \
                 only when you genuinely need clarification."
                .into(),
        }
    }

    /// Override the registered tool name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the description shown to the LLM.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
}

#[async_trait]
impl Tool for HumanTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(HumanToolInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: HumanToolInput = serde_json::from_value(input.into_json()).map_err(|e| {
            CognisError::ToolValidationError(format!("ask_human: invalid args: {e}"))
        })?;
        let prompt = match parsed.context.as_deref() {
            Some(ctx) if !ctx.is_empty() => {
                format!("[context: {ctx}]\n\n{q}", q = parsed.question)
            }
            _ => parsed.question.clone(),
        };
        let reply = self.responder.ask(&prompt).await?;
        Ok(ToolOutput::Text(reply))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn static_responder_returns_canned_reply() {
        let t = HumanTool::new(StaticResponder::new("yes, proceed"));
        let mut m = HashMap::new();
        m.insert("question".to_string(), json!("are you sure?"));
        let out = t._run(ToolInput::Structured(m)).await.unwrap();
        match out {
            ToolOutput::Text(s) => assert_eq!(s, "yes, proceed"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn closure_responder_receives_prompt() {
        use std::sync::Mutex;
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let c2 = captured.clone();
        let t = HumanTool::new(move |q: String| {
            let c3 = c2.clone();
            async move {
                *c3.lock().unwrap() = Some(q);
                Ok("ok".into())
            }
        });
        let mut m = HashMap::new();
        m.insert("question".to_string(), json!("hi?"));
        m.insert("context".to_string(), json!("user is logged in"));
        let _ = t._run(ToolInput::Structured(m)).await.unwrap();
        let seen = captured.lock().unwrap().clone().unwrap();
        assert!(seen.contains("hi?"));
        assert!(seen.contains("user is logged in"));
    }

    #[tokio::test]
    async fn custom_name_and_description() {
        let t = HumanTool::new(StaticResponder::new("y"))
            .with_name("ask_user")
            .with_description("custom description");
        assert_eq!(t.name(), "ask_user");
        assert_eq!(t.description(), "custom description");
    }

    #[tokio::test]
    async fn missing_question_errors() {
        let t = HumanTool::new(StaticResponder::new("ignored"));
        let res = t._run(ToolInput::Structured(HashMap::new())).await;
        assert!(res.is_err());
    }
}
