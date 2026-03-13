//! Summarization middleware — manages context window by summarizing older messages.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};

use crate::middleware::{AgentState, Middleware, Result};

/// Default prompt used to request a conversation summary from the model.
const DEFAULT_SUMMARY_PROMPT: &str =
    "Summarize the following conversation concisely, preserving all key information, decisions, and context needed to continue the conversation:\n\n";

/// Middleware that automatically summarizes older messages when the conversation
/// grows beyond a configured threshold.
///
/// When the number of messages in the state exceeds `max_messages`, this middleware:
/// 1. Takes all messages except the most recent `keep_recent` messages.
/// 2. Sends them to the summarization model with a summary prompt.
/// 3. Replaces the old messages with a single `SystemMessage` containing the summary.
/// 4. Keeps the recent messages intact.
pub struct SummarizationMiddleware {
    /// The chat model used for summarization.
    model: Arc<dyn BaseChatModel>,
    /// Trigger summarization when message count exceeds this value.
    pub max_messages: usize,
    /// Number of recent messages to always keep intact.
    pub keep_recent: usize,
    /// Prompt template prepended to the messages when requesting a summary.
    pub summary_prompt: String,
}

impl SummarizationMiddleware {
    /// Create a new `SummarizationMiddleware` with the given model and default settings.
    ///
    /// Defaults: `max_messages = 20`, `keep_recent = 5`.
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        Self {
            model,
            max_messages: 20,
            keep_recent: 5,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Set the maximum number of messages before summarization is triggered.
    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = max_messages;
        self
    }

    /// Set the number of recent messages to keep intact during summarization.
    pub fn with_keep_recent(mut self, keep_recent: usize) -> Self {
        self.keep_recent = keep_recent;
        self
    }

    /// Set a custom summary prompt.
    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl Middleware for SummarizationMiddleware {
    fn name(&self) -> &str {
        "summarization"
    }

    /// Before the model is invoked, check if the message count exceeds
    /// `max_messages`. If so, summarize the older messages and replace
    /// them with a single system message.
    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let messages = match state.get("messages").and_then(|v| v.as_array()) {
            Some(arr) => arr.clone(),
            None => return Ok(()),
        };

        let msg_count = messages.len();
        if msg_count <= self.max_messages {
            return Ok(());
        }

        // Split into old messages (to summarize) and recent messages (to keep).
        let split_point = msg_count.saturating_sub(self.keep_recent);
        let old_messages = &messages[..split_point];
        let recent_messages = &messages[split_point..];

        // Build the summarization request.
        let mut conversation_text = String::new();
        for msg_value in old_messages {
            let role = msg_value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let content = msg_value
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            conversation_text.push_str(&format!("{role}: {content}\n"));
        }

        let prompt_text = format!("{}{}", self.summary_prompt, conversation_text);

        let summarization_messages = vec![Message::Human(HumanMessage::new(&prompt_text))];

        let summary_result = self
            .model
            ._generate(&summarization_messages, None)
            .await
            .map_err(|e| {
                crate::agent::DeepAgentError::MiddlewareError(format!(
                    "Summarization model call failed: {e}"
                ))
            })?;

        let summary_text = summary_result
            .generations
            .first()
            .map(|g| g.message.content().text())
            .unwrap_or_default();

        // Build a new messages array: summary system message + recent messages.
        let summary_msg = json!({
            "type": "system",
            "content": format!("## Conversation Summary\n{summary_text}")
        });

        let mut new_messages = vec![summary_msg];
        new_messages.extend(recent_messages.iter().cloned());

        state["messages"] = Value::Array(new_messages);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::FakeMessagesListChatModel;
    use cognis_core::messages::AIMessage;

    /// Helper: build a state with `n` human messages.
    fn make_state(n: usize) -> Value {
        let messages: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "type": "human",
                    "content": format!("Message {i}")
                })
            })
            .collect();
        json!({ "messages": messages })
    }

    #[tokio::test]
    async fn test_messages_under_limit_not_summarized() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("This should not be called"),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(10)
            .with_keep_recent(3);

        let mut state = make_state(5); // 5 < 10, no summarization
        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5); // unchanged
    }

    #[tokio::test]
    async fn test_messages_over_limit_get_summarized() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Here is the summary of the conversation."),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(5)
            .with_keep_recent(2);

        let mut state = make_state(8); // 8 > 5, should summarize
        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // 1 summary system message + 2 recent = 3
        assert_eq!(messages.len(), 3);

        // First message is the summary.
        let summary = &messages[0];
        assert_eq!(summary["type"].as_str().unwrap(), "system");
        assert!(summary["content"]
            .as_str()
            .unwrap()
            .contains("summary of the conversation"));

        // Recent messages preserved.
        assert_eq!(messages[1]["content"].as_str().unwrap(), "Message 6");
        assert_eq!(messages[2]["content"].as_str().unwrap(), "Message 7");
    }

    #[tokio::test]
    async fn test_recent_messages_preserved_intact() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Summary content"),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(3)
            .with_keep_recent(2);

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "old1"},
                {"type": "human", "content": "old2"},
                {"type": "human", "content": "old3"},
                {"type": "human", "content": "recent1"},
                {"type": "human", "content": "recent2"}
            ]
        });

        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3); // 1 summary + 2 recent

        // Recent messages must be exactly the last 2.
        assert_eq!(messages[1]["content"].as_str().unwrap(), "recent1");
        assert_eq!(messages[2]["content"].as_str().unwrap(), "recent2");
    }

    #[tokio::test]
    async fn test_summarization_middleware_name() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("x"),
        )]));
        let mw = SummarizationMiddleware::new(model);
        assert_eq!(mw.name(), "summarization");
    }

    #[tokio::test]
    async fn test_no_messages_key_is_noop() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("x"),
        )]));
        let mw = SummarizationMiddleware::new(model);
        let mut state = json!({});
        mw.before_model(&mut state).await.unwrap();
        assert!(state.get("messages").is_none());
    }

    #[tokio::test]
    async fn test_exact_limit_not_summarized() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("should not be called"),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(5)
            .with_keep_recent(2);

        let mut state = make_state(5); // exactly at limit
        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5); // unchanged
    }

    #[tokio::test]
    async fn test_custom_summary_prompt() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Custom summary"),
        )]));

        let mw = SummarizationMiddleware::new(model)
            .with_max_messages(2)
            .with_keep_recent(1)
            .with_summary_prompt("Please provide a brief summary:\n");

        let mut state = make_state(4);
        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        // 1 summary + 1 recent = 2
        assert_eq!(messages.len(), 2);
    }
}
