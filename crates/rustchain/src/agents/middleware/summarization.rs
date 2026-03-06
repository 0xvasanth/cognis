//! Summarization middleware — context window management.
//!
//! Mirrors Python `langchain.agents.middleware.summarization`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::Message;

use super::types::{AgentMiddleware, AgentState};

/// How to specify context size thresholds.
#[derive(Debug, Clone)]
pub enum ContextSize {
    /// Fraction of the model's max context (0.0 to 1.0).
    Fraction(f64),
    /// Absolute number of tokens.
    Tokens(usize),
    /// Number of messages.
    Messages(usize),
}

/// Configuration for summarization behavior.
#[derive(Debug, Clone)]
pub struct SummarizationConfig {
    /// When to trigger summarization.
    pub trigger: ContextSize,
    /// How much context to preserve after summarization.
    pub keep: ContextSize,
    /// System prompt for the summarization model.
    pub summary_prompt: String,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            trigger: ContextSize::Fraction(0.75),
            keep: ContextSize::Messages(10),
            summary_prompt: "You are a conversation summarizer. Summarize the following conversation \
                concisely but thoroughly. Preserve all key information including: decisions made, \
                tool results, important facts, user preferences, and any pending tasks or open questions. \
                Focus on information that would be needed to continue the conversation effectively."
                .into(),
        }
    }
}

/// Middleware that summarizes conversation history when token limits are approached.
///
/// When the conversation exceeds the configured trigger threshold, the middleware
/// splits messages into "to summarize" and "to keep" portions. If a model is
/// provided, it uses the LLM to generate a proper summary. Otherwise, it falls
/// back to concatenating message text.
#[derive(Default)]
pub struct SummarizationMiddleware {
    pub config: SummarizationConfig,
    /// Optional model to use for generating summaries via LLM.
    pub model: Option<Arc<dyn BaseChatModel>>,
}

impl SummarizationMiddleware {
    pub fn new(config: SummarizationConfig) -> Self {
        Self {
            config,
            model: None,
        }
    }

    /// Set the model to use for LLM-based summarization.
    pub fn with_model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Determine the split point between messages to summarize and messages to keep,
    /// ensuring AI/Tool message pairs are not split apart.
    fn compute_keep_boundary(&self, messages: &[Message], keep_count: usize) -> usize {
        if messages.len() <= keep_count {
            return 0;
        }
        let mut boundary = messages.len() - keep_count;

        // Walk backward from the boundary to avoid splitting an AI+Tool pair.
        // If the message at the boundary is a Tool message, include the preceding
        // AI message that triggered it in the "keep" portion.
        while boundary > 0 {
            if let Message::Tool(_) = &messages[boundary] {
                // The tool response belongs with its preceding AI message
                boundary -= 1;
            } else {
                break;
            }
        }
        boundary
    }

    /// Build a concatenated summary from messages (fallback when no model is available).
    fn fallback_summarize(&self, messages: &[Message]) -> String {
        let summary_text: String = messages
            .iter()
            .map(|m| {
                let role = m.message_type().as_str();
                let content = m.content().text();
                format!("{}: {}", role, content)
            })
            .collect::<Vec<_>>()
            .join("\n");
        summary_text
    }

    /// Check whether summarization should trigger based on the current state.
    fn should_trigger(&self, state: &AgentState) -> bool {
        match &self.config.trigger {
            ContextSize::Messages(max) => state.messages.len() > *max,
            ContextSize::Tokens(max) => {
                // Estimate: ~4 chars per token
                let est_tokens: usize = state
                    .messages
                    .iter()
                    .map(|m| m.content().text().len() / 4)
                    .sum();
                est_tokens > *max
            }
            ContextSize::Fraction(frac) => {
                // Without exact model context info, use a heuristic:
                // estimate total chars and compare against a reasonable threshold.
                // Assume ~100k chars as a typical context window (~25k tokens).
                let total_chars: usize = state
                    .messages
                    .iter()
                    .map(|m| m.content().text().len())
                    .sum();
                let threshold = (100_000.0 * frac) as usize;
                total_chars > threshold
            }
        }
    }

    /// Determine the number of messages to keep.
    fn keep_count(&self) -> usize {
        match &self.config.keep {
            ContextSize::Messages(n) => *n,
            ContextSize::Tokens(_) | ContextSize::Fraction(_) => 10,
        }
    }
}

#[async_trait]
impl AgentMiddleware for SummarizationMiddleware {
    fn name(&self) -> &str {
        "SummarizationMiddleware"
    }

    async fn before_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        if !self.should_trigger(state) {
            return Ok(None);
        }

        let keep_count = self.keep_count();

        // Preserve the last N messages, respecting AI/Tool pairs
        let boundary = self.compute_keep_boundary(&state.messages, keep_count);
        if boundary == 0 {
            return Ok(None);
        }

        let to_summarize = &state.messages[..boundary];
        let to_keep = &state.messages[boundary..];

        // Generate summary using LLM if available, otherwise fallback
        let summary_text = if let Some(model) = &self.model {
            // Build messages for the summarization LLM call
            let mut summarize_messages = vec![Message::system(&self.config.summary_prompt)];
            // Include the conversation to summarize as a human message
            let conversation_text = self.fallback_summarize(to_summarize);
            summarize_messages.push(Message::human(format!(
                "Please summarize the following conversation:\n\n{}",
                conversation_text
            )));

            match model.invoke_messages(&summarize_messages, None).await {
                Ok(ai_msg) => ai_msg.base.content.text(),
                Err(_) => {
                    // Fall back to concatenation if the LLM call fails
                    self.fallback_summarize(to_summarize)
                }
            }
        } else {
            self.fallback_summarize(to_summarize)
        };

        // Replace older messages with a summary system message + kept messages
        let summary_msg = Message::system(format!(
            "[Summary of previous conversation]\n{}",
            summary_text
        ));

        let mut new_messages = vec![summary_msg];
        new_messages.extend_from_slice(to_keep);

        let mut updates = HashMap::new();
        updates.insert("messages".into(), serde_json::to_value(&new_messages)?);
        Ok(Some(updates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarization_default() {
        let mw = SummarizationMiddleware::default();
        assert_eq!(mw.name(), "SummarizationMiddleware");
    }

    #[test]
    fn test_context_size_messages() {
        let size = ContextSize::Messages(20);
        match size {
            ContextSize::Messages(n) => assert_eq!(n, 20),
            _ => panic!("Expected Messages variant"),
        }
    }
}
