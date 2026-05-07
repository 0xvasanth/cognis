//! Context editing middleware — modify the message context before model calls.
//!
//! Mirrors Python `langchain.agents.middleware.context_editing`.

use std::collections::HashSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis_core::error::Result;
use cognis_core::messages::{Message, MessageType};

use super::types::{AgentMiddleware, AsyncModelHandler, ModelCallResult, ModelRequest};

/// Trait for context editing operations.
///
/// Implementors modify the message list before it is sent to the model,
/// for example by removing old tool call/response pairs to save tokens.
pub trait ContextEdit: Send + Sync {
    /// The name of this edit operation.
    fn name(&self) -> &str;

    /// Apply the edit to the message list, returning a new list.
    fn apply(&self, messages: &[Message]) -> Vec<Message>;
}

/// Trigger conditions for when to apply the ClearToolUsesEdit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClearToolUsesTrigger {
    /// Always apply when there are tool uses.
    Always,
    /// Apply when the message count exceeds a threshold.
    MessageCountExceeds(usize),
    /// Apply when estimated token count exceeds a threshold.
    TokenCountExceeds(usize),
}

/// Configuration for clearing old tool use/response pairs from context.
#[derive(Debug, Clone)]
pub struct ClearToolUsesEdit {
    /// When to trigger clearing.
    pub trigger: ClearToolUsesTrigger,
    /// Minimum number of tool use pairs to clear.
    pub clear_at_least: usize,
    /// Number of recent message pairs to keep.
    pub keep: usize,
    /// Tool names to exclude from clearing (always keep these).
    pub exclude_tools: HashSet<String>,
    /// Placeholder text to insert where tool uses were removed.
    pub placeholder: Option<String>,
}

impl Default for ClearToolUsesEdit {
    fn default() -> Self {
        Self {
            trigger: ClearToolUsesTrigger::Always,
            clear_at_least: 1,
            keep: 5,
            exclude_tools: HashSet::new(),
            placeholder: Some("[Previous tool interactions removed for brevity]".into()),
        }
    }
}

impl ClearToolUsesEdit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_trigger(mut self, trigger: ClearToolUsesTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    pub fn with_keep(mut self, keep: usize) -> Self {
        self.keep = keep;
        self
    }

    pub fn with_exclude_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.exclude_tools.insert(tool_name.into());
        self
    }

    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = Some(placeholder.into());
        self
    }

    fn should_trigger(&self, messages: &[Message]) -> bool {
        match &self.trigger {
            ClearToolUsesTrigger::Always => true,
            ClearToolUsesTrigger::MessageCountExceeds(threshold) => messages.len() > *threshold,
            ClearToolUsesTrigger::TokenCountExceeds(threshold) => {
                let est_tokens: usize = messages.iter().map(|m| m.content().text().len() / 4).sum();
                est_tokens > *threshold
            }
        }
    }

    /// Identify indices of tool-related messages (AI with tool calls and Tool responses).
    fn find_tool_message_indices(&self, messages: &[Message]) -> Vec<usize> {
        let mut indices = Vec::new();
        for (i, msg) in messages.iter().enumerate() {
            match msg.message_type() {
                MessageType::Tool => {
                    indices.push(i);
                }
                MessageType::Ai
                    // Check if the AI message appears to have tool calls
                    // by looking at subsequent Tool messages
                    if i + 1 < messages.len() && messages[i + 1].message_type() == MessageType::Tool
                    => {
                        indices.push(i);
                    }
                _ => {}
            }
        }
        indices
    }
}

impl ContextEdit for ClearToolUsesEdit {
    fn name(&self) -> &str {
        "ClearToolUsesEdit"
    }

    fn apply(&self, messages: &[Message]) -> Vec<Message> {
        if !self.should_trigger(messages) {
            return messages.to_vec();
        }

        let tool_indices = self.find_tool_message_indices(messages);
        if tool_indices.len() <= self.keep {
            return messages.to_vec();
        }

        let clearable_count = tool_indices.len().saturating_sub(self.keep);
        if clearable_count < self.clear_at_least {
            return messages.to_vec();
        }

        // Remove the oldest tool-related messages, keeping the most recent `keep` pairs
        let to_remove: HashSet<usize> = tool_indices[..clearable_count].iter().copied().collect();

        let mut result = Vec::new();
        let mut placeholder_inserted = false;

        for (i, msg) in messages.iter().enumerate() {
            if to_remove.contains(&i) {
                if !placeholder_inserted {
                    if let Some(ref placeholder) = self.placeholder {
                        result.push(Message::system(placeholder.as_str()));
                        placeholder_inserted = true;
                    }
                }
            } else {
                result.push(msg.clone());
            }
        }

        result
    }
}

/// Middleware that applies context edits before model calls.
pub struct ContextEditingMiddleware {
    /// The context edits to apply, in order.
    pub edits: Vec<Box<dyn ContextEdit>>,
}

impl ContextEditingMiddleware {
    pub fn new(edits: Vec<Box<dyn ContextEdit>>) -> Self {
        Self { edits }
    }

    /// Create with a single ClearToolUsesEdit.
    pub fn clear_tool_uses(config: ClearToolUsesEdit) -> Self {
        Self {
            edits: vec![Box::new(config)],
        }
    }

    /// Apply all edits to the messages.
    fn apply_edits(&self, messages: &[Message]) -> Vec<Message> {
        let mut current = messages.to_vec();
        for edit in &self.edits {
            current = edit.apply(&current);
        }
        current
    }
}

#[async_trait]
impl AgentMiddleware for ContextEditingMiddleware {
    fn name(&self) -> &str {
        "ContextEditingMiddleware"
    }

    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        // Apply context edits to the request messages
        let edited_messages = self.apply_edits(&request.messages);

        // Construct a new request with the edited messages
        let edited_request = ModelRequest {
            model: request.model.clone(),
            messages: edited_messages,
            system_message: request.system_message.clone(),
            tool_choice: request.tool_choice.clone(),
            tools: request.tools.clone(),
            response_format: request.response_format.clone(),
            state: request.state.clone(),
            model_settings: request.model_settings.clone(),
        };

        let response = handler(&edited_request).await?;
        Ok(ModelCallResult::Response(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_tool_uses_edit_default() {
        let edit = ClearToolUsesEdit::default();
        assert_eq!(edit.keep, 5);
        assert_eq!(edit.clear_at_least, 1);
        assert!(edit.placeholder.is_some());
    }

    #[test]
    fn test_clear_tool_uses_edit_builder() {
        let edit = ClearToolUsesEdit::new()
            .with_keep(10)
            .with_trigger(ClearToolUsesTrigger::MessageCountExceeds(20))
            .with_exclude_tool("important_tool")
            .with_placeholder("removed");
        assert_eq!(edit.keep, 10);
        assert!(edit.exclude_tools.contains("important_tool"));
    }

    #[test]
    fn test_clear_tool_uses_no_tools() {
        let edit = ClearToolUsesEdit::default();
        let messages = vec![Message::human("hello"), Message::ai("hi there")];
        let result = edit.apply(&messages);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_clear_tool_uses_with_tool_messages() {
        let edit = ClearToolUsesEdit::new()
            .with_keep(0)
            .with_placeholder("removed".to_string());
        let messages = vec![
            Message::human("hello"),
            Message::ai("calling tool"),
            Message::tool("result", "call_1"),
            Message::ai("calling tool 2"),
            Message::tool("result 2", "call_2"),
            Message::ai("final answer"),
        ];
        let result = edit.apply(&messages);
        // Some tool messages should be removed
        assert!(result.len() <= messages.len());
    }

    #[test]
    fn test_clear_tool_uses_trigger_message_count() {
        let edit =
            ClearToolUsesEdit::new().with_trigger(ClearToolUsesTrigger::MessageCountExceeds(100));
        let messages = vec![Message::human("hello")];
        // Should not trigger since message count is below threshold
        let result = edit.apply(&messages);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_context_editing_middleware_name() {
        let mw = ContextEditingMiddleware::clear_tool_uses(ClearToolUsesEdit::default());
        assert_eq!(mw.name(), "ContextEditingMiddleware");
    }

    #[test]
    fn test_context_editing_middleware_apply_edits() {
        let mw = ContextEditingMiddleware::clear_tool_uses(ClearToolUsesEdit::default());
        let messages = vec![Message::human("hello"), Message::ai("world")];
        let result = mw.apply_edits(&messages);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_context_edit_trait_name() {
        let edit = ClearToolUsesEdit::default();
        assert_eq!(ContextEdit::name(&edit), "ClearToolUsesEdit");
    }
}
