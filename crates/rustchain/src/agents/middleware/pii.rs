//! PII middleware — detect and redact personally identifiable information.
//!
//! Mirrors Python `langchain.agents.middleware.pii`.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::messages::{Message, MessageType};
use rustchain_core::messages::base::MessageContent;

use super::redaction::{
    Detector, PIIMatch, RedactionStrategy, ResolvedRedactionRule,
    apply_rules, builtin_detectors, get_builtin_detector,
};
use super::types::{AgentMiddleware, AgentState};

/// Middleware that detects and redacts PII in messages.
pub struct PIIMiddleware {
    /// The PII type to detect (e.g., "email", "credit_card", "ip", "mac_address", "url").
    /// If `None`, all built-in detectors are applied.
    pub pii_type: Option<String>,
    /// The redaction strategy to apply when PII is detected.
    pub strategy: RedactionStrategy,
    /// Whether to apply PII detection to user input (before model).
    pub apply_to_input: bool,
    /// Whether to apply PII detection to model output (after model).
    pub apply_to_output: bool,
    /// Whether to apply PII detection to tool results.
    pub apply_to_tool_results: bool,
    /// Optional custom detector. When set, this detector is used instead of built-in ones.
    pub detector: Option<Detector>,
}

impl PIIMiddleware {
    /// Create a new PIIMiddleware with the given strategy applied to all PII types.
    pub fn new(strategy: RedactionStrategy) -> Self {
        Self {
            pii_type: None,
            strategy,
            apply_to_input: true,
            apply_to_output: true,
            apply_to_tool_results: false,
            detector: None,
        }
    }

    /// Create a PIIMiddleware targeting a specific PII type.
    pub fn for_type(pii_type: impl Into<String>, strategy: RedactionStrategy) -> Self {
        Self {
            pii_type: Some(pii_type.into()),
            strategy,
            apply_to_input: true,
            apply_to_output: true,
            apply_to_tool_results: false,
            detector: None,
        }
    }

    /// Set a custom detector function.
    pub fn with_detector(mut self, detector: Detector) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Run all relevant detectors on the given text.
    fn detect(&self, text: &str) -> Vec<PIIMatch> {
        // If a custom detector is set, use it exclusively
        if let Some(ref detector) = self.detector {
            return detector(text);
        }

        // If a specific pii_type is set, use only that detector
        if let Some(ref pii_type) = self.pii_type {
            if let Some(detector) = get_builtin_detector(pii_type) {
                return detector(text);
            }
            return Vec::new();
        }

        // Otherwise, run ALL built-in detectors
        let mut all = Vec::new();
        for (_name, detector) in builtin_detectors() {
            all.extend(detector(text));
        }
        all.sort_by_key(|m| m.start);
        all
    }

    /// Redact PII from text, returning the cleaned text.
    fn redact_text(&self, text: &str) -> String {
        let matches = self.detect(text);
        if matches.is_empty() {
            return text.to_string();
        }

        if self.strategy == RedactionStrategy::Block {
            return String::new();
        }

        let rules: Vec<ResolvedRedactionRule> = matches
            .into_iter()
            .map(|m| ResolvedRedactionRule {
                pii_match: m,
                strategy: self.strategy.clone(),
            })
            .collect();
        apply_rules(text, rules).unwrap_or_default()
    }

    /// Redact a single message, preserving all metadata (id, name, tool_calls, etc.).
    /// Returns the message with only the text content replaced.
    fn redact_message(&self, msg: &Message) -> (Message, bool) {
        let text = msg.content().text();
        let redacted = self.redact_text(&text);
        if redacted == text {
            return (msg.clone(), false);
        }

        // Clone the message and replace only the content, preserving all metadata
        let mut new_msg = msg.clone();
        let new_content = MessageContent::Text(redacted);
        match &mut new_msg {
            Message::Human(m) => m.base.content = new_content,
            Message::Ai(m) => m.base.content = new_content,
            Message::System(m) => m.base.content = new_content,
            Message::Tool(m) => m.base.content = new_content,
            Message::Function(m) => m.base.content = new_content,
            Message::Chat(m) => m.base.content = new_content,
            Message::HumanChunk(m) => m.base.content = new_content,
            Message::AiChunk(m) => m.base.content = new_content,
            Message::SystemChunk(m) => m.base.content = new_content,
            Message::ToolChunk(m) => m.base.content = new_content,
            Message::FunctionChunk(m) => m.base.content = new_content,
            Message::ChatChunk(m) => m.base.content = new_content,
            Message::Remove(_) => {} // no content to redact
        }
        (new_msg, true)
    }

    /// Redact PII from the last HumanMessage in the list (for input redaction).
    /// Also redacts ToolMessages after the last AIMessage (tool results).
    fn redact_input_messages(&self, messages: &[Message]) -> Option<Vec<Message>> {
        if messages.is_empty() {
            return None;
        }

        let mut new_messages = messages.to_vec();
        let mut changed = false;

        // Find the last HumanMessage and redact it
        if let Some(last_human_idx) = new_messages.iter().rposition(|m| m.message_type() == MessageType::Human) {
            let (redacted, did_change) = self.redact_message(&new_messages[last_human_idx]);
            if did_change {
                new_messages[last_human_idx] = redacted;
                changed = true;
            }
        }

        // If apply_to_tool_results, redact ToolMessages after the last AIMessage
        if self.apply_to_tool_results {
            let last_ai_idx = new_messages.iter().rposition(|m| m.message_type() == MessageType::Ai);
            let start_idx = last_ai_idx.map(|i| i + 1).unwrap_or(0);
            for msg in new_messages.iter_mut().skip(start_idx) {
                if msg.message_type() == MessageType::Tool {
                    let (redacted, did_change) = self.redact_message(msg);
                    if did_change {
                        *msg = redacted;
                        changed = true;
                    }
                }
            }
        }

        if changed { Some(new_messages) } else { None }
    }

    /// Redact PII from the last AIMessage in the list (for output redaction).
    fn redact_output_messages(&self, messages: &[Message]) -> Option<Vec<Message>> {
        if messages.is_empty() {
            return None;
        }

        let mut new_messages = messages.to_vec();

        // Find the last AIMessage and redact it
        if let Some(last_ai_idx) = new_messages.iter().rposition(|m| m.message_type() == MessageType::Ai) {
            let (redacted, did_change) = self.redact_message(&new_messages[last_ai_idx]);
            if did_change {
                new_messages[last_ai_idx] = redacted;
                return Some(new_messages);
            }
        }

        None
    }
}

#[async_trait]
impl AgentMiddleware for PIIMiddleware {
    fn name(&self) -> &str {
        "PIIMiddleware"
    }

    async fn before_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        if !self.apply_to_input {
            return Ok(None);
        }

        if let Some(redacted) = self.redact_input_messages(&state.messages) {
            let mut updates = HashMap::new();
            updates.insert("messages".into(), serde_json::to_value(&redacted)?);
            Ok(Some(updates))
        } else {
            Ok(None)
        }
    }

    async fn after_model(&self, state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        if !self.apply_to_output {
            return Ok(None);
        }

        if let Some(redacted) = self.redact_output_messages(&state.messages) {
            let mut updates = HashMap::new();
            updates.insert("messages".into(), serde_json::to_value(&redacted)?);
            Ok(Some(updates))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_pii_middleware_detect_email() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let matches = mw.detect("Send to user@example.com");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, "email");
    }

    #[test]
    fn test_pii_middleware_redact_text() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let result = mw.redact_text("Contact user@example.com please");
        assert!(result.contains("[REDACTED_EMAIL]"));
        assert!(!result.contains("user@example.com"));
    }

    #[test]
    fn test_pii_middleware_all_types() {
        let mw = PIIMiddleware::new(RedactionStrategy::Redact);
        let text = "Email: a@b.com, IP: 10.0.0.1";
        let matches = mw.detect(text);
        assert!(matches.len() >= 2);
    }

    #[test]
    fn test_pii_middleware_block_strategy() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Block);
        let result = mw.redact_text("Email: a@b.com");
        assert_eq!(result, "");
    }

    #[test]
    fn test_pii_middleware_no_match() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let result = mw.redact_text("No PII here");
        assert_eq!(result, "No PII here");
    }

    #[test]
    fn test_pii_middleware_redact_input_messages_only_last_human() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let msgs = vec![
            Message::human("My email is first@example.com"),
            Message::ai("Got it"),
            Message::human("My email is test@example.com"),
        ];
        let redacted = mw.redact_input_messages(&msgs);
        assert!(redacted.is_some());
        let redacted = redacted.unwrap();
        // First human message should NOT be redacted (only last human is redacted)
        let first_text = redacted[0].content().text();
        assert!(first_text.contains("first@example.com"));
        // Last human message should be redacted
        let last_text = redacted[2].content().text();
        assert!(last_text.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn test_pii_middleware_redact_output_messages_only_last_ai() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let msgs = vec![
            Message::human("Hi"),
            Message::ai("First reply with first@example.com"),
            Message::human("Another"),
            Message::ai("Second reply with second@example.com"),
        ];
        let redacted = mw.redact_output_messages(&msgs);
        assert!(redacted.is_some());
        let redacted = redacted.unwrap();
        // First AI message should NOT be redacted
        let first_ai = redacted[1].content().text();
        assert!(first_ai.contains("first@example.com"));
        // Last AI message should be redacted
        let last_ai = redacted[3].content().text();
        assert!(last_ai.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn test_pii_middleware_preserves_metadata() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        // Create an AI message with tool calls
        let mut ai_msg = rustchain_core::messages::AIMessage::new("Email is ai@test.com");
        ai_msg.base.id = Some("msg-123".to_string());
        ai_msg.base.name = Some("assistant".to_string());
        ai_msg.tool_calls = vec![rustchain_core::messages::tool_types::ToolCall {
            name: "search".into(),
            args: Default::default(),
            id: Some("tc-1".into()),
        }];
        let msgs = vec![Message::Ai(ai_msg)];
        let redacted = mw.redact_output_messages(&msgs);
        assert!(redacted.is_some());
        let redacted = redacted.unwrap();
        // Check metadata is preserved
        if let Message::Ai(ref m) = redacted[0] {
            assert_eq!(m.base.id, Some("msg-123".to_string()));
            assert_eq!(m.base.name, Some("assistant".to_string()));
            assert_eq!(m.tool_calls.len(), 1);
            assert_eq!(m.tool_calls[0].name, "search");
        } else {
            panic!("Expected AI message");
        }
    }

    #[test]
    fn test_pii_middleware_custom_detector() {
        let custom_detector: Detector = Arc::new(|text: &str| {
            let mut matches = Vec::new();
            if let Some(start) = text.find("SECRET") {
                matches.push(PIIMatch {
                    pii_type: "secret".into(),
                    value: "SECRET".into(),
                    start,
                    end: start + 6,
                });
            }
            matches
        });
        let mw = PIIMiddleware::new(RedactionStrategy::Redact)
            .with_detector(custom_detector);
        let result = mw.redact_text("My SECRET code");
        assert_eq!(result, "My [REDACTED_SECRET] code");
    }

    #[test]
    fn test_pii_middleware_name() {
        let mw = PIIMiddleware::new(RedactionStrategy::Mask);
        assert_eq!(mw.name(), "PIIMiddleware");
    }

    #[tokio::test]
    async fn test_pii_middleware_before_model_redacts() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let state = AgentState::new(vec![Message::human("Send to foo@bar.com")]);
        let result = mw.before_model(&state).await.unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_pii_middleware_before_model_no_pii() {
        let mw = PIIMiddleware::for_type("email", RedactionStrategy::Redact);
        let state = AgentState::new(vec![Message::human("Hello world")]);
        let result = mw.before_model(&state).await.unwrap();
        assert!(result.is_none());
    }
}
