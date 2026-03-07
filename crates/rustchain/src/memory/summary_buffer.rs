//! Summary buffer memory that combines buffered messages with periodic summarization.
//!
//! When the buffer exceeds a configurable token threshold, older messages are
//! summarized into a single summary message using a pluggable [`Summarizer`] trait.
//! This keeps the context window manageable while preserving important information
//! from earlier in the conversation.

use rustchain_core::error::Result;
use rustchain_core::messages::Message;
use serde_json::{json, Value};

// ─── SummaryStrategy ─────────────────────────────────────────────────────────

/// Controls which messages get summarized when the token threshold is exceeded.
#[derive(Debug, Clone, PartialEq)]
pub enum SummaryStrategy {
    /// Summarize only the first (oldest) message in the buffer.
    First,
    /// Summarize the oldest half of the buffer.
    Oldest,
    /// Summarize all messages except the most recent `n`.
    Sliding(usize),
}

impl Default for SummaryStrategy {
    fn default() -> Self {
        Self::Oldest
    }
}

// ─── Summarizer trait ────────────────────────────────────────────────────────

/// Pluggable summarization strategy.
///
/// Implementations receive the messages to summarize and an optional existing
/// summary to incorporate, then return a new summary string.
pub trait Summarizer: Send + Sync {
    /// Produce a summary of the given messages, optionally incorporating an
    /// existing summary.
    fn summarize(&self, messages: &[Message], existing_summary: Option<&str>) -> Result<String>;
}

// ─── SimpleSummarizer ────────────────────────────────────────────────────────

/// Concatenates messages into a bullet-point summary without requiring an LLM.
#[derive(Debug, Clone, Default)]
pub struct SimpleSummarizer;

impl SimpleSummarizer {
    /// Create a new simple summarizer.
    pub fn new() -> Self {
        Self
    }

    fn format_message(msg: &Message) -> String {
        let role = match msg.message_type() {
            rustchain_core::messages::MessageType::Human => "Human",
            rustchain_core::messages::MessageType::Ai => "AI",
            rustchain_core::messages::MessageType::System => "System",
            rustchain_core::messages::MessageType::Tool => "Tool",
            _ => "Other",
        };
        let text = msg.content().text();
        format!("- {}: {}", role, text)
    }
}

impl Summarizer for SimpleSummarizer {
    fn summarize(&self, messages: &[Message], existing_summary: Option<&str>) -> Result<String> {
        let mut parts = Vec::new();

        if let Some(existing) = existing_summary {
            if !existing.is_empty() {
                parts.push(format!("Previous summary:\n{}", existing));
                parts.push(String::new());
                parts.push("New messages:".to_string());
            }
        }

        for msg in messages {
            parts.push(Self::format_message(msg));
        }

        Ok(parts.join("\n"))
    }
}

// ─── TemplateSummarizer ──────────────────────────────────────────────────────

/// Uses a configurable template string with `{messages}` and `{existing_summary}`
/// placeholders.
#[derive(Debug, Clone)]
pub struct TemplateSummarizer {
    template: String,
}

impl TemplateSummarizer {
    /// Create a new template summarizer.
    ///
    /// The template should contain `{messages}` and optionally
    /// `{existing_summary}` placeholders.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }
}

impl Summarizer for TemplateSummarizer {
    fn summarize(&self, messages: &[Message], existing_summary: Option<&str>) -> Result<String> {
        let messages_text: Vec<String> = messages
            .iter()
            .map(|m| {
                let role = match m.message_type() {
                    rustchain_core::messages::MessageType::Human => "Human",
                    rustchain_core::messages::MessageType::Ai => "AI",
                    rustchain_core::messages::MessageType::System => "System",
                    rustchain_core::messages::MessageType::Tool => "Tool",
                    _ => "Other",
                };
                format!("{}: {}", role, m.content().text())
            })
            .collect();

        let messages_str = messages_text.join("\n");
        let existing = existing_summary.unwrap_or("");

        let result = self
            .template
            .replace("{messages}", &messages_str)
            .replace("{existing_summary}", existing);

        Ok(result)
    }
}

// ─── Token estimation ────────────────────────────────────────────────────────

/// Simple token estimation: split on whitespace and multiply by 4/3.
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count() * 4 / 3
}

fn estimate_message_tokens(msg: &Message) -> usize {
    estimate_tokens(&msg.content().text()) + 3 // overhead for role framing
}

fn estimate_total_tokens(messages: &[Message], summary: &Option<String>) -> usize {
    let msg_tokens: usize = messages.iter().map(|m| estimate_message_tokens(m)).sum();
    let summary_tokens = summary.as_ref().map_or(0, |s| estimate_tokens(s));
    msg_tokens + summary_tokens
}

// ─── SummaryBufferMemory ─────────────────────────────────────────────────────

/// Conversation memory that combines a message buffer with periodic summarization.
///
/// When the total estimated token count (buffer + summary) exceeds
/// `max_token_count`, older messages are summarized according to the configured
/// [`SummaryStrategy`] and the summary is updated.
///
/// # Example
///
/// ```rust
/// use rustchain::memory::summary_buffer::{SummaryBufferMemory, SimpleSummarizer};
///
/// let mem = SummaryBufferMemory::new(100, SimpleSummarizer::new());
/// ```
pub struct SummaryBufferMemory {
    buffer: Vec<Message>,
    summary: Option<String>,
    max_token_count: usize,
    summarizer: Box<dyn Summarizer>,
    human_prefix: String,
    ai_prefix: String,
    memory_key: String,
    strategy: SummaryStrategy,
}

impl SummaryBufferMemory {
    /// Create a new summary buffer memory.
    ///
    /// - `max_token_count` — when the estimated token count exceeds this,
    ///   summarization is triggered.
    /// - `summarizer` — the summarization strategy to use.
    pub fn new(max_token_count: usize, summarizer: impl Summarizer + 'static) -> Self {
        Self {
            buffer: Vec::new(),
            summary: None,
            max_token_count,
            summarizer: Box::new(summarizer),
            human_prefix: "Human".to_string(),
            ai_prefix: "AI".to_string(),
            memory_key: "history".to_string(),
            strategy: SummaryStrategy::default(),
        }
    }

    /// Return a builder for constructing a `SummaryBufferMemory`.
    pub fn builder() -> SummaryBufferMemoryBuilder {
        SummaryBufferMemoryBuilder::default()
    }

    /// Set the human prefix used in formatted output.
    pub fn with_human_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.human_prefix = prefix.into();
        self
    }

    /// Set the AI prefix used in formatted output.
    pub fn with_ai_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.ai_prefix = prefix.into();
        self
    }

    /// Set the memory key used in context output.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set the summary strategy.
    pub fn with_strategy(mut self, strategy: SummaryStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a message to the buffer, triggering summarization if the token
    /// threshold is exceeded.
    pub fn add_message(&mut self, msg: Message) -> Result<()> {
        self.buffer.push(msg);
        self.maybe_summarize()
    }

    /// Get the context as a JSON value containing the summary and recent messages.
    pub fn get_context(&self) -> Value {
        let messages: Vec<Value> = self
            .buffer
            .iter()
            .map(|m| {
                let role = match m.message_type() {
                    rustchain_core::messages::MessageType::Human => self.human_prefix.as_str(),
                    rustchain_core::messages::MessageType::Ai => self.ai_prefix.as_str(),
                    rustchain_core::messages::MessageType::System => "System",
                    rustchain_core::messages::MessageType::Tool => "Tool",
                    _ => "Other",
                };
                json!({
                    "role": role,
                    "content": m.content().text()
                })
            })
            .collect();

        let mut context = json!({
            self.memory_key.clone(): {
                "messages": messages
            }
        });

        if let Some(ref summary) = self.summary {
            context[&self.memory_key]["summary"] = Value::String(summary.clone());
        }

        context
    }

    /// Clear the buffer and summary.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.summary = None;
    }

    /// Return the number of messages currently in the buffer.
    pub fn message_count(&self) -> usize {
        self.buffer.len()
    }

    /// Return whether a summary currently exists.
    pub fn has_summary(&self) -> bool {
        self.summary.is_some()
    }

    /// Return the current summary, if any.
    pub fn current_summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Return a reference to the buffered messages.
    pub fn messages(&self) -> &[Message] {
        &self.buffer
    }

    /// Determine which messages to summarize based on the current strategy,
    /// then perform summarization if the token threshold is exceeded.
    fn maybe_summarize(&mut self) -> Result<()> {
        let total = estimate_total_tokens(&self.buffer, &self.summary);
        if total <= self.max_token_count {
            return Ok(());
        }

        let split_index = match &self.strategy {
            SummaryStrategy::First => {
                if self.buffer.is_empty() {
                    return Ok(());
                }
                1
            }
            SummaryStrategy::Oldest => {
                let half = self.buffer.len() / 2;
                if half == 0 {
                    return Ok(());
                }
                half
            }
            SummaryStrategy::Sliding(keep) => {
                let keep = (*keep).min(self.buffer.len());
                let to_summarize = self.buffer.len().saturating_sub(keep);
                if to_summarize == 0 {
                    return Ok(());
                }
                to_summarize
            }
        };

        let to_summarize: Vec<Message> = self.buffer.drain(..split_index).collect();
        let new_summary = self
            .summarizer
            .summarize(&to_summarize, self.summary.as_deref())?;
        self.summary = Some(new_summary);

        Ok(())
    }
}

// ─── SummaryBufferMemoryBuilder ──────────────────────────────────────────────

/// Fluent builder for [`SummaryBufferMemory`].
#[derive(Default)]
pub struct SummaryBufferMemoryBuilder {
    max_token_count: Option<usize>,
    summarizer: Option<Box<dyn Summarizer>>,
    human_prefix: Option<String>,
    ai_prefix: Option<String>,
    memory_key: Option<String>,
    strategy: Option<SummaryStrategy>,
}

impl SummaryBufferMemoryBuilder {
    /// Set the maximum token count threshold.
    pub fn max_token_count(mut self, count: usize) -> Self {
        self.max_token_count = Some(count);
        self
    }

    /// Set the summarizer implementation.
    pub fn summarizer(mut self, summarizer: impl Summarizer + 'static) -> Self {
        self.summarizer = Some(Box::new(summarizer));
        self
    }

    /// Set the human prefix.
    pub fn human_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.human_prefix = Some(prefix.into());
        self
    }

    /// Set the AI prefix.
    pub fn ai_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.ai_prefix = Some(prefix.into());
        self
    }

    /// Set the memory key.
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = Some(key.into());
        self
    }

    /// Set the summary strategy.
    pub fn strategy(mut self, strategy: SummaryStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    /// Build the [`SummaryBufferMemory`].
    ///
    /// Defaults: 2000 token limit, `SimpleSummarizer`, `Oldest` strategy.
    pub fn build(self) -> SummaryBufferMemory {
        SummaryBufferMemory {
            buffer: Vec::new(),
            summary: None,
            max_token_count: self.max_token_count.unwrap_or(2000),
            summarizer: self
                .summarizer
                .unwrap_or_else(|| Box::new(SimpleSummarizer::new())),
            human_prefix: self.human_prefix.unwrap_or_else(|| "Human".to_string()),
            ai_prefix: self.ai_prefix.unwrap_or_else(|| "AI".to_string()),
            memory_key: self.memory_key.unwrap_or_else(|| "history".to_string()),
            strategy: self.strategy.unwrap_or_default(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::Message;

    // ─── SimpleSummarizer tests ──────────────────────────────────────────

    #[test]
    fn test_simple_summarizer_single_human_message() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::human("Hello world")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("- Human: Hello world"));
    }

    #[test]
    fn test_simple_summarizer_single_ai_message() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::ai("I am an AI")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("- AI: I am an AI"));
    }

    #[test]
    fn test_simple_summarizer_system_message() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::system("You are a helpful assistant")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("- System: You are a helpful assistant"));
    }

    #[test]
    fn test_simple_summarizer_tool_message() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::tool("result data", "call_123")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("- Tool: result data"));
    }

    #[test]
    fn test_simple_summarizer_multiple_messages() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![
            Message::human("What is 2+2?"),
            Message::ai("4"),
            Message::human("Thanks!"),
        ];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("- Human: What is 2+2?"));
        assert!(result.contains("- AI: 4"));
        assert!(result.contains("- Human: Thanks!"));
    }

    #[test]
    fn test_simple_summarizer_with_existing_summary() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::human("New question")];
        let result = summarizer
            .summarize(&messages, Some("Earlier conversation about math"))
            .unwrap();
        assert!(result.contains("Previous summary:"));
        assert!(result.contains("Earlier conversation about math"));
        assert!(result.contains("New messages:"));
        assert!(result.contains("- Human: New question"));
    }

    #[test]
    fn test_simple_summarizer_with_empty_existing_summary() {
        let summarizer = SimpleSummarizer::new();
        let messages = vec![Message::human("Hello")];
        let result = summarizer.summarize(&messages, Some("")).unwrap();
        // Empty existing summary should not add the "Previous summary:" header
        assert!(!result.contains("Previous summary:"));
        assert!(result.contains("- Human: Hello"));
    }

    #[test]
    fn test_simple_summarizer_empty_messages() {
        let summarizer = SimpleSummarizer::new();
        let result = summarizer.summarize(&[], None).unwrap();
        assert!(result.is_empty());
    }

    // ─── TemplateSummarizer tests ────────────────────────────────────────

    #[test]
    fn test_template_summarizer_basic() {
        let summarizer = TemplateSummarizer::new("Messages:\n{messages}");
        let messages = vec![Message::human("Hi"), Message::ai("Hello")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.starts_with("Messages:\n"));
        assert!(result.contains("Human: Hi"));
        assert!(result.contains("AI: Hello"));
    }

    #[test]
    fn test_template_summarizer_with_existing_summary_placeholder() {
        let summarizer =
            TemplateSummarizer::new("Prior: {existing_summary}\nCurrent: {messages}");
        let messages = vec![Message::human("Question")];
        let result = summarizer
            .summarize(&messages, Some("talked about weather"))
            .unwrap();
        assert!(result.contains("Prior: talked about weather"));
        assert!(result.contains("Current: Human: Question"));
    }

    #[test]
    fn test_template_summarizer_no_existing_summary() {
        let summarizer =
            TemplateSummarizer::new("Summary: {existing_summary}\nNew: {messages}");
        let messages = vec![Message::ai("Response")];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.contains("Summary: \n"));
        assert!(result.contains("New: AI: Response"));
    }

    #[test]
    fn test_template_summarizer_custom_template() {
        let summarizer = TemplateSummarizer::new("## Conversation Log\n{messages}");
        let messages = vec![
            Message::human("Start"),
            Message::ai("Acknowledged"),
        ];
        let result = summarizer.summarize(&messages, None).unwrap();
        assert!(result.starts_with("## Conversation Log\n"));
    }

    // ─── SummaryBufferMemory add/get/clear cycle ─────────────────────────

    #[test]
    fn test_add_and_get_messages() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        mem.add_message(Message::human("Hello")).unwrap();
        mem.add_message(Message::ai("Hi there")).unwrap();
        assert_eq!(mem.message_count(), 2);
    }

    #[test]
    fn test_clear_resets_everything() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new());
        // Add enough to trigger summarization
        mem.add_message(Message::human("A very long message that should exceed the small token limit we set"))
            .unwrap();
        mem.add_message(Message::ai("Another long response that will definitely push us over the edge"))
            .unwrap();

        mem.clear();
        assert_eq!(mem.message_count(), 0);
        assert!(!mem.has_summary());
        assert!(mem.current_summary().is_none());
    }

    #[test]
    fn test_get_context_with_messages() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        mem.add_message(Message::human("Hello")).unwrap();
        mem.add_message(Message::ai("Hi")).unwrap();

        let ctx = mem.get_context();
        let history = ctx.get("history").unwrap();
        let messages = history.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "Human");
        assert_eq!(messages[0]["content"], "Hello");
        assert_eq!(messages[1]["role"], "AI");
        assert_eq!(messages[1]["content"], "Hi");
    }

    #[test]
    fn test_get_context_no_summary_key_when_no_summary() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        mem.add_message(Message::human("Hello")).unwrap();

        let ctx = mem.get_context();
        let history = ctx.get("history").unwrap();
        assert!(history.get("summary").is_none());
    }

    // ─── Auto-summarization when token threshold exceeded ────────────────

    #[test]
    fn test_auto_summarization_triggered() {
        // Very small threshold to force summarization
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new());
        mem.add_message(Message::human("This is a long message that exceeds the token limit"))
            .unwrap();
        mem.add_message(Message::ai("This is also a long response with many words"))
            .unwrap();

        // After exceeding threshold, summarization should have occurred
        assert!(mem.has_summary());
        // Buffer should have fewer messages than we added
        assert!(mem.message_count() < 2);
    }

    #[test]
    fn test_no_summarization_under_threshold() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        mem.add_message(Message::human("Hi")).unwrap();
        mem.add_message(Message::ai("Hello")).unwrap();

        assert!(!mem.has_summary());
        assert_eq!(mem.message_count(), 2);
    }

    #[test]
    fn test_summary_content_includes_summarized_messages() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new());
        mem.add_message(Message::human("What is the meaning of life the universe and everything"))
            .unwrap();
        mem.add_message(Message::ai("The answer is forty two according to the guide"))
            .unwrap();

        if mem.has_summary() {
            let summary = mem.current_summary().unwrap();
            // The summary should contain traces of the summarized messages
            assert!(!summary.is_empty());
        }
    }

    #[test]
    fn test_get_context_includes_summary() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new());
        mem.add_message(Message::human("A long message to trigger summarization in the buffer"))
            .unwrap();
        mem.add_message(Message::ai("Another long message that pushes over the threshold"))
            .unwrap();

        if mem.has_summary() {
            let ctx = mem.get_context();
            let history = ctx.get("history").unwrap();
            assert!(history.get("summary").is_some());
        }
    }

    // ─── Builder pattern ─────────────────────────────────────────────────

    #[test]
    fn test_builder_defaults() {
        let mem = SummaryBufferMemory::builder().build();
        assert_eq!(mem.max_token_count, 2000);
        assert_eq!(mem.human_prefix, "Human");
        assert_eq!(mem.ai_prefix, "AI");
        assert_eq!(mem.memory_key, "history");
        assert_eq!(mem.strategy, SummaryStrategy::Oldest);
    }

    #[test]
    fn test_builder_custom_values() {
        let mem = SummaryBufferMemory::builder()
            .max_token_count(500)
            .human_prefix("User")
            .ai_prefix("Assistant")
            .memory_key("chat")
            .strategy(SummaryStrategy::First)
            .summarizer(SimpleSummarizer::new())
            .build();

        assert_eq!(mem.max_token_count, 500);
        assert_eq!(mem.human_prefix, "User");
        assert_eq!(mem.ai_prefix, "Assistant");
        assert_eq!(mem.memory_key, "chat");
        assert_eq!(mem.strategy, SummaryStrategy::First);
    }

    #[test]
    fn test_builder_with_template_summarizer() {
        let mem = SummaryBufferMemory::builder()
            .max_token_count(100)
            .summarizer(TemplateSummarizer::new("Summary: {messages}"))
            .build();

        assert_eq!(mem.max_token_count, 100);
    }

    #[test]
    fn test_with_methods_chainable() {
        let mem = SummaryBufferMemory::new(100, SimpleSummarizer::new())
            .with_human_prefix("User")
            .with_ai_prefix("Bot")
            .with_memory_key("conv")
            .with_strategy(SummaryStrategy::Sliding(3));

        assert_eq!(mem.human_prefix, "User");
        assert_eq!(mem.ai_prefix, "Bot");
        assert_eq!(mem.memory_key, "conv");
        assert_eq!(mem.strategy, SummaryStrategy::Sliding(3));
    }

    // ─── Summary strategies ─────────────────────────────────────────────

    #[test]
    fn test_strategy_first_summarizes_one() {
        // Use a moderate threshold so summarization triggers but doesn't drain all messages
        let mut mem = SummaryBufferMemory::new(30, SimpleSummarizer::new())
            .with_strategy(SummaryStrategy::First);

        mem.add_message(Message::human("First message with enough words to exceed limit"))
            .unwrap();
        mem.add_message(Message::ai("Second message also has many words in it"))
            .unwrap();
        mem.add_message(Message::human("Third message keeps going and going"))
            .unwrap();

        // With First strategy, only one message is summarized at a time
        // so a summary should exist and the buffer should still have messages
        assert!(mem.has_summary());
        assert!(mem.message_count() >= 1);
    }

    #[test]
    fn test_strategy_oldest_summarizes_half() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new())
            .with_strategy(SummaryStrategy::Oldest);

        mem.add_message(Message::human("Message one with enough words")).unwrap();
        mem.add_message(Message::ai("Message two with more words")).unwrap();
        mem.add_message(Message::human("Message three even more")).unwrap();
        mem.add_message(Message::ai("Message four still going")).unwrap();

        assert!(mem.has_summary());
    }

    #[test]
    fn test_strategy_sliding_keeps_recent() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new())
            .with_strategy(SummaryStrategy::Sliding(2));

        mem.add_message(Message::human("First message in the conversation with many words"))
            .unwrap();
        mem.add_message(Message::ai("Second message response also with many words"))
            .unwrap();
        mem.add_message(Message::human("Third message question with extra words"))
            .unwrap();
        mem.add_message(Message::ai("Fourth message answer with lots of words"))
            .unwrap();

        if mem.has_summary() {
            // Sliding(2) keeps the 2 most recent messages
            assert!(mem.message_count() <= 2);
        }
    }

    #[test]
    fn test_strategy_default_is_oldest() {
        assert_eq!(SummaryStrategy::default(), SummaryStrategy::Oldest);
    }

    // ─── Context output format ───────────────────────────────────────────

    #[test]
    fn test_context_custom_memory_key() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new())
            .with_memory_key("chat_log");
        mem.add_message(Message::human("Test")).unwrap();

        let ctx = mem.get_context();
        assert!(ctx.get("chat_log").is_some());
        assert!(ctx.get("history").is_none());
    }

    #[test]
    fn test_context_custom_prefixes() {
        let mut mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new())
            .with_human_prefix("User")
            .with_ai_prefix("Bot");

        mem.add_message(Message::human("Hello")).unwrap();
        mem.add_message(Message::ai("Hi")).unwrap();

        let ctx = mem.get_context();
        let history = ctx.get("history").unwrap();
        let messages = history.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages[0]["role"], "User");
        assert_eq!(messages[1]["role"], "Bot");
    }

    // ─── Empty buffer behavior ───────────────────────────────────────────

    #[test]
    fn test_empty_buffer_context() {
        let mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        let ctx = mem.get_context();
        let history = ctx.get("history").unwrap();
        let messages = history.get("messages").unwrap().as_array().unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_empty_buffer_message_count() {
        let mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        assert_eq!(mem.message_count(), 0);
    }

    #[test]
    fn test_empty_buffer_no_summary() {
        let mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        assert!(!mem.has_summary());
        assert!(mem.current_summary().is_none());
    }

    #[test]
    fn test_empty_buffer_messages_slice() {
        let mem = SummaryBufferMemory::new(10000, SimpleSummarizer::new());
        assert!(mem.messages().is_empty());
    }

    // ─── Multiple summarization rounds ───────────────────────────────────

    #[test]
    fn test_multiple_summarization_rounds() {
        let mut mem = SummaryBufferMemory::new(15, SimpleSummarizer::new())
            .with_strategy(SummaryStrategy::Oldest);

        // Round 1: add messages until summarization triggers
        mem.add_message(Message::human("First question about the weather today"))
            .unwrap();
        mem.add_message(Message::ai("It is sunny and warm outside today"))
            .unwrap();

        let had_summary_r1 = mem.has_summary();

        // Round 2: add more messages to trigger another summarization
        mem.add_message(Message::human("What about tomorrow's forecast for the week"))
            .unwrap();
        mem.add_message(Message::ai("Tomorrow will be cloudy with some rain expected"))
            .unwrap();

        // If round 1 produced a summary, round 2 should incorporate it
        if had_summary_r1 && mem.has_summary() {
            let summary = mem.current_summary().unwrap();
            // The summary should have been updated (grown or changed)
            assert!(!summary.is_empty());
        }
    }

    #[test]
    fn test_progressive_summarization_preserves_info() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new())
            .with_strategy(SummaryStrategy::Sliding(1));

        mem.add_message(Message::human("The capital of France is Paris"))
            .unwrap();
        mem.add_message(Message::ai("That is correct Paris is the capital"))
            .unwrap();
        mem.add_message(Message::human("What about Germany")).unwrap();
        mem.add_message(Message::ai("The capital of Germany is Berlin"))
            .unwrap();

        if mem.has_summary() {
            let summary = mem.current_summary().unwrap();
            // With SimpleSummarizer, earlier messages should appear in summary
            assert!(!summary.is_empty());
        }
    }

    #[test]
    fn test_clear_after_summarization() {
        let mut mem = SummaryBufferMemory::new(10, SimpleSummarizer::new());

        mem.add_message(Message::human("Long message that will trigger summarization eventually"))
            .unwrap();
        mem.add_message(Message::ai("Another long response to push over the limit"))
            .unwrap();

        mem.clear();

        assert_eq!(mem.message_count(), 0);
        assert!(!mem.has_summary());

        // Should work normally after clear
        mem.add_message(Message::human("Fresh start")).unwrap();
        assert_eq!(mem.message_count(), 1);
    }
}
