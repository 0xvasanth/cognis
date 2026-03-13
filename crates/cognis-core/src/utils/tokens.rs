//! Token counting and context window management utilities.
//!
//! Provides simple heuristic-based token estimation for text and message
//! sequences, a context window trimmer that preserves system messages, and a
//! lookup table for well-known model context window sizes.

use crate::messages::Message;

/// Estimate token count using the ~4 characters per token heuristic.
///
/// This is a rough approximation that works reasonably well for English text
/// across most tokenizers. For precise counts, use a model-specific tokenizer.
///
/// # Examples
///
/// ```
/// use cognis_core::utils::tokens::estimate_token_count;
///
/// assert_eq!(estimate_token_count("hello"), 2); // 5 chars / 4 = 1.25 → ceil = 2
/// assert_eq!(estimate_token_count(""), 0);
/// ```
pub fn estimate_token_count(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() as f64 / 4.0).ceil() as usize
}

/// Count tokens across a list of messages.
///
/// Each message incurs a fixed overhead of `tokens_per_message` tokens (to
/// account for the role label, separators, etc.) plus the token count of its
/// content. For AI messages that contain tool calls, the tool call name and
/// serialized arguments are also counted.
///
/// An additional 3 tokens are added at the end to account for the assistant
/// reply priming prefix.
///
/// # Arguments
///
/// * `messages` - The message sequence to count.
/// * `tokens_per_message` - Per-message overhead (OpenAI uses 3 for
///   gpt-3.5-turbo and 4 for gpt-4).
pub fn count_message_tokens(messages: &[Message], tokens_per_message: usize) -> usize {
    let mut total = 0;
    for msg in messages {
        total += tokens_per_message; // role overhead
        total += estimate_token_count(&msg.content().text());
        // Add tool call tokens if present
        if let Message::Ai(ai) = msg {
            for tc in &ai.tool_calls {
                total += estimate_token_count(&tc.name);
                total += estimate_token_count(&serde_json::to_string(&tc.args).unwrap_or_default());
            }
        }
    }
    total + 3 // every reply is primed with assistant prefix
}

/// Trim messages to fit within a token limit, keeping the most recent messages.
///
/// System messages at the start of the sequence are always preserved. From the
/// remaining (non-system) messages, the function takes as many as possible from
/// the end (most recent first) until adding another message would exceed
/// `max_tokens`.
///
/// If even the system messages alone exceed the limit, they are still returned
/// so the caller can decide how to handle the overflow.
///
/// # Arguments
///
/// * `messages` - The full message sequence.
/// * `max_tokens` - The maximum token budget.
/// * `tokens_per_message` - Per-message overhead passed to
///   [`count_message_tokens`].
///
/// # Returns
///
/// A new `Vec<Message>` containing the system messages followed by the most
/// recent non-system messages that fit within the budget.
pub fn trim_messages(
    messages: &[Message],
    max_tokens: usize,
    tokens_per_message: usize,
) -> Vec<Message> {
    // 1. Separate leading system messages from the rest.
    let mut system_messages: Vec<Message> = Vec::new();
    let mut other_messages: Vec<&Message> = Vec::new();
    let mut seen_non_system = false;

    for msg in messages {
        if !seen_non_system && msg.message_type() == crate::messages::MessageType::System {
            system_messages.push(msg.clone());
        } else {
            seen_non_system = true;
            other_messages.push(msg);
        }
    }

    // 2. Count system message tokens (always kept).
    let system_tokens = if system_messages.is_empty() {
        3 // just the assistant priming
    } else {
        count_message_tokens(&system_messages, tokens_per_message)
    };

    if system_tokens >= max_tokens {
        // System messages alone exceed the budget; return them anyway.
        return system_messages;
    }

    let remaining_budget = max_tokens - system_tokens;

    // 3. From the remaining messages, take from the end until we hit the limit.
    let mut kept: Vec<Message> = Vec::new();
    let mut used_tokens: usize = 0;

    for msg in other_messages.iter().rev() {
        let msg_tokens = tokens_per_message + estimate_token_count(&(*msg).content().text());
        // Account for tool call tokens on AI messages.
        let tool_tokens = if let Message::Ai(ai) = *msg {
            ai.tool_calls
                .iter()
                .map(|tc| {
                    estimate_token_count(&tc.name)
                        + estimate_token_count(&serde_json::to_string(&tc.args).unwrap_or_default())
                })
                .sum::<usize>()
        } else {
            0
        };
        let total_msg_tokens = msg_tokens + tool_tokens;

        if used_tokens + total_msg_tokens > remaining_budget {
            break;
        }
        used_tokens += total_msg_tokens;
        kept.push((*msg).clone());
    }

    // Reverse to restore chronological order.
    kept.reverse();

    // 4. Return system messages + trimmed recent messages.
    system_messages.extend(kept);
    system_messages
}

/// Look up the context window size (in tokens) for a well-known model.
///
/// Returns `None` if the model name is not recognized.
///
/// # Examples
///
/// ```
/// use cognis_core::utils::tokens::get_model_context_window;
///
/// assert_eq!(get_model_context_window("gpt-4o-mini"), Some(128_000));
/// assert_eq!(get_model_context_window("claude-3-opus"), Some(200_000));
/// assert_eq!(get_model_context_window("unknown-model"), None);
/// ```
pub fn get_model_context_window(model: &str) -> Option<usize> {
    match model {
        s if s.starts_with("gpt-4o") => Some(128_000),
        s if s.starts_with("gpt-4-turbo") => Some(128_000),
        s if s.starts_with("gpt-4") => Some(8_192),
        s if s.starts_with("gpt-3.5") => Some(16_385),
        s if s.starts_with("claude-3")
            || s.starts_with("claude-sonnet")
            || s.starts_with("claude-opus") =>
        {
            Some(200_000)
        }
        s if s.starts_with("gemini") => Some(1_000_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::tool_types::ToolCall;
    use crate::messages::{AIMessage, Message};
    use std::collections::HashMap;

    #[test]
    fn test_estimate_token_count_empty() {
        assert_eq!(estimate_token_count(""), 0);
    }

    #[test]
    fn test_estimate_token_count_known_text() {
        // "hello world" is 11 chars → 11/4 = 2.75 → ceil = 3
        assert_eq!(estimate_token_count("hello world"), 3);
        // "a" is 1 char → 1/4 = 0.25 → ceil = 1
        assert_eq!(estimate_token_count("a"), 1);
        // 8 chars → 8/4 = 2.0 → ceil = 2
        assert_eq!(estimate_token_count("abcdefgh"), 2);
    }

    #[test]
    fn test_count_message_tokens_basic() {
        let messages = vec![Message::human("Hello"), Message::ai("Hi there!")];
        let tokens = count_message_tokens(&messages, 3);
        // Human: 3 (overhead) + ceil(5/4)=2 = 5
        // AI:    3 (overhead) + ceil(9/4)=3 = 6
        // + 3 assistant priming
        // Total: 5 + 6 + 3 = 14
        assert_eq!(tokens, 14);
    }

    #[test]
    fn test_count_message_tokens_with_tool_calls() {
        let mut args = HashMap::new();
        args.insert("query".to_string(), serde_json::json!("weather"));
        let tc = ToolCall {
            name: "search".to_string(),
            args,
            id: Some("tc_1".to_string()),
        };
        let ai_msg = AIMessage::new("Let me search").with_tool_calls(vec![tc]);
        let messages = vec![Message::Ai(ai_msg)];
        let tokens = count_message_tokens(&messages, 3);
        // overhead=3, content "Let me search" (13 chars → 4), tool name "search" (6 chars → 2),
        // tool args serialized {"query":"weather"} (19 chars → 5)
        // 3 + 4 + 2 + 5 + 3 = 17
        assert_eq!(tokens, 17);
    }

    #[test]
    fn test_trim_messages_keeps_system() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::human("Oldest question"),
            Message::ai("Oldest answer"),
            Message::human("Newest question"),
        ];
        // Use a small budget that can fit system + newest question only
        let trimmed = trim_messages(&messages, 20, 3);
        // System message should always be first
        assert_eq!(
            trimmed[0].message_type(),
            crate::messages::MessageType::System
        );
        // Should have dropped some older messages
        assert!(trimmed.len() < messages.len());
        // Last message should be the newest question
        assert_eq!(trimmed.last().unwrap().content().text(), "Newest question");
    }

    #[test]
    fn test_trim_messages_removes_oldest_first() {
        let messages = vec![
            Message::human("First"),
            Message::ai("Second"),
            Message::human("Third"),
            Message::ai("Fourth"),
        ];
        // Budget that fits ~2 messages + priming
        let trimmed = trim_messages(&messages, 15, 3);
        // Should keep most recent messages
        if trimmed.len() < messages.len() {
            // The last message in trimmed should be "Fourth"
            assert_eq!(trimmed.last().unwrap().content().text(), "Fourth");
        }
    }

    #[test]
    fn test_trim_messages_all_fit() {
        let messages = vec![Message::system("Be helpful."), Message::human("Hi")];
        let trimmed = trim_messages(&messages, 1000, 3);
        assert_eq!(trimmed.len(), messages.len());
    }

    #[test]
    fn test_get_model_context_window_known() {
        assert_eq!(get_model_context_window("gpt-4o"), Some(128_000));
        assert_eq!(get_model_context_window("gpt-4o-mini"), Some(128_000));
        assert_eq!(
            get_model_context_window("gpt-4-turbo-preview"),
            Some(128_000)
        );
        assert_eq!(get_model_context_window("gpt-4"), Some(8_192));
        assert_eq!(get_model_context_window("gpt-3.5-turbo"), Some(16_385));
        assert_eq!(get_model_context_window("claude-3-opus"), Some(200_000));
        assert_eq!(get_model_context_window("claude-sonnet-4"), Some(200_000));
        assert_eq!(get_model_context_window("claude-opus-4"), Some(200_000));
        assert_eq!(get_model_context_window("gemini-pro"), Some(1_000_000));
    }

    #[test]
    fn test_get_model_context_window_unknown() {
        assert_eq!(get_model_context_window("unknown-model"), None);
        assert_eq!(get_model_context_window("llama-3"), None);
    }
}
