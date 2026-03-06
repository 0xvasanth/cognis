use serde_json::Value;

use super::base::{MessageContent, MessageType};
use super::chunks::{
    ChatMessageChunk, FunctionMessageChunk, HumanMessageChunk, MessageChunkTrait,
    SystemMessageChunk, ToolMessageChunk,
};
use super::{
    AIMessage, AIMessageChunk, ChatMessage, FunctionMessage, HumanMessage, Message, SystemMessage,
    ToolMessage,
};
use crate::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// get_buffer_string
// ---------------------------------------------------------------------------

/// Convert a list of messages to a readable string with role prefixes.
///
/// Supports customizable prefixes for all message types and a configurable separator.
pub fn get_buffer_string(
    messages: &[Message],
    human_prefix: &str,
    ai_prefix: &str,
) -> String {
    get_buffer_string_full(messages, human_prefix, ai_prefix, "System", "Function", "Tool", "\n")
}

/// Full version of `get_buffer_string` with all configurable prefixes and separator.
pub fn get_buffer_string_full(
    messages: &[Message],
    human_prefix: &str,
    ai_prefix: &str,
    system_prefix: &str,
    function_prefix: &str,
    tool_prefix: &str,
    message_separator: &str,
) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let role = match msg.message_type() {
            MessageType::Human => human_prefix.to_string(),
            MessageType::Ai => ai_prefix.to_string(),
            MessageType::System => system_prefix.to_string(),
            MessageType::Tool => tool_prefix.to_string(),
            MessageType::Function => function_prefix.to_string(),
            MessageType::Chat => {
                if let Message::Chat(cm) = msg {
                    cm.role.clone()
                } else if let Message::ChatChunk(cm) = msg {
                    cm.role.clone()
                } else {
                    "Chat".to_string()
                }
            }
            MessageType::Remove => continue,
        };

        let mut text = msg.content().text();

        // For AI messages with tool_calls, append them (matching Python behavior)
        if let Message::Ai(ai) = msg {
            if !ai.tool_calls.is_empty() {
                let tc_str = serde_json::to_string(&ai.tool_calls).unwrap_or_default();
                text.push_str(&tc_str);
            }
        }

        parts.push(format!("{}: {}", role, text));
    }
    parts.join(message_separator)
}

// ---------------------------------------------------------------------------
// MessageLike — flexible input for convert_to_messages
// ---------------------------------------------------------------------------

/// Flexible representation of a message, matching Python's MessageLikeRepresentation.
pub enum MessageLike {
    /// An already-constructed Message.
    Msg(Box<Message>),
    /// A plain string (treated as human message).
    Text(String),
    /// A (role, content) pair.
    Tuple(String, String),
    /// A dict with "role" or "type" + "content" keys.
    Dict(Value),
}

/// Convert any `MessageLike` to a `Message`.
fn convert_single(item: MessageLike) -> Result<Message> {
    match item {
        MessageLike::Msg(m) => Ok(*m),
        MessageLike::Text(s) => Ok(Message::Human(HumanMessage::new(s))),
        MessageLike::Tuple(role, content) => Ok(create_message_from_role(&role, &content)),
        MessageLike::Dict(v) => {
            let obj = v
                .as_object()
                .ok_or_else(|| RustChainError::Other("Expected JSON object for message dict".into()))?;
            let role = obj
                .get("role")
                .or_else(|| obj.get("type"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    RustChainError::Other(
                        "Message dict must have 'role' or 'type' key".into(),
                    )
                })?;
            let content = obj
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(create_message_from_role(role, content))
        }
    }
}

fn create_message_from_role(role: &str, content: &str) -> Message {
    match role.to_lowercase().as_str() {
        "human" | "user" => Message::Human(HumanMessage::new(content)),
        "ai" | "assistant" => Message::Ai(AIMessage::new(content)),
        "system" | "developer" => Message::System(SystemMessage::new(content)),
        "tool" => Message::Tool(ToolMessage::new(content, "")),
        "function" => Message::Function(FunctionMessage::new("", content)),
        "remove" => Message::Remove(super::RemoveMessage::new(content)),
        _ => Message::Chat(ChatMessage::new(role, content)),
    }
}

/// Convert `(role, content)` tuples to Message variants.
///
/// This is the simple tuple-only API. For richer input types, use `convert_to_messages_flex`.
pub fn convert_to_messages(input: Vec<(String, String)>) -> Vec<Message> {
    input
        .into_iter()
        .map(|(role, content)| create_message_from_role(&role, &content))
        .collect()
}

/// Convert flexible message representations to Message variants.
///
/// Accepts Messages, strings, (role, content) tuples, and dicts.
pub fn convert_to_messages_flex(input: Vec<MessageLike>) -> Result<Vec<Message>> {
    input.into_iter().map(convert_single).collect()
}

// ---------------------------------------------------------------------------
// filter_messages
// ---------------------------------------------------------------------------

/// Filter messages by name, type, or id with include/exclude logic.
///
/// `exclude_tool_calls`: if `true`, drops AIMessages with tool_calls and ToolMessages.
pub fn filter_messages(
    messages: &[Message],
    include_names: Option<&[&str]>,
    include_types: Option<&[MessageType]>,
    exclude_names: Option<&[&str]>,
    exclude_types: Option<&[MessageType]>,
    exclude_ids: Option<&[&str]>,
) -> Vec<Message> {
    filter_messages_full(
        messages,
        include_names,
        include_types,
        None,
        exclude_names,
        exclude_types,
        exclude_ids,
        false,
    )
}

/// Full filter with all options including `include_ids` and `exclude_tool_calls`.
#[allow(clippy::too_many_arguments)]
pub fn filter_messages_full(
    messages: &[Message],
    include_names: Option<&[&str]>,
    include_types: Option<&[MessageType]>,
    include_ids: Option<&[&str]>,
    exclude_names: Option<&[&str]>,
    exclude_types: Option<&[MessageType]>,
    exclude_ids: Option<&[&str]>,
    exclude_tool_calls: bool,
) -> Vec<Message> {
    messages
        .iter()
        .filter(|msg| {
            // Exclusion phase
            if let Some(types) = exclude_types {
                if types.contains(&msg.message_type()) {
                    return false;
                }
            }
            if let Some(names) = exclude_names {
                if let Some(n) = msg.base().and_then(|b| b.name.as_deref()) {
                    if names.contains(&n) {
                        return false;
                    }
                }
            }
            if let Some(ids) = exclude_ids {
                if let Some(id) = msg.base().and_then(|b| b.id.as_deref()) {
                    if ids.contains(&id) {
                        return false;
                    }
                }
            }
            // exclude_tool_calls: drop AI messages with tool_calls and ToolMessages
            if exclude_tool_calls {
                if let Message::Ai(ai) = msg {
                    if !ai.tool_calls.is_empty() {
                        return false;
                    }
                }
                if matches!(msg, Message::Tool(_) | Message::ToolChunk(_)) {
                    return false;
                }
            }

            // Inclusion phase
            let has_include = include_names.is_some() || include_types.is_some() || include_ids.is_some();
            if has_include {
                let mut included = false;
                if let Some(types) = include_types {
                    if types.contains(&msg.message_type()) {
                        included = true;
                    }
                }
                if let Some(names) = include_names {
                    if let Some(n) = msg.base().and_then(|b| b.name.as_deref()) {
                        if names.contains(&n) {
                            included = true;
                        }
                    }
                }
                if let Some(ids) = include_ids {
                    if let Some(id) = msg.base().and_then(|b| b.id.as_deref()) {
                        if ids.contains(&id) {
                            included = true;
                        }
                    }
                }
                if !included {
                    return false;
                }
            }

            true
        })
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// merge_message_runs
// ---------------------------------------------------------------------------

/// Convert a Message to a chunk variant for merging.
fn msg_to_chunk(msg: &Message) -> Message {
    match msg {
        Message::Human(m) => Message::HumanChunk(HumanMessageChunk {
            base: m.base.clone(),
        }),
        Message::Ai(m) => Message::AiChunk(AIMessageChunk {
            base: m.base.clone(),
            tool_calls: m.tool_calls.clone(),
            invalid_tool_calls: m.invalid_tool_calls.clone(),
            tool_call_chunks: Vec::new(),
            usage_metadata: m.usage_metadata.clone(),
            chunk_position: None,
        }),
        Message::System(m) => Message::SystemChunk(SystemMessageChunk {
            base: m.base.clone(),
        }),
        Message::Tool(m) => Message::ToolChunk(ToolMessageChunk {
            base: m.base.clone(),
            tool_call_id: m.tool_call_id.clone(),
            tool_call_chunks: Vec::new(),
            artifact: m.artifact.clone(),
            status: m.status,
        }),
        Message::Function(m) => Message::FunctionChunk(FunctionMessageChunk {
            base: m.base.clone(),
        }),
        Message::Chat(m) => Message::ChatChunk(ChatMessageChunk {
            role: m.role.clone(),
            base: m.base.clone(),
        }),
        // Already chunks or Remove — pass through
        other => other.clone(),
    }
}

/// Add two chunk Messages together using their chunk semantics.
fn add_chunks(left: Message, right: Message) -> Message {
    match (left, right) {
        (Message::HumanChunk(l), Message::HumanChunk(r)) => Message::HumanChunk(l.add(r)),
        (Message::AiChunk(l), Message::AiChunk(r)) => Message::AiChunk(l.add(r)),
        (Message::SystemChunk(l), Message::SystemChunk(r)) => Message::SystemChunk(l.add(r)),
        (Message::ToolChunk(l), Message::ToolChunk(r)) => Message::ToolChunk(l.add(r)),
        (Message::FunctionChunk(l), Message::FunctionChunk(r)) => Message::FunctionChunk(l.add(r)),
        (Message::ChatChunk(l), Message::ChatChunk(r)) => Message::ChatChunk(l.add(r)),
        // Fallback: just keep left (shouldn't happen if types match)
        (l, _) => l,
    }
}

/// Merge consecutive messages of the same type using chunk semantics.
///
/// ToolMessages are never merged (they have unique tool_call_ids).
/// Content between merged messages is separated by `chunk_separator`.
pub fn merge_message_runs(messages: &[Message]) -> Vec<Message> {
    merge_message_runs_with_separator(messages, "\n")
}

/// Merge consecutive messages of the same type with a configurable separator.
pub fn merge_message_runs_with_separator(messages: &[Message], chunk_separator: &str) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Message> = Vec::new();

    for msg in messages {
        // ToolMessages are never merged (unique tool_call_ids)
        let is_tool = matches!(msg, Message::Tool(_) | Message::ToolChunk(_));

        let should_merge = !is_tool
            && result
                .last()
                .map(|last| {
                    let same_type = std::mem::discriminant(last) == std::mem::discriminant(msg)
                        || last.message_type() == msg.message_type();
                    let last_is_tool = matches!(last, Message::Tool(_) | Message::ToolChunk(_));
                    same_type && !last_is_tool
                })
                .unwrap_or(false);

        if should_merge {
            let last = result.pop().unwrap();
            let left_chunk = msg_to_chunk(&last);
            let mut right_chunk = msg_to_chunk(msg);

            // Insert chunk_separator between non-empty string contents
            let left_text = left_chunk.content().text();
            let right_text = right_chunk.content().text();
            if !left_text.is_empty() && !right_text.is_empty() && !chunk_separator.is_empty() {
                // Prepend separator to right chunk's content
                if let Some(base) = match &mut right_chunk {
                    Message::HumanChunk(c) => Some(&mut c.base),
                    Message::AiChunk(c) => Some(&mut c.base),
                    Message::SystemChunk(c) => Some(&mut c.base),
                    Message::FunctionChunk(c) => Some(&mut c.base),
                    Message::ChatChunk(c) => Some(&mut c.base),
                    Message::ToolChunk(c) => Some(&mut c.base),
                    _ => None,
                } {
                    let new_content = format!("{}{}", chunk_separator, right_text);
                    base.content = MessageContent::Text(new_content);
                }
            }

            // Clear response_metadata on the incoming chunk to avoid stale metadata
            if let Some(base) = match &mut right_chunk {
                Message::HumanChunk(c) => Some(&mut c.base),
                Message::AiChunk(c) => Some(&mut c.base),
                Message::SystemChunk(c) => Some(&mut c.base),
                Message::FunctionChunk(c) => Some(&mut c.base),
                Message::ChatChunk(c) => Some(&mut c.base),
                Message::ToolChunk(c) => Some(&mut c.base),
                _ => None,
            } {
                base.response_metadata.clear();
            }

            let merged_chunk = add_chunks(left_chunk, right_chunk);
            // Convert back to non-chunk message
            result.push(message_chunk_to_message(&merged_chunk));
        } else {
            result.push(msg.clone());
        }
    }

    result
}

// ---------------------------------------------------------------------------
// trim_messages
// ---------------------------------------------------------------------------

/// Strategy for trimming messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimStrategy {
    /// Keep messages from the beginning up to max_tokens.
    First,
    /// Keep messages from the end up to max_tokens.
    Last,
}

/// Trim messages to fit within a token budget.
///
/// - `include_system`: when using `Last` strategy, preserve the first message if it's a SystemMessage.
/// - `start_on`: when using `Last` strategy, discard messages before the first matching type.
/// - `end_on`: discard messages after the last matching type.
pub fn trim_messages(
    messages: &[Message],
    max_tokens: usize,
    token_counter: &dyn Fn(&str) -> usize,
    strategy: TrimStrategy,
) -> Vec<Message> {
    trim_messages_full(messages, max_tokens, token_counter, strategy, false, None, None)
}

/// Full trim with `include_system`, `start_on`, and `end_on` options.
pub fn trim_messages_full(
    messages: &[Message],
    max_tokens: usize,
    token_counter: &dyn Fn(&str) -> usize,
    strategy: TrimStrategy,
    include_system: bool,
    start_on: Option<&[MessageType]>,
    end_on: Option<&[MessageType]>,
) -> Vec<Message> {
    match strategy {
        TrimStrategy::First => {
            let mut result = Vec::new();
            let mut total = 0;
            for msg in messages {
                let tokens = token_counter(&msg.content().text());
                if total + tokens > max_tokens {
                    break;
                }
                total += tokens;
                result.push(msg.clone());
            }
            // Apply end_on: pop from end until last message matches
            if let Some(types) = end_on {
                while let Some(last) = result.last() {
                    if types.contains(&last.message_type()) {
                        break;
                    }
                    result.pop();
                }
            }
            result
        }
        TrimStrategy::Last => {
            let mut system_msg: Option<Message> = None;
            let mut working = messages.to_vec();
            let mut budget = max_tokens;

            // Extract leading system message if include_system
            if include_system {
                if let Some(first) = working.first() {
                    if first.message_type() == MessageType::System {
                        let sys = working.remove(0);
                        let sys_tokens = token_counter(&sys.content().text());
                        budget = budget.saturating_sub(sys_tokens);
                        system_msg = Some(sys);
                    }
                }
            }

            // Apply end_on: pop from end until last message matches
            if let Some(types) = end_on {
                while let Some(last) = working.last() {
                    if types.contains(&last.message_type()) {
                        break;
                    }
                    working.pop();
                }
            }

            // Collect from the end
            let mut result = Vec::new();
            let mut total = 0;
            for msg in working.iter().rev() {
                let tokens = token_counter(&msg.content().text());
                if total + tokens > budget {
                    break;
                }
                total += tokens;
                result.push(msg.clone());
            }
            result.reverse();

            // Apply start_on: drop from front until first message matches
            if let Some(types) = start_on {
                while let Some(first) = result.first() {
                    if types.contains(&first.message_type()) {
                        break;
                    }
                    result.remove(0);
                }
            }

            // Prepend system message
            if let Some(sys) = system_msg {
                result.insert(0, sys);
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// message_chunk_to_message
// ---------------------------------------------------------------------------

/// Convert a chunk variant to a regular message variant.
pub fn message_chunk_to_message(chunk: &Message) -> Message {
    match chunk {
        Message::HumanChunk(c) => Message::Human(HumanMessage {
            base: c.base.clone(),
        }),
        Message::AiChunk(c) => Message::Ai(AIMessage {
            base: c.base.clone(),
            tool_calls: c.tool_calls.clone(),
            invalid_tool_calls: c.invalid_tool_calls.clone(),
            usage_metadata: c.usage_metadata.clone(),
        }),
        Message::SystemChunk(c) => Message::System(SystemMessage {
            base: c.base.clone(),
        }),
        Message::ToolChunk(c) => Message::Tool(ToolMessage {
            base: c.base.clone(),
            tool_call_id: c.tool_call_id.clone(),
            artifact: c.artifact.clone(),
            status: c.status,
        }),
        Message::FunctionChunk(c) => Message::Function(FunctionMessage {
            base: c.base.clone(),
        }),
        Message::ChatChunk(c) => Message::Chat(ChatMessage {
            role: c.role.clone(),
            base: c.base.clone(),
        }),
        // Non-chunk variants pass through unchanged
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// messages_to_dict / messages_from_dict
// ---------------------------------------------------------------------------

/// Serialize messages to a list of JSON values.
pub fn messages_to_dict(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .filter_map(|m| serde_json::to_value(m).ok())
        .collect()
}

/// Deserialize messages from a list of JSON values.
pub fn messages_from_dict(values: &[Value]) -> Result<Vec<Message>> {
    values
        .iter()
        .map(|v| {
            serde_json::from_value::<Message>(v.clone())
                .map_err(|e| RustChainError::Other(format!("Failed to deserialize message: {}", e)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tokens::estimate_token_count;

    // -----------------------------------------------------------------------
    // trim_messages tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trim_messages_last_strategy() {
        let messages = vec![
            Message::human("First message"),
            Message::ai("Second message"),
            Message::human("Third message"),
            Message::ai("Fourth message"),
        ];
        let counter = |s: &str| estimate_token_count(s);
        let trimmed = trim_messages(&messages, 10, &counter, TrimStrategy::Last);
        // Should keep the most recent messages that fit
        assert!(!trimmed.is_empty());
        assert_eq!(
            trimmed.last().unwrap().content().text(),
            "Fourth message"
        );
        assert!(trimmed.len() <= messages.len());
    }

    #[test]
    fn test_trim_messages_first_strategy() {
        let messages = vec![
            Message::human("First message"),
            Message::ai("Second message"),
            Message::human("Third message"),
            Message::ai("Fourth message"),
        ];
        let counter = |s: &str| estimate_token_count(s);
        let trimmed = trim_messages(&messages, 10, &counter, TrimStrategy::First);
        // Should keep the oldest messages that fit
        assert!(!trimmed.is_empty());
        assert_eq!(
            trimmed.first().unwrap().content().text(),
            "First message"
        );
        assert!(trimmed.len() <= messages.len());
    }

    #[test]
    fn test_trim_messages_preserving_system() {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::human("Oldest question"),
            Message::ai("Oldest answer with lots of tokens to push it over the budget"),
            Message::human("Newest question"),
        ];
        let counter = |s: &str| estimate_token_count(s);
        let trimmed = trim_messages_full(
            &messages,
            20,
            &counter,
            TrimStrategy::Last,
            true, // include_system
            None,
            None,
        );
        // System message should always be preserved
        assert_eq!(trimmed[0].message_type(), MessageType::System);
        assert_eq!(trimmed[0].content().text(), "You are a helpful assistant.");
        // Should have dropped some older messages
        assert!(trimmed.len() < messages.len());
    }

    #[test]
    fn test_trim_messages_exact_token_boundary() {
        // Each message is "abcd" = 4 chars = 1 token
        let messages = vec![
            Message::human("abcd"),
            Message::ai("abcd"),
            Message::human("abcd"),
        ];
        let counter = |s: &str| estimate_token_count(s);
        // Budget of exactly 3 tokens should fit all 3 messages
        let trimmed = trim_messages(&messages, 3, &counter, TrimStrategy::Last);
        assert_eq!(trimmed.len(), 3);

        // Budget of exactly 2 tokens should fit only 2 messages
        let trimmed = trim_messages(&messages, 2, &counter, TrimStrategy::Last);
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed.last().unwrap().content().text(), "abcd");
    }

    #[test]
    fn test_trim_messages_all_exceeding_budget() {
        let messages = vec![
            Message::human("This is a very long message with many tokens"),
            Message::ai("Another very long message with many tokens"),
        ];
        let counter = |s: &str| estimate_token_count(s);
        // Very small budget - nothing should fit
        let trimmed = trim_messages(&messages, 1, &counter, TrimStrategy::Last);
        assert!(trimmed.is_empty());
    }

    // -----------------------------------------------------------------------
    // filter_messages tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_messages_include_types() {
        let messages = vec![
            Message::system("System prompt"),
            Message::human("Hello"),
            Message::ai("Hi there"),
            Message::human("How are you?"),
        ];
        let filtered = filter_messages(
            &messages,
            None,
            Some(&[MessageType::Human]),
            None,
            None,
            None,
        );
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|m| m.message_type() == MessageType::Human));
    }

    #[test]
    fn test_filter_messages_exclude_types() {
        let messages = vec![
            Message::system("System prompt"),
            Message::human("Hello"),
            Message::ai("Hi there"),
            Message::human("How are you?"),
        ];
        let filtered = filter_messages(
            &messages,
            None,
            None,
            None,
            Some(&[MessageType::System]),
            None,
        );
        assert_eq!(filtered.len(), 3);
        assert!(filtered.iter().all(|m| m.message_type() != MessageType::System));
    }

    // -----------------------------------------------------------------------
    // merge_message_runs tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_message_runs_consecutive_same_types() {
        let messages = vec![
            Message::human("Hello"),
            Message::human("How are you?"),
            Message::ai("I'm fine"),
            Message::ai("Thanks for asking"),
        ];
        let merged = merge_message_runs(&messages);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].message_type(), MessageType::Human);
        let content = merged[0].content().text();
        assert!(content.contains("Hello"));
        assert!(content.contains("How are you?"));
        assert_eq!(merged[1].message_type(), MessageType::Ai);
        let ai_content = merged[1].content().text();
        assert!(ai_content.contains("I'm fine"));
        assert!(ai_content.contains("Thanks for asking"));
    }

    #[test]
    fn test_merge_message_runs_no_merging_needed() {
        let messages = vec![
            Message::human("Hello"),
            Message::ai("Hi there"),
            Message::human("How are you?"),
            Message::ai("Fine, thanks"),
        ];
        let merged = merge_message_runs(&messages);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].content().text(), "Hello");
        assert_eq!(merged[1].content().text(), "Hi there");
    }

    // -----------------------------------------------------------------------
    // messages_to_dict / messages_from_dict tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_messages_to_dict_round_trip() {
        let messages = vec![
            Message::human("Hello"),
            Message::ai("World"),
            Message::system("Be helpful"),
        ];
        let dicts = messages_to_dict(&messages);
        assert_eq!(dicts.len(), 3);

        let restored = messages_from_dict(&dicts).unwrap();
        assert_eq!(restored.len(), 3);
        assert_eq!(restored[0].content().text(), "Hello");
        assert_eq!(restored[0].message_type(), MessageType::Human);
        assert_eq!(restored[1].content().text(), "World");
        assert_eq!(restored[1].message_type(), MessageType::Ai);
        assert_eq!(restored[2].content().text(), "Be helpful");
        assert_eq!(restored[2].message_type(), MessageType::System);
    }

    #[test]
    fn test_messages_from_dict_various_types() {
        // Use messages_to_dict to get the correct serialization format, then verify round-trip
        let original = vec![
            Message::human("Hi"),
            Message::ai("Hello"),
            Message::system("System"),
            Message::tool("Result", "tc_1"),
        ];
        let dicts = messages_to_dict(&original);
        assert_eq!(dicts.len(), 4);

        let messages = messages_from_dict(&dicts).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].message_type(), MessageType::Human);
        assert_eq!(messages[0].content().text(), "Hi");
        assert_eq!(messages[1].message_type(), MessageType::Ai);
        assert_eq!(messages[1].content().text(), "Hello");
        assert_eq!(messages[2].message_type(), MessageType::System);
        assert_eq!(messages[2].content().text(), "System");
        assert_eq!(messages[3].message_type(), MessageType::Tool);
        assert_eq!(messages[3].content().text(), "Result");
    }

    // -----------------------------------------------------------------------
    // Empty input tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_input_trim_messages() {
        let counter = |s: &str| estimate_token_count(s);
        let trimmed = trim_messages(&[], 100, &counter, TrimStrategy::Last);
        assert!(trimmed.is_empty());
        let trimmed = trim_messages(&[], 100, &counter, TrimStrategy::First);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_empty_input_filter_messages() {
        let filtered = filter_messages(&[], None, Some(&[MessageType::Human]), None, None, None);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_empty_input_merge_message_runs() {
        let merged = merge_message_runs(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_empty_input_messages_to_dict() {
        let dicts = messages_to_dict(&[]);
        assert!(dicts.is_empty());
        let restored = messages_from_dict(&[]).unwrap();
        assert!(restored.is_empty());
    }
}
