//! Message types for LLM conversations.
//!
//! `Message` is a tagged enum. Each variant carries a small content struct;
//! the variants stay flat to keep pattern matching ergonomic.

use serde::{Deserialize, Serialize};

/// A single message in an LLM conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// User input.
    Human(HumanMessage),
    /// Assistant response.
    Ai(AiMessage),
    /// System prompt or instruction.
    System(SystemMessage),
    /// Tool execution result.
    Tool(ToolMessage),
}

/// Convenience constructors.
impl Message {
    /// Build a `Human` message with text only.
    pub fn human(content: impl Into<String>) -> Self {
        Self::Human(HumanMessage {
            content: content.into(),
            parts: Vec::new(),
        })
    }

    /// Build a `Human` message that carries multimodal parts alongside text.
    /// Providers that support multimodal will serialize the parts; others
    /// silently ignore them and use the text content alone.
    pub fn human_with_parts(
        content: impl Into<String>,
        parts: Vec<crate::content::ContentPart>,
    ) -> Self {
        Self::Human(HumanMessage {
            content: content.into(),
            parts,
        })
    }

    /// Build an `Ai` message with text only (no tool calls, no parts).
    pub fn ai(content: impl Into<String>) -> Self {
        Self::Ai(AiMessage {
            content: content.into(),
            tool_calls: Vec::new(),
            parts: Vec::new(),
        })
    }

    /// Build an `Ai` message with multimodal parts alongside text.
    pub fn ai_with_parts(
        content: impl Into<String>,
        parts: Vec<crate::content::ContentPart>,
    ) -> Self {
        Self::Ai(AiMessage {
            content: content.into(),
            tool_calls: Vec::new(),
            parts,
        })
    }

    /// Build a `System` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemMessage {
            content: content.into(),
        })
    }

    /// Build a `Tool` message.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool(ToolMessage {
            tool_call_id: call_id.into(),
            content: content.into(),
        })
    }

    /// Get the message's primary text content (empty string for messages
    /// that are tool-call-only with no text).
    pub fn content(&self) -> &str {
        match self {
            Self::Human(m) => &m.content,
            Self::Ai(m) => &m.content,
            Self::System(m) => &m.content,
            Self::Tool(m) => &m.content,
        }
    }

    /// Returns the tool calls if this is an `Ai` message; empty otherwise.
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Ai(m) => &m.tool_calls,
            _ => &[],
        }
    }

    /// True if this is an `Ai` message with at least one tool call.
    pub fn has_tool_calls(&self) -> bool {
        matches!(self, Self::Ai(m) if !m.tool_calls.is_empty())
    }

    /// Multimodal parts on a `Human` or `Ai` message. Empty for `System`
    /// / `Tool` (they don't carry multimodal content).
    pub fn parts(&self) -> &[crate::content::ContentPart] {
        match self {
            Self::Human(m) => &m.parts,
            Self::Ai(m) => &m.parts,
            _ => &[],
        }
    }
}

/// A human/user message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HumanMessage {
    /// The message text.
    pub content: String,
    /// Multimodal parts (images, audio). Empty for plain text. Providers
    /// that don't support multimodal silently ignore this field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<crate::content::ContentPart>,
}

/// An AI/assistant message, optionally carrying tool call requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AiMessage {
    /// The message text.
    pub content: String,
    /// Tool calls requested by the model (omitted from JSON when empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Multimodal parts. Empty for plain text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<crate::content::ContentPart>,
}

/// A system prompt or instruction message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMessage {
    /// The system prompt text.
    pub content: String,
}

/// A tool execution result message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolMessage {
    /// The ID of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The result content.
    pub content: String,
}

/// One tool invocation requested by the LLM in an `AiMessage`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned ID (used to match tool results back to calls).
    pub id: String,
    /// Tool name as registered with the LLM.
    pub name: String,
    /// Arguments — opaque JSON, deserialized by the tool.
    pub arguments: serde_json::Value,
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Self::human(s)
    }
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Self::human(s)
    }
}

// ---------------------------------------------------------------------------
// Streaming chunk types — partial messages emitted while a model streams.
// Concatenate via `+=` to reconstruct a complete `Message`.
// ---------------------------------------------------------------------------

/// A partial message emitted while a model streams. Variants mirror the
/// final [`Message`] enum so consumers can match on type before the full
/// message has arrived.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum MessageChunk {
    /// Streaming user-message chunk (rare — most providers stream assistant).
    Human(HumanChunk),
    /// Streaming assistant-message chunk.
    Ai(AiChunk),
    /// Streaming system-message chunk (rare).
    System(SystemChunk),
    /// Streaming tool-result chunk.
    Tool(ToolChunk),
}

/// Streaming text fragment for a `Human` message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HumanChunk {
    /// Incremental text fragment.
    pub content: String,
}

/// Streaming fragment for an `Ai` message. May carry partial text and/or
/// partial tool calls. Concatenating chunks merges text and tool calls
/// (matched by index).
///
/// `extras` is a forward-compatibility bag for provider-specific
/// metadata (finish reasons, logprobs, raw fragments) that should survive
/// round-trips without forcing the core enum to change.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct AiChunk {
    /// Incremental text fragment.
    pub content: String,
    /// Incremental tool-call updates (provider-indexed; same index in two
    /// chunks refers to the same call and the second chunk's fields
    /// extend the first).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallChunk>,
    /// Provider-specific extras (e.g. `finish_reason`, `logprobs`). Merged
    /// on `extend()` — later writes win on key collision.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

/// Streaming text fragment for a `System` message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemChunk {
    /// Incremental text fragment.
    pub content: String,
}

/// Streaming text fragment for a `Tool` message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolChunk {
    /// Tool call ID this fragment belongs to.
    pub tool_call_id: String,
    /// Incremental text fragment.
    pub content: String,
}

/// Streaming fragment of one tool call inside an [`AiChunk`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolCallChunk {
    /// Provider-indexed position. Used to merge fragments referring to the
    /// same call.
    pub index: usize,
    /// Provider-assigned ID. Set on the first fragment; subsequent
    /// fragments may leave it empty.
    pub id: String,
    /// Tool name. Set on the first fragment.
    pub name: String,
    /// Argument fragment — typically a JSON string under construction.
    pub arguments: String,
    /// Provider-specific extras (e.g. partial-finish flags). Merged on
    /// `extend()` — later writes win on key collision.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

impl MessageChunk {
    /// Plain-text fragment carried by this chunk.
    pub fn content(&self) -> &str {
        match self {
            Self::Human(c) => &c.content,
            Self::Ai(c) => &c.content,
            Self::System(c) => &c.content,
            Self::Tool(c) => &c.content,
        }
    }

    /// Concatenate `other` into `self`. Returns an error if the chunks
    /// describe different message types (mixing `Ai` + `Human`, etc.).
    pub fn extend(&mut self, other: MessageChunk) -> crate::Result<()> {
        match (self, other) {
            (Self::Human(a), Self::Human(b)) => {
                a.content.push_str(&b.content);
                Ok(())
            }
            (Self::System(a), Self::System(b)) => {
                a.content.push_str(&b.content);
                Ok(())
            }
            (Self::Tool(a), Self::Tool(b)) => {
                if a.tool_call_id.is_empty() {
                    a.tool_call_id = b.tool_call_id;
                }
                a.content.push_str(&b.content);
                Ok(())
            }
            (Self::Ai(a), Self::Ai(b)) => {
                a.content.push_str(&b.content);
                for tc in b.tool_calls {
                    match a.tool_calls.iter_mut().find(|x| x.index == tc.index) {
                        Some(existing) => {
                            if existing.id.is_empty() {
                                existing.id = tc.id;
                            }
                            if existing.name.is_empty() {
                                existing.name = tc.name;
                            }
                            existing.arguments.push_str(&tc.arguments);
                            for (k, v) in tc.extras {
                                existing.extras.insert(k, v);
                            }
                        }
                        None => a.tool_calls.push(tc),
                    }
                }
                for (k, v) in b.extras {
                    a.extras.insert(k, v);
                }
                Ok(())
            }
            _ => Err(crate::CognisError::Internal(
                "cannot merge MessageChunks of different roles".into(),
            )),
        }
    }
}

/// Reconstruct a complete [`Message`] from a stream of chunks. Returns an
/// error if the chunks span different roles or if tool-call arguments are
/// not valid JSON when reassembled.
pub fn message_from_chunks<I: IntoIterator<Item = MessageChunk>>(
    chunks: I,
) -> crate::Result<Message> {
    let mut iter = chunks.into_iter();
    let mut acc = match iter.next() {
        Some(c) => c,
        None => {
            return Err(crate::CognisError::Internal(
                "message_from_chunks: empty chunk stream".into(),
            ))
        }
    };
    for next in iter {
        acc.extend(next)?;
    }
    Ok(match acc {
        MessageChunk::Human(c) => Message::Human(HumanMessage {
            content: c.content,
            parts: Vec::new(),
        }),
        MessageChunk::System(c) => Message::System(SystemMessage { content: c.content }),
        MessageChunk::Tool(c) => Message::Tool(ToolMessage {
            tool_call_id: c.tool_call_id,
            content: c.content,
        }),
        MessageChunk::Ai(c) => {
            let tool_calls = c
                .tool_calls
                .into_iter()
                .map(|tc| {
                    let arguments = if tc.arguments.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::from_str(&tc.arguments).map_err(|e| {
                            crate::CognisError::Serialization(format!(
                                "tool call `{}` arguments: {e}",
                                tc.name
                            ))
                        })?
                    };
                    Ok(ToolCall {
                        id: tc.id,
                        name: tc.name,
                        arguments,
                    })
                })
                .collect::<crate::Result<Vec<_>>>()?;
            Message::Ai(AiMessage {
                content: c.content,
                tool_calls,
                parts: Vec::new(),
            })
        }
    })
}

// ---------------------------------------------------------------------------
// RemoveMessage marker — emitted by graph nodes to delete a specific
// message from the conversation history. The reducer/middleware applies it.
// ---------------------------------------------------------------------------

/// Marker that requests deletion of a specific message from a conversation
/// history. Tagged with the target message's ID so reducers (e.g.
/// `merge_message_history`) can locate and drop it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RemoveMessage {
    /// Stable ID of the message to remove. Use `Some("__all__")` to clear
    /// the entire history.
    pub id: String,
}

impl RemoveMessage {
    /// Sentinel ID meaning "remove every message in the history".
    pub const ALL: &'static str = "__all__";

    /// Build a `RemoveMessage` for a specific message ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// Build a `RemoveMessage` that clears the whole history.
    pub fn all() -> Self {
        Self {
            id: Self::ALL.to_string(),
        }
    }

    /// True if this marker requests clearing the entire history.
    pub fn is_all(&self) -> bool {
        self.id == Self::ALL
    }
}

// ---------------------------------------------------------------------------
// Conversation utilities — trim by token budget; merge consecutive same-
// role messages.
// ---------------------------------------------------------------------------

/// Strategy for [`trim_messages`] when the running token budget is
/// exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimStrategy {
    /// Drop the **oldest** messages first (preserve the tail). Useful for
    /// chat history where the latest turn is most relevant.
    First,
    /// Drop the **newest** messages first (preserve the head). Rare; used
    /// when the system prompt + earliest context is most important.
    Last,
}

/// Trim a message list down to fit `max_tokens`. The system prompt (if it
/// is the first message) is always preserved.
///
/// `strategy` controls which end is dropped when the budget is exceeded.
/// Returns a new `Vec<Message>`; the input is not modified.
pub fn trim_messages<T: crate::tokenizer::Tokenizer + ?Sized>(
    messages: &[Message],
    max_tokens: usize,
    tokenizer: &T,
    strategy: TrimStrategy,
) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }
    let pinned = matches!(messages.first(), Some(Message::System(_))) as usize;
    let pinned_msgs: Vec<Message> = messages[..pinned].to_vec();
    let pinned_cost: usize = pinned_msgs
        .iter()
        .map(|m| tokenizer.count(m.content()))
        .sum();
    let budget = max_tokens.saturating_sub(pinned_cost);

    let candidates: &[Message] = &messages[pinned..];
    let costs: Vec<usize> = candidates
        .iter()
        .map(|m| tokenizer.count(m.content()))
        .collect();

    let order: Vec<usize> = match strategy {
        // Drop oldest first → walk from the tail backwards.
        TrimStrategy::First => (0..candidates.len()).rev().collect(),
        // Drop newest first → walk from the head forwards.
        TrimStrategy::Last => (0..candidates.len()).collect(),
    };

    let mut keep = vec![false; candidates.len()];
    let mut running = 0usize;
    for idx in order {
        let cost = costs[idx];
        if running + cost > budget {
            break;
        }
        running += cost;
        keep[idx] = true;
    }

    let mut out = pinned_msgs;
    out.extend(candidates.iter().zip(keep.iter()).filter_map(|(m, &k)| {
        if k {
            Some(m.clone())
        } else {
            None
        }
    }));
    out
}

/// Trim with a fully user-defined predicate.
///
/// The predicate receives `(message, running_token_cost, message_index)`
/// and returns `true` to keep the message. Caller is responsible for
/// honoring any token budget — this function provides only the running
/// cost so the predicate doesn't need a closure-captured tokenizer.
///
/// Pinned-system-message preservation is *not* automatic in custom mode;
/// the predicate decides everything.
pub fn trim_messages_custom<F>(
    messages: &[Message],
    tokenizer: &dyn crate::tokenizer::Tokenizer,
    mut keep: F,
) -> Vec<Message>
where
    F: FnMut(&Message, usize, usize) -> bool,
{
    let mut out = Vec::with_capacity(messages.len());
    let mut running = 0usize;
    for (i, m) in messages.iter().enumerate() {
        let cost = tokenizer.count(m.content());
        if keep(m, running, i) {
            running += cost;
            out.push(m.clone());
        }
    }
    out
}

/// Collapse consecutive messages that share the same role into a single
/// message whose content is joined by `"\n\n"`. Tool calls and parts are
/// concatenated; tool message IDs are preserved on the first occurrence.
pub fn merge_message_runs(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for msg in messages {
        let same_role = match (out.last(), msg) {
            (Some(Message::Human(_)), Message::Human(_)) => true,
            (Some(Message::Ai(_)), Message::Ai(_)) => true,
            (Some(Message::System(_)), Message::System(_)) => true,
            // Tool messages keyed by tool_call_id should not merge across
            // different IDs even if the role matches.
            (Some(Message::Tool(a)), Message::Tool(b)) => a.tool_call_id == b.tool_call_id,
            _ => false,
        };
        if !same_role {
            out.push(msg.clone());
            continue;
        }
        let last = out.last_mut().expect("checked non-empty above");
        match (last, msg) {
            (Message::Human(a), Message::Human(b)) => {
                if !a.content.is_empty() && !b.content.is_empty() {
                    a.content.push_str("\n\n");
                }
                a.content.push_str(&b.content);
                a.parts.extend(b.parts.iter().cloned());
            }
            (Message::Ai(a), Message::Ai(b)) => {
                if !a.content.is_empty() && !b.content.is_empty() {
                    a.content.push_str("\n\n");
                }
                a.content.push_str(&b.content);
                a.tool_calls.extend(b.tool_calls.iter().cloned());
                a.parts.extend(b.parts.iter().cloned());
            }
            (Message::System(a), Message::System(b)) => {
                if !a.content.is_empty() && !b.content.is_empty() {
                    a.content.push_str("\n\n");
                }
                a.content.push_str(&b.content);
            }
            (Message::Tool(a), Message::Tool(b)) => {
                if !a.content.is_empty() && !b.content.is_empty() {
                    a.content.push_str("\n\n");
                }
                a.content.push_str(&b.content);
            }
            _ => unreachable!(),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convenience_constructors() {
        assert_eq!(Message::human("hi").content(), "hi");
        assert_eq!(Message::ai("hello").content(), "hello");
        assert_eq!(Message::system("be terse").content(), "be terse");
        let t = Message::tool("call_1", "result");
        assert_eq!(t.content(), "result");
        if let Message::Tool(tm) = t {
            assert_eq!(tm.tool_call_id, "call_1");
        }
    }

    #[test]
    fn tool_calls_accessor() {
        let m = Message::ai("none here");
        assert!(m.tool_calls().is_empty());
        assert!(!m.has_tool_calls());

        let m = Message::Ai(AiMessage {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
            parts: Vec::new(),
        });
        assert_eq!(m.tool_calls().len(), 1);
        assert!(m.has_tool_calls());
    }

    #[test]
    fn roundtrip_serde() {
        let m = Message::human("hi");
        let s = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
        assert!(s.contains("\"role\":\"human\""));
    }

    #[test]
    fn message_chunks_merge_text() {
        let mut a = MessageChunk::Ai(AiChunk {
            content: "Hel".into(),
            ..Default::default()
        });
        a.extend(MessageChunk::Ai(AiChunk {
            content: "lo".into(),
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(a.content(), "Hello");
    }

    #[test]
    fn message_chunks_merge_tool_call_arguments() {
        let mut a = MessageChunk::Ai(AiChunk {
            tool_calls: vec![ToolCallChunk {
                index: 0,
                id: "c1".into(),
                name: "search".into(),
                arguments: "{\"q\":\"ru".into(),
                ..Default::default()
            }],
            ..Default::default()
        });
        a.extend(MessageChunk::Ai(AiChunk {
            tool_calls: vec![ToolCallChunk {
                index: 0,
                arguments: "st\"}".into(),
                ..Default::default()
            }],
            ..Default::default()
        }))
        .unwrap();
        let final_msg = message_from_chunks(std::iter::once(a)).unwrap();
        let calls = final_msg.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments["q"], "rust");
    }

    #[test]
    fn message_chunks_reject_role_mix() {
        let mut a = MessageChunk::Ai(AiChunk::default());
        let err = a
            .extend(MessageChunk::Human(HumanChunk {
                content: "x".into(),
            }))
            .unwrap_err();
        assert!(matches!(err, crate::CognisError::Internal(_)));
    }

    #[test]
    fn message_from_chunks_empty_errors() {
        let err = message_from_chunks(std::iter::empty::<MessageChunk>()).unwrap_err();
        assert!(matches!(err, crate::CognisError::Internal(_)));
    }

    #[test]
    fn remove_message_constructors() {
        let r = RemoveMessage::new("m1");
        assert_eq!(r.id, "m1");
        assert!(!r.is_all());
        assert!(RemoveMessage::all().is_all());
    }

    #[test]
    fn trim_messages_drops_oldest_first() {
        let tok = crate::tokenizer::CharTokenizer;
        let msgs = vec![
            Message::system("sys"),  // 3 chars, pinned
            Message::human("aaaaa"), // 5
            Message::ai("bbbbb"),    // 5
            Message::human("ccccc"), // 5
        ];
        // Budget 13 → pin 3, leave 10 → keep two trailing 5-char msgs.
        let out = trim_messages(&msgs, 13, &tok, TrimStrategy::First);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].content(), "sys");
        assert_eq!(out[1].content(), "bbbbb");
        assert_eq!(out[2].content(), "ccccc");
    }

    #[test]
    fn trim_messages_drops_newest_first() {
        let tok = crate::tokenizer::CharTokenizer;
        let msgs = vec![
            Message::human("aaaaa"),
            Message::human("bbbbb"),
            Message::human("ccccc"),
        ];
        let out = trim_messages(&msgs, 10, &tok, TrimStrategy::Last);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content(), "aaaaa");
        assert_eq!(out[1].content(), "bbbbb");
    }

    #[test]
    fn trim_messages_returns_empty_when_budget_too_small_and_no_system() {
        let tok = crate::tokenizer::CharTokenizer;
        let msgs = vec![Message::human("longtext")];
        let out = trim_messages(&msgs, 3, &tok, TrimStrategy::First);
        assert!(out.is_empty());
    }

    #[test]
    fn merge_message_runs_collapses_consecutive_same_role() {
        let msgs = vec![
            Message::system("sys"),
            Message::human("a"),
            Message::human("b"),
            Message::ai("c"),
            Message::human("d"),
            Message::human("e"),
        ];
        let out = merge_message_runs(&msgs);
        assert_eq!(out.len(), 4);
        assert_eq!(out[1].content(), "a\n\nb");
        assert_eq!(out[3].content(), "d\n\ne");
    }

    #[test]
    fn message_chunks_merge_extras_map() {
        let mut a = MessageChunk::Ai(AiChunk {
            content: "x".into(),
            extras: serde_json::Map::from_iter([(
                "finish_reason".to_string(),
                serde_json::Value::String("stop".into()),
            )]),
            ..Default::default()
        });
        a.extend(MessageChunk::Ai(AiChunk {
            content: "y".into(),
            extras: serde_json::Map::from_iter([(
                "logprobs".to_string(),
                serde_json::json!([{"token": "x"}]),
            )]),
            ..Default::default()
        }))
        .unwrap();
        if let MessageChunk::Ai(ref ai) = a {
            assert_eq!(ai.extras.get("finish_reason").unwrap(), "stop");
            assert!(ai.extras.contains_key("logprobs"));
        } else {
            panic!("expected Ai");
        }
    }

    #[test]
    fn trim_messages_custom_uses_predicate() {
        let tok = crate::tokenizer::CharTokenizer;
        let msgs = vec![
            Message::human("aaa"),      // 3
            Message::human("bbbbbbbb"), // 8
            Message::human("c"),        // 1
        ];
        // Keep only messages whose content starts with 'a' or 'c'.
        let out = trim_messages_custom(&msgs, &tok, |m, _running, _i| {
            m.content().starts_with('a') || m.content().starts_with('c')
        });
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].content(), "aaa");
        assert_eq!(out[1].content(), "c");
    }

    #[test]
    fn merge_message_runs_does_not_merge_tool_with_different_ids() {
        let msgs = vec![Message::tool("c1", "first"), Message::tool("c2", "second")];
        let out = merge_message_runs(&msgs);
        assert_eq!(out.len(), 2);
    }
}
