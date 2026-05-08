//! Typed value types over `Vec<Message>` history.
//!
//! V2 stores conversations as flat `Vec<Message>` — that's the canonical
//! shape for LLM APIs. This module gives ergonomic typed views:
//!
//! - [`ConversationTurn`] — one user input + the assistant's reply +
//!   any tool messages that happened in between.
//! - [`ConversationSummary`] — aggregate counts and quick previews
//!   (first user input, last AI output, total characters).
//! - [`turns_from_messages`] / [`summarize`] convert flat history into
//!   either view.
//!
//! These are *views*, not new storage — `cognis::history` /
//! `cognis::session` still own the message list. Reach for these when
//! you want to render a transcript, generate a conversation summary, or
//! emit metrics per turn.
//!
//! ```
//! use cognis::conversation::{summarize, turns_from_messages};
//! use cognis_core::Message;
//!
//! let history = vec![
//!     Message::system("be brief"),
//!     Message::human("hello"),
//!     Message::ai("hi"),
//!     Message::human("how are you?"),
//!     Message::ai("good"),
//! ];
//! let turns = turns_from_messages(&history);
//! assert_eq!(turns.len(), 2);
//!
//! let s = summarize(&history);
//! assert_eq!(s.turn_count, 2);
//! ```

use cognis_core::Message;

/// One user-input → assistant-reply turn. Tool calls and tool result
/// messages that landed between the user and assistant entries are
/// captured in `tool_messages`. System messages are not part of any
/// turn (they're conversation-wide context).
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// The user's input. `None` for an "AI-initiated" first turn (rare —
    /// happens when an agent runs with seed context but no user prompt).
    pub user: Option<Message>,
    /// The assistant's reply. `None` for an "incomplete" trailing turn
    /// — i.e. the user has spoken but the agent hasn't responded yet.
    pub assistant: Option<Message>,
    /// Tool messages (calls + results) that occurred in this turn.
    pub tool_messages: Vec<Message>,
}

impl ConversationTurn {
    /// `true` if the turn has both a user input and an assistant reply.
    pub fn is_complete(&self) -> bool {
        self.user.is_some() && self.assistant.is_some()
    }

    /// Total messages in this turn (user + assistant + tools).
    pub fn message_count(&self) -> usize {
        self.user.as_ref().map_or(0, |_| 1)
            + self.assistant.as_ref().map_or(0, |_| 1)
            + self.tool_messages.len()
    }
}

/// Aggregate stats over a conversation's message list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversationSummary {
    /// Total turns (complete + incomplete).
    pub turn_count: usize,
    /// Number of complete turns (user + assistant pair present).
    pub complete_turn_count: usize,
    /// Total messages across the whole history.
    pub message_count: usize,
    /// Per-role breakdowns.
    pub user_message_count: usize,
    /// Per-role breakdowns.
    pub ai_message_count: usize,
    /// Per-role breakdowns.
    pub tool_message_count: usize,
    /// Per-role breakdowns.
    pub system_message_count: usize,
    /// First user input as plain text (truncated at 200 chars).
    pub first_user_input: Option<String>,
    /// Last AI output as plain text (truncated at 200 chars).
    pub last_ai_output: Option<String>,
    /// Sum of `Message::content().len()` across every message.
    pub total_chars: usize,
}

/// Group flat messages into [`ConversationTurn`]s.
///
/// Algorithm: walk the history left-to-right; on each `Human` message,
/// open a new turn; collect subsequent tool messages onto its
/// `tool_messages`; close the turn at the next `Ai`. System messages
/// are ignored (they're not part of any turn). A trailing user message
/// with no AI reply yet is emitted as an incomplete turn.
pub fn turns_from_messages(messages: &[Message]) -> Vec<ConversationTurn> {
    let mut out: Vec<ConversationTurn> = Vec::new();
    let mut current: Option<ConversationTurn> = None;
    for m in messages {
        match m {
            Message::System(_) => continue,
            Message::Human(_) => {
                // Close the prior turn, if any (it had no AI reply).
                if let Some(t) = current.take() {
                    out.push(t);
                }
                current = Some(ConversationTurn {
                    user: Some(m.clone()),
                    assistant: None,
                    tool_messages: Vec::new(),
                });
            }
            Message::Ai(_) => {
                let mut t = current.take().unwrap_or(ConversationTurn {
                    user: None,
                    assistant: None,
                    tool_messages: Vec::new(),
                });
                t.assistant = Some(m.clone());
                out.push(t);
            }
            Message::Tool(_) => match current.as_mut() {
                Some(t) => t.tool_messages.push(m.clone()),
                None => {
                    // Tool message with no open turn — start one.
                    current = Some(ConversationTurn {
                        user: None,
                        assistant: None,
                        tool_messages: vec![m.clone()],
                    });
                }
            },
        }
    }
    if let Some(t) = current.take() {
        out.push(t);
    }
    out
}

/// Compute a [`ConversationSummary`] from flat messages.
pub fn summarize(messages: &[Message]) -> ConversationSummary {
    let mut s = ConversationSummary {
        message_count: messages.len(),
        ..Default::default()
    };
    for m in messages {
        s.total_chars += m.content().len();
        match m {
            Message::System(_) => s.system_message_count += 1,
            Message::Human(_) => {
                s.user_message_count += 1;
                if s.first_user_input.is_none() {
                    s.first_user_input = Some(truncate(m.content(), 200));
                }
            }
            Message::Ai(_) => {
                s.ai_message_count += 1;
                s.last_ai_output = Some(truncate(m.content(), 200));
            }
            Message::Tool(_) => s.tool_message_count += 1,
        }
    }
    let turns = turns_from_messages(messages);
    s.turn_count = turns.len();
    s.complete_turn_count = turns.iter().filter(|t| t.is_complete()).count();
    s
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> Message {
        Message::human(s)
    }
    fn a(s: &str) -> Message {
        Message::ai(s)
    }
    fn sys(s: &str) -> Message {
        Message::system(s)
    }

    #[test]
    fn empty_messages_produce_empty_summary() {
        let s = summarize(&[]);
        assert_eq!(s, ConversationSummary::default());
        assert!(turns_from_messages(&[]).is_empty());
    }

    #[test]
    fn pairs_user_with_next_ai() {
        let msgs = vec![h("hi"), a("hello"), h("how"), a("good")];
        let turns = turns_from_messages(&msgs);
        assert_eq!(turns.len(), 2);
        assert!(turns.iter().all(|t| t.is_complete()));
    }

    #[test]
    fn trailing_user_is_incomplete_turn() {
        let msgs = vec![h("hi"), a("hello"), h("waiting")];
        let turns = turns_from_messages(&msgs);
        assert_eq!(turns.len(), 2);
        assert!(turns[0].is_complete());
        assert!(!turns[1].is_complete());
        assert_eq!(turns[1].user.as_ref().unwrap().content(), "waiting");
        assert!(turns[1].assistant.is_none());
    }

    #[test]
    fn system_messages_dont_count_as_turns() {
        let msgs = vec![sys("preamble"), h("hi"), a("hello")];
        let turns = turns_from_messages(&msgs);
        assert_eq!(turns.len(), 1);
        let s = summarize(&msgs);
        assert_eq!(s.system_message_count, 1);
        assert_eq!(s.turn_count, 1);
    }

    #[test]
    fn summary_aggregates_counts_and_previews() {
        let msgs = vec![
            sys("be brief"),
            h("first question"),
            a("first reply"),
            h("second question"),
            a("second reply"),
        ];
        let s = summarize(&msgs);
        assert_eq!(s.message_count, 5);
        assert_eq!(s.user_message_count, 2);
        assert_eq!(s.ai_message_count, 2);
        assert_eq!(s.system_message_count, 1);
        assert_eq!(s.turn_count, 2);
        assert_eq!(s.complete_turn_count, 2);
        assert_eq!(s.first_user_input.as_deref(), Some("first question"));
        assert_eq!(s.last_ai_output.as_deref(), Some("second reply"));
    }

    #[test]
    fn truncates_long_previews() {
        let long = "a".repeat(500);
        let msgs = vec![h(&long), a(&long)];
        let s = summarize(&msgs);
        let preview = s.first_user_input.unwrap();
        assert!(preview.chars().count() <= 200);
        assert!(preview.ends_with('…'));
    }
}
