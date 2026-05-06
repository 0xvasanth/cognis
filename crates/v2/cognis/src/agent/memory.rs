//! Conversation memory — the per-agent message buffer.
//!
//! Slice 1 ships `Window` (FIFO drop with system message pinned).
//! Token-aware (`TokenBudget`), summary-buffer (`SummaryBuffer`), and
//! vector-backed (`VectorMemory`) impls land in slice 2.

use std::collections::VecDeque;

use cognis2_core::Message;

/// Pluggable memory backend. The `Agent` reads via `seed()` to build
/// initial state, and writes incremental messages via `write()`.
pub trait Memory: Send + Sync {
    /// All currently buffered messages.
    fn read(&self) -> &[Message];

    /// Append one message.
    fn write(&mut self, msg: Message);

    /// Clear all buffered messages (system pinned ones survive in the Window impl).
    fn clear(&mut self);

    /// Build the seed messages for a fresh graph run. Default: `read().to_vec()`.
    fn seed(&self) -> Vec<Message> {
        self.read().to_vec()
    }
}

/// Bounded-capacity sliding window. Drops oldest non-system messages
/// when capacity is hit. The system message (if pinned) is kept at
/// index 0 across all writes and clears.
#[derive(Debug, Clone)]
pub struct Window {
    capacity: usize,
    system_pinned: Option<Message>,
    buf: VecDeque<Message>,
}

impl Window {
    /// New empty window with the given capacity (for non-system messages).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            system_pinned: None,
            buf: VecDeque::with_capacity(capacity),
        }
    }

    /// Pin a system message that survives writes and clears.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_pinned = Some(Message::system(prompt));
        self
    }
}

impl Memory for Window {
    fn read(&self) -> &[Message] {
        // Build a temp slice including system_pinned at the start. Since
        // `&[Message]` requires contiguous storage and we keep system
        // separate, we expose the buf only here. `seed()` (overridden
        // below) handles the merge for callers that need both.
        self.buf.as_slices().0
    }

    fn write(&mut self, msg: Message) {
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(msg);
    }

    fn clear(&mut self) {
        self.buf.clear();
    }

    fn seed(&self) -> Vec<Message> {
        let mut out = Vec::with_capacity(self.buf.len() + 1);
        if let Some(s) = &self.system_pinned {
            out.push(s.clone());
        }
        out.extend(self.buf.iter().cloned());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_below_capacity() {
        let mut w = Window::new(5);
        w.write(Message::human("a"));
        w.write(Message::human("b"));
        assert_eq!(w.seed().len(), 2);
    }

    #[test]
    fn fifo_drop_above_capacity() {
        let mut w = Window::new(2);
        w.write(Message::human("1"));
        w.write(Message::human("2"));
        w.write(Message::human("3"));
        let seed = w.seed();
        assert_eq!(seed.len(), 2);
        assert_eq!(seed[0].content(), "2");
        assert_eq!(seed[1].content(), "3");
    }

    #[test]
    fn system_pinned_survives_clear() {
        let mut w = Window::new(5).with_system("you are helpful");
        w.write(Message::human("hi"));
        w.clear();
        let seed = w.seed();
        assert_eq!(seed.len(), 1);
        assert_eq!(seed[0].content(), "you are helpful");
    }

    #[test]
    fn system_pinned_at_index_0() {
        let mut w = Window::new(5).with_system("system!");
        w.write(Message::human("u1"));
        w.write(Message::human("u2"));
        let seed = w.seed();
        assert_eq!(seed.len(), 3);
        assert_eq!(seed[0].content(), "system!");
        assert_eq!(seed[1].content(), "u1");
        assert_eq!(seed[2].content(), "u2");
    }
}
