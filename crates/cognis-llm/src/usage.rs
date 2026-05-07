//! Running-total usage tracker for budget enforcement.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::chat::Usage;

/// Aggregates `Usage` across many `ChatResponse`s. Cheap to share via `Arc`.
#[derive(Debug, Default)]
pub struct UsageTracker {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    total_tokens: AtomicU64,
    requests: AtomicU64,
}

impl UsageTracker {
    /// Empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one `Usage` to the running totals.
    pub fn record(&self, u: &Usage) {
        self.prompt_tokens
            .fetch_add(u.prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(u.completion_tokens as u64, Ordering::Relaxed);
        self.total_tokens
            .fetch_add(u.total_tokens as u64, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the totals as a `Usage` (truncates `u64 → u32`).
    pub fn snapshot(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed) as u32,
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed) as u32,
            total_tokens: self.total_tokens.load(Ordering::Relaxed) as u32,
        }
    }

    /// Number of recorded calls.
    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
        self.requests.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot() {
        let t = UsageTracker::new();
        t.record(&Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        });
        t.record(&Usage {
            prompt_tokens: 20,
            completion_tokens: 8,
            total_tokens: 28,
        });
        let snap = t.snapshot();
        assert_eq!(snap.prompt_tokens, 30);
        assert_eq!(snap.completion_tokens, 13);
        assert_eq!(snap.total_tokens, 43);
        assert_eq!(t.requests(), 2);
    }

    #[test]
    fn reset_zeros_everything() {
        let t = UsageTracker::new();
        t.record(&Usage {
            prompt_tokens: 5,
            completion_tokens: 5,
            total_tokens: 10,
        });
        t.reset();
        assert_eq!(t.requests(), 0);
        assert_eq!(t.snapshot().total_tokens, 0);
    }
}
