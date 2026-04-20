//! Cooperative cancellation primitives.
//!
//! Provides a [`CancellationToken`] that can be cloned across tasks and
//! signalled to request cooperative cancellation of long-running work
//! (agent loops, LLM calls, tool executions, graph execution, …).
//!
//! The token pairs an atomic boolean with a [`tokio::sync::Notify`] so
//! that consumers can either poll [`CancellationToken::is_cancelled`] at
//! loop boundaries or `.await` [`CancellationToken::cancelled`] in a
//! `tokio::select!` arm to interleave cancellation with other I/O.
//!
//! # Example
//!
//! ```
//! use cognis_core::CancellationToken;
//!
//! # async fn example() {
//! let token = CancellationToken::new();
//! let clone = token.clone();
//!
//! tokio::spawn(async move {
//!     clone.cancel();
//! });
//!
//! tokio::select! {
//!     _ = token.cancelled() => {
//!         // cooperative shutdown path
//!     }
//!     _ = async { /* do work */ } => {}
//! }
//! # }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

/// Cooperative cancellation token.
///
/// Cloning a token shares the cancellation state. Signalling cancellation
/// on any clone wakes every task that is awaiting [`cancelled`](Self::cancelled)
/// on any clone.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancellationToken {
    /// Create a new, uncancelled token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Create a token that is already cancelled.
    ///
    /// Useful as a sentinel in tests and for pre-cancelled configs where
    /// the work should abort before it begins.
    pub fn cancelled_now() -> Self {
        let t = Self::new();
        t.cancel();
        t
    }

    /// Signal cancellation.
    ///
    /// All current and future awaiters of [`cancelled`](Self::cancelled)
    /// observe this immediately. Calling `cancel` multiple times is a no-op
    /// after the first call.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Returns a future that resolves when cancellation is signalled.
    ///
    /// If the token is already cancelled the future resolves immediately.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        // Wait for notification. Re-check after wake since we might get
        // spurious wakes or the token could be cancelled between check
        // and wait.
        loop {
            self.notify.notified().await;
            if self.is_cancelled() {
                return;
            }
        }
    }

    /// Reset the token so it can be reused.
    ///
    /// Note that there is no synchronisation between `reset` and any
    /// concurrent waiters; reset is intended for single-threaded reuse
    /// patterns (e.g. reusing a token across successive agent turns).
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    /// Return `Err(CognisError::Cancelled(reason))` when the token has been
    /// cancelled, otherwise `Ok(())`.
    ///
    /// Convenience helper for loop-boundary checks in synchronous code paths
    /// where a full `tokio::select!` would be overkill.
    pub fn check(&self, reason: &str) -> crate::error::Result<()> {
        if self.is_cancelled() {
            Err(crate::error::CognisError::Cancelled(reason.to_string()))
        } else {
            Ok(())
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn initial_state_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_sets_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn reset_clears_cancellation() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let token = CancellationToken::new();
        let token2 = token.clone();
        token.cancel();
        assert!(token2.is_cancelled());
    }

    #[test]
    fn default_is_not_cancelled() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[tokio::test]
    async fn cancelled_future_resolves_when_signalled() {
        let token = CancellationToken::new();
        let token2 = token.clone();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            token2.cancel();
        });

        tokio::time::timeout(Duration::from_secs(2), token.cancelled())
            .await
            .expect("cancelled() should resolve when token is cancelled");
    }

    #[tokio::test]
    async fn cancelled_future_resolves_immediately_when_already_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        tokio::time::timeout(Duration::from_millis(50), token.cancelled())
            .await
            .expect("cancelled() should resolve immediately when already cancelled");
    }

    #[test]
    fn cancelled_now_constructs_pre_cancelled() {
        let token = CancellationToken::cancelled_now();
        assert!(token.is_cancelled());
    }

    #[test]
    fn check_method_returns_ok_when_not_cancelled() {
        let token = CancellationToken::new();
        assert!(token.check("unused").is_ok());
    }

    #[test]
    fn check_method_returns_err_when_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        let err = token.check("stop now").unwrap_err();
        match err {
            crate::error::CognisError::Cancelled(reason) => assert_eq!(reason, "stop now"),
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }
}
