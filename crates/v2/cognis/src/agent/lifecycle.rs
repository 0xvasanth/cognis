//! Per-agent lifecycle hooks.
//!
//! Implementations register via [`AgentBuilder::with_lifecycle`]. The
//! agent invokes them around `run()`. Cheap to ignore — the default
//! impls do nothing.

use async_trait::async_trait;
use uuid::Uuid;

use cognis2_core::{CognisError, Message};

use super::AgentResponse;

/// Lifecycle hook surface. All methods have no-op defaults; override only
/// what you need.
#[async_trait]
pub trait AgentLifecycle: Send + Sync {
    /// Fires before the agent's graph runs.
    async fn on_start(&self, run_id: Uuid, input: &Message) {
        let _ = (run_id, input);
    }
    /// Fires when the run completes successfully.
    async fn on_stop(&self, run_id: Uuid, response: &AgentResponse) {
        let _ = (run_id, response);
    }
    /// Fires when the run errors.
    async fn on_error(&self, run_id: Uuid, error: &CognisError) {
        let _ = (run_id, error);
    }
}

/// Convenience: a closure-based lifecycle that observes only `on_start`.
pub struct OnStart<F: Fn(Uuid, &Message) + Send + Sync>(pub F);

#[async_trait]
impl<F: Fn(Uuid, &Message) + Send + Sync> AgentLifecycle for OnStart<F> {
    async fn on_start(&self, run_id: Uuid, input: &Message) {
        (self.0)(run_id, input)
    }
}
