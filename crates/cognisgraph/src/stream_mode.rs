//! Granular streaming modes for `CompiledGraph`.
//!
//! Compared to the default [`cognis_core::Runnable::stream_events`] (which
//! emits the full structured `Event` taxonomy), [`StreamMode`] gives callers
//! a coarser, name-keyed control over what they observe:
//!
//! | Mode          | What's emitted                                                  |
//! |---------------|-----------------------------------------------------------------|
//! | `Values`      | Whole state after every superstep (requires `S: Serialize`).    |
//! | `Updates`     | `(node_name, update_json)` per node end (requires `Update: Serialize`). |
//! | `Messages`    | Per-token / per-tool-call deltas (forwarded `OnLlmToken` etc.). |
//! | `Tasks`       | `OnNodeStart` only — task scheduling.                           |
//! | `Checkpoints` | One emit per persisted checkpoint.                              |
//! | `Debug`       | Everything: every `Event` variant — equivalent to `stream_events`. |
//! | `Custom`      | Only `Event::Custom` payloads written by nodes via `NodeCtx::write`. |
//!
//! Multiple modes can be requested; events are kept if they match any mode.

use cognis_core::Event;

/// Selectable stream-output modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// Whole state at every superstep boundary.
    Values,
    /// Per-node delta after each node finishes.
    Updates,
    /// LLM token / tool-call deltas.
    Messages,
    /// Node-start signals (scheduling).
    Tasks,
    /// Each persisted checkpoint.
    Checkpoints,
    /// All events (no filtering).
    Debug,
    /// Only `Event::Custom` payloads.
    Custom,
}

impl StreamMode {
    /// Does this mode want this event?
    pub fn matches(self, event: &Event) -> bool {
        match self {
            StreamMode::Debug => true,
            StreamMode::Values => matches!(event, Event::OnEnd { .. }),
            StreamMode::Updates => matches!(event, Event::OnNodeEnd { .. }),
            StreamMode::Tasks => matches!(event, Event::OnNodeStart { .. }),
            StreamMode::Messages => matches!(
                event,
                Event::OnLlmToken { .. } | Event::OnToolStart { .. } | Event::OnToolEnd { .. }
            ),
            StreamMode::Checkpoints => matches!(event, Event::OnCheckpoint { .. }),
            StreamMode::Custom => matches!(event, Event::Custom { .. }),
        }
    }
}

/// A set of selected modes. Keeps an event if any mode matches.
#[derive(Debug, Clone, Default)]
pub struct StreamModes(Vec<StreamMode>);

impl StreamModes {
    /// Empty selector — emits nothing.
    pub fn none() -> Self {
        Self(Vec::new())
    }

    /// Select `Debug` (everything).
    pub fn debug() -> Self {
        Self(vec![StreamMode::Debug])
    }

    /// Select one mode.
    pub fn only(mode: StreamMode) -> Self {
        Self(vec![mode])
    }

    /// Build from an explicit list.
    pub fn from_modes<I: IntoIterator<Item = StreamMode>>(modes: I) -> Self {
        Self(modes.into_iter().collect())
    }

    /// Add another mode.
    pub fn with(mut self, mode: StreamMode) -> Self {
        if !self.0.contains(&mode) {
            self.0.push(mode);
        }
        self
    }

    /// True if `event` matches any selected mode.
    pub fn matches(&self, event: &Event) -> bool {
        self.0.iter().any(|m| m.matches(event))
    }

    /// All selected modes.
    pub fn modes(&self) -> &[StreamMode] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn ev_node_start() -> Event {
        Event::OnNodeStart {
            node: "x".into(),
            step: 0,
            run_id: Uuid::nil(),
        }
    }
    fn ev_node_end() -> Event {
        Event::OnNodeEnd {
            node: "x".into(),
            step: 0,
            output: serde_json::Value::Null,
            run_id: Uuid::nil(),
        }
    }
    fn ev_on_end() -> Event {
        Event::OnEnd {
            runnable: "graph".into(),
            run_id: Uuid::nil(),
            output: serde_json::Value::Null,
        }
    }
    fn ev_token() -> Event {
        Event::OnLlmToken {
            token: "t".into(),
            run_id: Uuid::nil(),
        }
    }

    #[test]
    fn debug_matches_everything() {
        assert!(StreamMode::Debug.matches(&ev_node_start()));
        assert!(StreamMode::Debug.matches(&ev_node_end()));
        assert!(StreamMode::Debug.matches(&ev_token()));
    }

    #[test]
    fn tasks_matches_only_starts() {
        assert!(StreamMode::Tasks.matches(&ev_node_start()));
        assert!(!StreamMode::Tasks.matches(&ev_node_end()));
    }

    #[test]
    fn updates_matches_only_node_end() {
        assert!(StreamMode::Updates.matches(&ev_node_end()));
        assert!(!StreamMode::Updates.matches(&ev_node_start()));
    }

    #[test]
    fn messages_matches_only_llm_or_tool() {
        assert!(StreamMode::Messages.matches(&ev_token()));
        assert!(!StreamMode::Messages.matches(&ev_node_end()));
    }

    #[test]
    fn values_matches_only_graph_on_end() {
        assert!(StreamMode::Values.matches(&ev_on_end()));
        assert!(!StreamMode::Values.matches(&ev_node_end()));
    }

    #[test]
    fn modes_set_unions_filters() {
        let modes = StreamModes::only(StreamMode::Tasks).with(StreamMode::Updates);
        assert!(modes.matches(&ev_node_start()));
        assert!(modes.matches(&ev_node_end()));
        assert!(!modes.matches(&ev_token()));
    }
}
