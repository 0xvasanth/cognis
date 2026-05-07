//! `Command<S>` — fluent node return type bundling `update + goto + payload`.
//!
//! Equivalent in expressive power to building a [`crate::node::NodeOut`]
//! directly, but reads more naturally inside node bodies:
//!
//! ```ignore
//! Command::new(MyUpdate { count: 1 })
//!     .goto("next")
//!     .with_payload(serde_json::json!({"hint": "..."}))
//!     .into()
//! ```
//!
//! `Command<S>: Into<NodeOut<S>>`, so existing `node_fn` closures returning
//! a `NodeOut<S>` work unchanged. `.payload(...)` is only honoured when the
//! goto fans out via `Goto::Send`; otherwise it's dropped (with a debug log).

use crate::goto::Goto;
use crate::node::NodeOut;
use crate::state::GraphState;

/// Fluent builder for what a node returns: state delta + routing + optional payload.
#[derive(Debug, Clone)]
pub struct Command<S: GraphState> {
    update: S::Update,
    goto: Goto,
    payload: Option<serde_json::Value>,
}

impl<S: GraphState> Command<S> {
    /// Build a `Command` with a state delta. Default routing is `Goto::End`.
    pub fn new(update: S::Update) -> Self {
        Self {
            update,
            goto: Goto::End,
            payload: None,
        }
    }

    /// Build with default state delta and a custom goto.
    pub fn goto_only(goto: Goto) -> Self
    where
        S::Update: Default,
    {
        Self {
            update: S::Update::default(),
            goto,
            payload: None,
        }
    }

    /// Route to a single named node.
    pub fn goto(mut self, name: impl Into<String>) -> Self {
        self.goto = Goto::Node(name.into());
        self
    }

    /// Fan out to multiple named nodes.
    pub fn goto_multiple<I, N>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        self.goto = Goto::Multiple(names.into_iter().map(Into::into).collect());
        self
    }

    /// Dispatch with per-target payloads (`Goto::Send`).
    pub fn send<I, N>(mut self, targets: I) -> Self
    where
        I: IntoIterator<Item = (N, serde_json::Value)>,
        N: Into<String>,
    {
        self.goto = Goto::Send(targets.into_iter().map(|(n, p)| (n.into(), p)).collect());
        self
    }

    /// Terminate the graph.
    pub fn end(mut self) -> Self {
        self.goto = Goto::End;
        self
    }

    /// Attach a payload. If the goto is a single-target `Goto::Node`, this
    /// promotes it to `Goto::Send` so the receiving node can read the payload.
    /// On `Goto::End` the payload is dropped.
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        match std::mem::replace(&mut self.goto, Goto::End) {
            Goto::Node(name) => {
                self.goto = Goto::Send(vec![(name, payload)]);
            }
            other @ Goto::End => {
                self.goto = other;
            }
            other => {
                // Multiple/Send: keep the existing routing; stash payload as a hint
                // for nodes that read NodeCtx::payload (only valid via Send).
                self.goto = other;
                self.payload = Some(payload);
            }
        }
        self
    }
}

impl<S: GraphState> From<Command<S>> for NodeOut<S> {
    fn from(c: Command<S>) -> Self {
        // The unused `payload` on Multiple/Send is intentional — Send already
        // carries payloads per target. We log so users who set both notice.
        if c.payload.is_some() && !matches!(c.goto, Goto::End) {
            tracing::debug!(
                "Command::with_payload set on a non-Node goto; payload ignored \
                 (use Command::send for per-target payloads)"
            );
        }
        NodeOut {
            update: c.update,
            goto: c.goto,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, Clone, PartialEq)]
    struct S {
        n: u32,
    }
    #[derive(Debug, Default, Clone)]
    struct SU {
        n: u32,
    }
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, u: Self::Update) {
            self.n += u.n;
        }
    }

    #[test]
    fn defaults_to_end() {
        let cmd: Command<S> = Command::new(SU { n: 1 });
        let out: NodeOut<S> = cmd.into();
        assert!(matches!(out.goto, Goto::End));
        assert_eq!(out.update.n, 1);
    }

    #[test]
    fn goto_routes_single() {
        let cmd: Command<S> = Command::new(SU { n: 0 }).goto("next");
        let out: NodeOut<S> = cmd.into();
        assert_eq!(out.goto, Goto::Node("next".into()));
    }

    #[test]
    fn with_payload_promotes_node_to_send() {
        let cmd: Command<S> = Command::new(SU { n: 0 })
            .goto("worker")
            .with_payload(serde_json::json!({"x": 1}));
        let out: NodeOut<S> = cmd.into();
        if let Goto::Send(t) = out.goto {
            assert_eq!(t.len(), 1);
            assert_eq!(t[0].0, "worker");
            assert_eq!(t[0].1["x"], 1);
        } else {
            panic!("expected Send");
        }
    }

    #[test]
    fn send_with_multiple_targets() {
        let cmd: Command<S> = Command::new(SU { n: 0 }).send([
            ("a", serde_json::json!({"i": 0})),
            ("b", serde_json::json!({"i": 1})),
        ]);
        let out: NodeOut<S> = cmd.into();
        if let Goto::Send(t) = out.goto {
            assert_eq!(t.len(), 2);
        } else {
            panic!("expected Send");
        }
    }

    #[test]
    fn goto_only_skips_update() {
        let cmd: Command<S> = Command::goto_only(Goto::node("x"));
        let out: NodeOut<S> = cmd.into();
        assert_eq!(out.goto, Goto::Node("x".into()));
        assert_eq!(out.update.n, 0);
    }
}
