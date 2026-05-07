//! Named barrier — wait for N parties before dispatching a target node.
//!
//! Use case: nodes `a`, `b`, `c` all need to complete before node `d`
//! runs once with all three outputs in hand. V2's existing
//! [`Goto::Multiple`] runs N nodes in parallel + atomically merges, but
//! doesn't give downstream nodes a single "all parties present" handoff.
//!
//! [`BarrierNode`] is a node implementation users add to their graph
//! under a name (e.g. `"join"`). Parties dispatch to it via
//! [`Goto::Send`] with their payload; the barrier accumulates payloads
//! until `expected` parties have arrived, then forwards everything to
//! the target node in the next superstep.
//!
//! Pending parties get [`Goto::Halt`] — their branch stops without
//! terminating the graph.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use cognis2_core::Result;
use uuid::Uuid;

use crate::goto::Goto;
use crate::node::{Node, NodeCtx, NodeOut};
use crate::state::GraphState;

/// Per-run accumulator: maps `run_id → ordered Vec<payload>`.
type Acc = Arc<Mutex<HashMap<Uuid, Vec<serde_json::Value>>>>;

/// Wait for N writes before dispatching `target`.
///
/// Pending parties return [`Goto::Halt`]; the last party's invocation
/// dispatches `target` via [`Goto::Send`] with the full payload list as
/// a single JSON array.
pub struct BarrierNode<S: GraphState> {
    name: String,
    expected: usize,
    target: String,
    acc: Acc,
    _phantom: std::marker::PhantomData<fn() -> S>,
}

impl<S: GraphState> BarrierNode<S> {
    /// Build a barrier that waits for `expected` parties before
    /// dispatching `target`.
    pub fn new(name: impl Into<String>, expected: usize, target: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected,
            target: target.into(),
            acc: Arc::new(Mutex::new(HashMap::new())),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S: GraphState> Node<S> for BarrierNode<S>
where
    S::Update: Default,
{
    async fn execute(&self, _state: &S, ctx: &NodeCtx<'_>) -> Result<NodeOut<S>> {
        let payload = ctx.payload().cloned().unwrap_or(serde_json::Value::Null);
        let (count, payloads) = {
            let mut acc = self
                .acc
                .lock()
                .map_err(|e| cognis2_core::CognisError::Internal(format!("barrier mutex: {e}")))?;
            let entry = acc.entry(ctx.run_id).or_default();
            entry.push(payload);
            (entry.len(), entry.clone())
        };

        if count < self.expected {
            // Still waiting on more parties. Halt this branch.
            return Ok(NodeOut {
                update: S::Update::default(),
                goto: Goto::Halt,
            });
        }

        // All parties present — dispatch target with the full payload list,
        // then clear our per-run accumulator so the barrier is reusable.
        {
            let mut acc = self
                .acc
                .lock()
                .map_err(|e| cognis2_core::CognisError::Internal(format!("barrier mutex: {e}")))?;
            acc.remove(&ctx.run_id);
        }
        Ok(NodeOut {
            update: S::Update::default(),
            goto: Goto::Send(vec![(self.target.clone(), serde_json::json!(payloads))]),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use cognis2_core::RunnableConfig;

    #[derive(Default, Clone)]
    struct S;
    #[derive(Default, Clone)]
    struct SU;
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, _: SU) {}
    }

    #[tokio::test]
    async fn pending_parties_halt() {
        let b: BarrierNode<S> = BarrierNode::new("join", 3, "next");
        let cfg = RunnableConfig::default();
        let payload_a = serde_json::json!("a");
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg).with_payload(&payload_a);
        let out = b.execute(&S, &ctx).await.unwrap();
        assert!(matches!(out.goto, Goto::Halt));
    }

    #[tokio::test]
    async fn last_party_dispatches_target_with_all_payloads() {
        let b: BarrierNode<S> = BarrierNode::new("join", 2, "next");
        let cfg = RunnableConfig::default();
        let p1 = serde_json::json!("first");
        let ctx1 = NodeCtx::new(Uuid::nil(), 0, &cfg).with_payload(&p1);
        let _ = b.execute(&S, &ctx1).await.unwrap();
        let p2 = serde_json::json!("second");
        let ctx2 = NodeCtx::new(Uuid::nil(), 0, &cfg).with_payload(&p2);
        let out = b.execute(&S, &ctx2).await.unwrap();
        match out.goto {
            Goto::Send(targets) => {
                assert_eq!(targets.len(), 1);
                assert_eq!(targets[0].0, "next");
                let arr = targets[0].1.as_array().unwrap();
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0], "first");
                assert_eq!(arr[1], "second");
            }
            _ => panic!("expected Goto::Send"),
        }
    }

    #[tokio::test]
    async fn barrier_resets_per_run() {
        // Same barrier used twice (different run_ids) — second run should
        // start fresh, not accumulate from the first.
        let b: BarrierNode<S> = BarrierNode::new("join", 1, "next");
        let cfg = RunnableConfig::default();
        let p = serde_json::json!("only");
        let run_a = Uuid::new_v4();
        let ctx_a = NodeCtx::new(run_a, 0, &cfg).with_payload(&p);
        let _ = b.execute(&S, &ctx_a).await.unwrap();
        // Same barrier, different run.
        let run_b = Uuid::new_v4();
        let ctx_b = NodeCtx::new(run_b, 0, &cfg).with_payload(&p);
        let out = b.execute(&S, &ctx_b).await.unwrap();
        match out.goto {
            Goto::Send(t) => assert_eq!(t[0].1.as_array().unwrap().len(), 1),
            _ => panic!(),
        }
    }
}
