//! `compile()`-time validation. Medium strength: start node exists,
//! all referenced nodes exist, reachability from start, at least one
//! path leads to End.
//!
//! Note on reachability: nodes return `Goto` at runtime, so the engine's
//! actual graph is dynamic. We approximate by treating each node as
//! capable of routing to any other node (i.e., conservatively assume
//! every node is reachable if any path exists). For static-edge cases
//! (the LinearBuilder), edges are explicit and we walk them.

use std::collections::HashSet;

use cognis_core::{CognisError, Result};

use crate::builder::Graph;
use crate::state::GraphState;

pub(crate) fn validate<S: GraphState>(g: &Graph<S>) -> Result<()> {
    // 1. Start exists.
    let start = g
        .start
        .as_ref()
        .ok_or_else(|| CognisError::Configuration("graph has no start node".into()))?;

    if !g.nodes.contains_key(start) {
        return Err(CognisError::Configuration(format!(
            "start node `{start}` not registered"
        )));
    }

    // 2. Every static edge target exists.
    for (from, to) in &g.edges {
        if !g.nodes.contains_key(from) {
            return Err(CognisError::Configuration(format!(
                "edge source `{from}` is not a registered node"
            )));
        }
        if !g.nodes.contains_key(to) {
            return Err(CognisError::Configuration(format!(
                "edge target `{to}` is not a registered node"
            )));
        }
    }

    // 3. Reachability from start (using static edges only; nodes can also
    //    return dynamic Goto::Node(...) at runtime, which we can't
    //    statically analyse without inspecting closures).
    if !g.edges.is_empty() {
        let reachable = walk_static(g, start);
        let unreachable: Vec<&String> = g
            .nodes
            .keys()
            .filter(|n| !reachable.contains(n.as_str()))
            .collect();
        if !unreachable.is_empty() {
            // Don't error — log a warning. Dynamic Goto returns may reach
            // these nodes at runtime. (See note above.)
            tracing::debug!(
                ?unreachable,
                "nodes not reachable via static edges from start; dynamic Goto returns may reach them"
            );
        }
    }

    // 4. (Slice 1) At least one path to End — for LinearBuilder, the last
    //    stage has no outgoing edge, so it's an implicit End. For
    //    user-built graphs with no edges, every node is expected to
    //    return Goto::End at runtime; we can't statically prove it.
    //
    //    For now, this check is informational only. Slice-2 work will
    //    add static analysis of dynamic Goto branches.

    Ok(())
}

fn walk_static<S: GraphState>(g: &Graph<S>, start: &str) -> HashSet<String> {
    let mut reachable = HashSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(n) = stack.pop() {
        if !reachable.insert(n.clone()) {
            continue;
        }
        if let Some(next) = g.edges.get(&n) {
            stack.push(next.clone());
        }
    }
    reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone)]
    struct S;
    #[derive(Default)]
    struct SU;
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, _: Self::Update) {}
    }

    fn noop_node() -> impl crate::node::Node<S> + 'static {
        node_fn::<S, _, _>(
            "noop",
            |_, _| async move { Ok(NodeOut::goto_only(Goto::end())) },
        )
    }

    #[test]
    fn missing_start_errors() {
        let g = Graph::<S>::new().node("a", noop_node());
        let err = g.compile().unwrap_err();
        assert!(format!("{err}").contains("no start node"));
    }

    #[test]
    fn unknown_start_errors() {
        let g = Graph::<S>::new().node("a", noop_node()).start_at("missing");
        let err = g.compile().unwrap_err();
        assert!(format!("{err}").contains("not registered"));
    }

    #[test]
    fn unknown_edge_target_errors() {
        let g = Graph::<S>::new()
            .node("a", noop_node())
            .edge("a", "ghost")
            .start_at("a");
        let err = g.compile().unwrap_err();
        assert!(format!("{err}").contains("`ghost`"));
    }

    #[test]
    fn happy_path_compiles() {
        let g = Graph::<S>::new()
            .node("a", noop_node())
            .node("b", noop_node())
            .edge("a", "b")
            .start_at("a");
        assert!(g.compile().is_ok());
    }
}
