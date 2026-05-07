//! Compiled-graph snapshot — the *shape* of a graph (not its node closures).
//!
//! Use cases: schema-evolution checks, cross-version comparison, generated
//! docs, "spec" files in CI. `from_snapshot` rebuilds via a caller-supplied
//! `node_factory` because the closures themselves aren't serializable.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};

use crate::builder::Graph;
use crate::compiled::CompiledGraph;
use crate::node::Node;
use crate::state::GraphState;

/// Serializable shape of a `CompiledGraph<S>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Registered node names (alphabetical).
    pub nodes: Vec<String>,
    /// Static `(from, to)` edges declared via `Graph::edge`.
    pub edges: Vec<(String, String)>,
    /// Start node, if set.
    pub start: Option<String>,
    /// Optional version tag (see [`crate::Graph::with_version`]).
    pub version: Option<String>,
}

impl<S: GraphState> CompiledGraph<S> {
    /// Snapshot the graph's static shape.
    pub fn snapshot(&self) -> GraphSnapshot {
        let mut nodes: Vec<String> = self.graph.nodes.keys().cloned().collect();
        nodes.sort();
        let mut edges: Vec<(String, String)> = self
            .graph
            .edges
            .iter()
            .map(|(f, t)| (f.clone(), t.clone()))
            .collect();
        edges.sort();
        GraphSnapshot {
            nodes,
            edges,
            start: self.graph.start.clone(),
            version: self.graph.version.clone(),
        }
    }
}

/// Factory function used by [`Graph::from_snapshot`] — looks up a node
/// implementation by name.
pub type NodeFactory<S> = dyn Fn(&str) -> Option<Arc<dyn Node<S>>>;

impl<S: GraphState> Graph<S> {
    /// Reconstruct a `Graph<S>` from a [`GraphSnapshot`] using a factory
    /// to produce node implementations by name. The factory must return
    /// a node for every name in `snap.nodes`; otherwise this errors.
    pub fn from_snapshot(snap: &GraphSnapshot, node_factory: &NodeFactory<S>) -> Result<Self> {
        let mut g = Graph::<S>::new();
        for name in &snap.nodes {
            let n = node_factory(name).ok_or_else(|| {
                CognisError::Configuration(format!(
                    "from_snapshot: factory returned no node for `{name}`"
                ))
            })?;
            g.nodes.insert(name.clone(), n);
        }
        let mut edges: HashMap<String, String> = HashMap::new();
        for (f, t) in &snap.edges {
            edges.insert(f.clone(), t.clone());
        }
        g.edges = edges;
        g.start = snap.start.clone();
        g.version = snap.version.clone();
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone, Debug, PartialEq)]
    struct S {
        n: u32,
    }
    #[derive(Default)]
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
    fn snapshot_roundtrip_via_factory() {
        let g = Graph::<S>::new()
            .node(
                "a",
                node_fn::<S, _, _>("a", |_, _| async {
                    Ok(NodeOut {
                        update: SU { n: 1 },
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("a")
            .compile()
            .unwrap();
        let snap = g.snapshot();
        assert_eq!(snap.nodes, vec!["a"]);
        assert_eq!(snap.start.as_deref(), Some("a"));

        let factory = |name: &str| -> Option<Arc<dyn Node<S>>> {
            if name == "a" {
                Some(Arc::new(node_fn::<S, _, _>("a", |_, _| async {
                    Ok(NodeOut {
                        update: SU { n: 1 },
                        goto: Goto::end(),
                    })
                })))
            } else {
                None
            }
        };
        let g2 = Graph::<S>::from_snapshot(&snap, &factory).unwrap();
        let snap2 = g2.compile().unwrap().snapshot();
        assert_eq!(snap2.nodes, snap.nodes);
        assert_eq!(snap2.start, snap.start);
    }
}
