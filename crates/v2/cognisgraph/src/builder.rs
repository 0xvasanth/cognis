//! Graph<S> builder — fluent API for constructing a graph.

use std::collections::HashMap;
use std::sync::Arc;

use cognis2_core::{CognisError, Result};

use crate::node::Node;
use crate::state::GraphState;

/// A graph under construction. Generic over the state type `S`. Convert to
/// an executable [`CompiledGraph`] via `.compile()`.
pub struct Graph<S: GraphState> {
    pub(crate) nodes: HashMap<String, Arc<dyn Node<S>>>,
    pub(crate) edges: HashMap<String, String>,
    pub(crate) start: Option<String>,
}

impl<S: GraphState> Default for Graph<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GraphState> Graph<S> {
    /// Empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            start: None,
        }
    }

    /// Add a node. The `name` is the routing key — `Goto::node("...")`
    /// must reference an existing node by this exact name.
    pub fn node(mut self, name: impl Into<String>, node: impl Node<S> + 'static) -> Self {
        self.nodes.insert(name.into(), Arc::new(node));
        self
    }

    /// Add an unconditional edge. The engine uses this only when the source
    /// node returns a Goto that doesn't specify a target (effectively, when
    /// the node returns Goto::Node("") or relies on a static edge). For
    /// most graphs, prefer returning Goto::Node("...") from inside the node.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.insert(from.into(), to.into());
        self
    }

    /// Set the start node.
    pub fn start_at(mut self, name: impl Into<String>) -> Self {
        self.start = Some(name.into());
        self
    }

    /// Validate and freeze into an executable [`CompiledGraph`].
    pub fn compile(self) -> Result<crate::compiled::CompiledGraph<S>> {
        crate::validate::validate(&self)?;
        Ok(crate::compiled::CompiledGraph::new(self))
    }
}

/// Builder for the special-case linear graph: each stage feeds the next,
/// no branching, no loops. Sugar over the full Graph<()> builder.
///
/// Linear graphs use `()` as the state type — there's nothing to merge,
/// nodes simply pass through. For typed-state pipelines, use the full
/// builder with a custom state struct.
pub struct LinearBuilder {
    stages: Vec<(String, Arc<dyn Node<()>>)>,
}

impl Default for LinearBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearBuilder {
    /// Empty linear builder.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage. Stages auto-name themselves `"0"`, `"1"`, ….
    pub fn then(mut self, node: impl Node<()> + 'static) -> Self {
        let idx = self.stages.len().to_string();
        self.stages.push((idx, Arc::new(node)));
        self
    }

    /// Compile to a `CompiledGraph<()>`.
    pub fn compile(self) -> Result<crate::compiled::CompiledGraph<()>> {
        if self.stages.is_empty() {
            return Err(CognisError::Configuration(
                "Graph::linear() requires at least one stage".into(),
            ));
        }
        let mut g = Graph::<()>::new();
        let last_idx = self.stages.len() - 1;
        for (i, (name, node)) in self.stages.into_iter().enumerate() {
            g.nodes.insert(name.clone(), node);
            if i < last_idx {
                let next = (i + 1).to_string();
                g.edges.insert(name, next);
            }
        }
        g.start = Some("0".to_string());
        g.compile()
    }
}

impl<S: GraphState> Graph<S> {
    /// Sugar for the linear-pipeline case. `Graph::linear()` returns a
    /// `LinearBuilder`. The result is a `CompiledGraph<()>` — same engine,
    /// same `Runnable<(), ()>` contract.
    pub fn linear() -> LinearBuilder {
        LinearBuilder::new()
    }
}

// `()` needs to implement GraphState so LinearBuilder compiles. The unit
// state has no fields, so apply is a no-op.
impl GraphState for () {
    type Update = ();
    fn apply(&mut self, _: Self::Update) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone, Debug)]
    struct S {
        msg: String,
    }
    #[derive(Default)]
    struct SU {
        msg: String,
    }
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, u: Self::Update) {
            self.msg.push_str(&u.msg);
        }
    }

    #[test]
    fn build_with_nodes_and_start() {
        let g = Graph::<S>::new()
            .node(
                "a",
                node_fn::<S, _, _>("a", |_s, _c| async move {
                    Ok(NodeOut {
                        update: SU { msg: "a".into() },
                        goto: Goto::node("b"),
                    })
                }),
            )
            .node(
                "b",
                node_fn::<S, _, _>("b", |_s, _c| async move {
                    Ok(NodeOut::end_with(SU { msg: "b".into() }))
                }),
            )
            .start_at("a");
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.start.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn linear_builder_chains_three_stages() {
        let n = node_fn::<(), _, _>("noop", |_s, _c| async move {
            Ok(NodeOut::goto_only(Goto::end()))
        });
        let n2 = node_fn::<(), _, _>("noop", |_s, _c| async move {
            Ok(NodeOut::goto_only(Goto::end()))
        });
        let n3 = node_fn::<(), _, _>("noop", |_s, _c| async move {
            Ok(NodeOut::goto_only(Goto::end()))
        });
        let cg = Graph::<()>::linear().then(n).then(n2).then(n3).compile();
        assert!(cg.is_ok());
    }

    #[test]
    fn linear_builder_rejects_empty() {
        let cg = LinearBuilder::new().compile();
        assert!(cg.is_err());
    }
}
