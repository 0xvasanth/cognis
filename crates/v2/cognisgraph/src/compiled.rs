//! Stub — full impl in Task 7.
use crate::builder::Graph;
use crate::state::GraphState;

/// A validated, ready-to-run graph.
pub struct CompiledGraph<S: GraphState> {
    pub(crate) graph: Graph<S>,
}

impl<S: GraphState> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("node_count", &self.graph.nodes.len())
            .finish()
    }
}

impl<S: GraphState> CompiledGraph<S> {
    pub(crate) fn new(graph: Graph<S>) -> Self {
        Self { graph }
    }
}
