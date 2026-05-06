//! Stub — full impl in Task 7.
use crate::builder::Graph;
use crate::state::GraphState;

#[allow(dead_code)]
/// A validated, ready-to-run graph.
pub struct CompiledGraph<S: GraphState> {
    pub(crate) graph: Graph<S>,
}

impl<S: GraphState> CompiledGraph<S> {
    pub(crate) fn new(graph: Graph<S>) -> Self {
        Self { graph }
    }
}
