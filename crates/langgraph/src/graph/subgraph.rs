//! Subgraph composition support.
//!
//! [`SubgraphNode`] wraps a [`CompiledStateGraph`] so it can be used as a node
//! inside another [`StateGraph`]. This enables hierarchical graph composition
//! where an entire compiled graph executes as a single step in a parent graph.

use std::sync::Arc;

use super::state::{AsyncNodeAction, CompiledStateGraph};

/// A wrapper that allows a [`CompiledStateGraph`] to be used as a node action
/// inside another [`StateGraph`].
///
/// When invoked, the `SubgraphNode` runs the inner graph's
/// [`invoke`](CompiledStateGraph::invoke) method with the parent's current
/// state and returns the subgraph's final output state, which is then merged
/// back into the parent graph's state.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
/// use langgraph::graph::subgraph::SubgraphNode;
/// use langgraph::errors::LangGraphError;
///
/// let action: AsyncNodeAction = Arc::new(|_s: Value| {
///     Box::pin(async { Ok(json!({"inner_done": true})) })
/// });
///
/// let inner = StateGraph::new()
///     .add_node("inner_a", action)
///     .set_entry_point("inner_a")
///     .set_finish_point("inner_a")
///     .compile()
///     .unwrap();
///
/// let outer = StateGraph::new()
///     .add_subgraph("my_subgraph", inner)
///     .set_entry_point("my_subgraph")
///     .set_finish_point("my_subgraph")
///     .compile()
///     .unwrap();
/// ```
pub struct SubgraphNode {
    /// The compiled subgraph to execute.
    graph: Arc<CompiledStateGraph>,
}

impl SubgraphNode {
    /// Create a new `SubgraphNode` wrapping the given compiled graph.
    pub fn new(graph: CompiledStateGraph) -> Self {
        Self {
            graph: Arc::new(graph),
        }
    }

    /// Convert this `SubgraphNode` into an [`AsyncNodeAction`] that can be
    /// added to a [`StateGraph`].
    pub fn into_action(self) -> AsyncNodeAction {
        let graph = self.graph;
        Arc::new(move |state| {
            let graph = graph.clone();
            Box::pin(async move { graph.invoke(state).await })
        })
    }
}
