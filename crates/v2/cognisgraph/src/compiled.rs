//! Compiled, executable graph. Implements `Runnable<S, S>` so a graph
//! composes anywhere a `Runnable` is expected (including as a node
//! inside another graph).

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Result, Runnable, RunnableConfig};

use crate::builder::Graph;
use crate::checkpoint::Checkpointer;
use crate::engine;
use crate::state::GraphState;

/// A validated, ready-to-run graph. Cheap to clone (the underlying nodes
/// are `Arc<dyn Node<S>>`).
#[derive(Clone)]
pub struct CompiledGraph<S: GraphState> {
    pub(crate) graph: Graph<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer<S>>>,
}

impl<S: GraphState> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("node_count", &self.graph.nodes.len())
            .field("has_checkpointer", &self.checkpointer.is_some())
            .finish()
    }
}

impl<S: GraphState> CompiledGraph<S> {
    pub(crate) fn new(graph: Graph<S>) -> Self {
        Self {
            graph,
            checkpointer: None,
        }
    }

    /// Number of registered nodes — useful for testing / introspection.
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Names of all registered nodes.
    pub fn node_names(&self) -> Vec<&str> {
        self.graph.nodes.keys().map(|s| s.as_str()).collect()
    }
}

impl<S: GraphState + Clone> CompiledGraph<S> {
    /// Attach a checkpointer; the engine will save state after each superstep.
    pub fn with_checkpointer(mut self, cp: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(cp);
        self
    }
}

#[async_trait]
impl<S> Runnable<S, S> for CompiledGraph<S>
where
    S: GraphState + Clone,
{
    async fn invoke(&self, input: S, config: RunnableConfig) -> Result<S> {
        engine::run(&self.graph, input, config, self.checkpointer.as_ref()).await
    }

    fn name(&self) -> &str {
        "CompiledGraph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone, Debug, PartialEq)]
    struct Counter {
        n: u32,
    }

    #[derive(Default)]
    struct CounterUpdate {
        n: u32,
    }

    impl GraphState for Counter {
        type Update = CounterUpdate;
        fn apply(&mut self, u: Self::Update) {
            self.n += u.n;
        }
    }

    #[tokio::test]
    async fn linear_two_nodes_runs_to_end() {
        let g = Graph::<Counter>::new()
            .node(
                "a",
                node_fn::<Counter, _, _>("a", |_s, _c| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 1 },
                        goto: Goto::node("b"),
                    })
                }),
            )
            .node(
                "b",
                node_fn::<Counter, _, _>("b", |_s, _c| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 10 },
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("a")
            .compile()
            .unwrap();

        let out = g
            .invoke(Counter::default(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, Counter { n: 11 });
    }

    #[tokio::test]
    async fn cycle_terminates_via_state_check() {
        // Loop until counter reaches 5.
        let g = Graph::<Counter>::new()
            .node(
                "tick",
                node_fn::<Counter, _, _>("tick", |s, _c| {
                    let cur = s.n;
                    async move {
                        if cur >= 5 {
                            Ok(NodeOut {
                                update: CounterUpdate { n: 0 },
                                goto: Goto::end(),
                            })
                        } else {
                            Ok(NodeOut {
                                update: CounterUpdate { n: 1 },
                                goto: Goto::node("tick"),
                            })
                        }
                    }
                }),
            )
            .start_at("tick")
            .compile()
            .unwrap();

        let out = g
            .invoke(Counter::default(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, Counter { n: 5 });
    }

    #[tokio::test]
    async fn recursion_limit_is_honored() {
        // Infinite loop → expect RecursionLimit error.
        let g = Graph::<Counter>::new()
            .node(
                "loop",
                node_fn::<Counter, _, _>("loop", |_s, _c| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 1 },
                        goto: Goto::node("loop"),
                    })
                }),
            )
            .start_at("loop")
            .compile()
            .unwrap();

        let cfg = RunnableConfig::default().with_recursion_limit(3);
        let err = g.invoke(Counter::default(), cfg).await.unwrap_err();
        assert!(matches!(
            err,
            cognis2_core::CognisError::RecursionLimit { limit: 3 }
        ));
    }

    #[tokio::test]
    async fn compiled_graph_clones_and_runs() {
        let g = Graph::<Counter>::new()
            .node(
                "a",
                node_fn::<Counter, _, _>("a", |_s, _c| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 1 },
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("a")
            .compile()
            .unwrap();
        let g2 = g.clone();
        let r1 = g.invoke(Counter::default(), RunnableConfig::default()).await.unwrap();
        let r2 = g2.invoke(Counter::default(), RunnableConfig::default()).await.unwrap();
        assert_eq!(r1.n, 1);
        assert_eq!(r2.n, 1);
    }

    #[tokio::test]
    async fn route_to_unknown_node_errors() {
        let g = Graph::<Counter>::new()
            .node(
                "bad",
                node_fn::<Counter, _, _>("bad", |_s, _c| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 0 },
                        goto: Goto::node("ghost"),
                    })
                }),
            )
            .start_at("bad")
            .compile()
            .unwrap();
        let err = g
            .invoke(Counter::default(), RunnableConfig::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("ghost"));
    }
}
