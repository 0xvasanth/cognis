//! Compiled, executable graph. Implements `Runnable<S, S>` so a graph
//! composes anywhere a `Runnable` is expected (including as a node
//! inside another graph).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::builder::Graph;
use crate::checkpoint::Checkpointer;
use crate::durability::Durability;
use crate::engine;
use crate::state::GraphState;
use crate::stream_mode::StreamModes;

/// A validated, ready-to-run graph. Cheap to clone (the underlying nodes
/// are `Arc<dyn Node<S>>`).
#[derive(Clone)]
pub struct CompiledGraph<S: GraphState> {
    pub(crate) graph: Graph<S>,
    pub(crate) checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    pub(crate) interrupt_before: HashSet<String>,
    pub(crate) interrupt_after: HashSet<String>,
    pub(crate) durability: Durability,
}

impl<S: GraphState> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("node_count", &self.graph.nodes.len())
            .field("has_checkpointer", &self.checkpointer.is_some())
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .finish()
    }
}

impl<S: GraphState> CompiledGraph<S> {
    pub(crate) fn new(graph: Graph<S>) -> Self {
        Self {
            graph,
            checkpointer: None,
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
            durability: Durability::default(),
        }
    }

    /// Override checkpoint timing relative to step execution. Default is
    /// [`Durability::Sync`].
    pub fn with_durability(mut self, d: Durability) -> Self {
        self.durability = d;
        self
    }

    /// Current durability mode.
    pub fn durability(&self) -> &Durability {
        &self.durability
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

impl<S: GraphState + Clone + Send + 'static> CompiledGraph<S> {
    /// Attach a checkpointer; the engine will save state after each superstep.
    pub fn with_checkpointer(mut self, cp: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    /// Pause the graph BEFORE each named node executes. Requires a checkpointer
    /// (errors at invoke time if not configured). Interrupt names are validated
    /// at invoke time, not compile time, because `with_interrupt_before` runs
    /// after `compile()`. Resume via `CompiledGraph::resume`.
    pub fn with_interrupt_before<I, N>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        self.interrupt_before
            .extend(names.into_iter().map(Into::into));
        self
    }

    /// Pause the graph AFTER each named node completes (state already updated).
    /// Requires a checkpointer. Resume via `CompiledGraph::resume`.
    pub fn with_interrupt_after<I, N>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        self.interrupt_after
            .extend(names.into_iter().map(Into::into));
        self
    }

    /// Continue execution from a previously-interrupted run.
    ///
    /// `state` is the (possibly user-edited) state to seed the next superstep
    /// with. `run_id` and `step` come from the original `GraphInterrupted` error.
    /// The resume's `RunnableConfig::run_id` is set to `run_id` so observers
    /// can correlate with the original run.
    pub async fn resume(
        &self,
        run_id: uuid::Uuid,
        step: u64,
        state: S,
        config: RunnableConfig,
    ) -> Result<S>
    where
        S::Update: Clone,
    {
        let mut cfg = config;
        cfg.run_id = run_id;
        engine::resume(self, state, cfg, step).await
    }

    // ---------------------------------------------------------------
    // State inspection (HITL / time-travel / debugging)
    // ---------------------------------------------------------------

    /// Latest checkpointed state for `run_id`. Returns `None` if there is
    /// no checkpointer attached or no state recorded for that run.
    pub async fn get_state(&self, run_id: uuid::Uuid) -> Result<Option<S>> {
        match &self.checkpointer {
            Some(cp) => cp.load(run_id, None).await,
            None => Ok(None),
        }
    }

    /// State at a specific superstep — for time-travel.
    pub async fn get_state_at(&self, run_id: uuid::Uuid, step: u64) -> Result<Option<S>> {
        match &self.checkpointer {
            Some(cp) => cp.load(run_id, Some(step)).await,
            None => Ok(None),
        }
    }

    /// Full step history for `run_id`. Each `(step, state)` pair is one
    /// superstep boundary — earliest first.
    pub async fn get_state_history(&self, run_id: uuid::Uuid) -> Result<Vec<(u64, S)>> {
        let cp = match &self.checkpointer {
            Some(cp) => cp,
            None => return Ok(Vec::new()),
        };
        let steps = cp.list(run_id).await?;
        let mut out = Vec::with_capacity(steps.len());
        for s in steps {
            if let Some(state) = cp.load(run_id, Some(s)).await? {
                out.push((s, state));
            }
        }
        Ok(out)
    }

    /// Save a (possibly user-edited) state at `step` for `run_id`. Used to
    /// patch state before resuming an interrupted run.
    ///
    /// Errors if no checkpointer is attached.
    pub async fn update_state(&self, run_id: uuid::Uuid, step: u64, state: &S) -> Result<()> {
        match &self.checkpointer {
            Some(cp) => cp.save(run_id, step, state).await,
            None => Err(cognis_core::CognisError::Configuration(
                "update_state requires a checkpointer; attach via .with_checkpointer(...)".into(),
            )),
        }
    }
}

impl<S> CompiledGraph<S>
where
    S: GraphState + Clone + Send + 'static,
    <S as GraphState>::Update: Clone,
{
    /// Stream events filtered by [`StreamModes`] — see the `stream_mode`
    /// module for what each mode captures.
    pub async fn stream_mode(
        &self,
        input: S,
        modes: StreamModes,
        config: RunnableConfig,
    ) -> Result<cognis_core::EventStream> {
        use cognis_core::Observer;
        use futures::StreamExt;
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        struct ChannelObserver(mpsc::UnboundedSender<cognis_core::Event>);
        impl Observer for ChannelObserver {
            fn on_event(&self, event: &cognis_core::Event) {
                let _ = self.0.send(event.clone());
            }
        }

        let (tx, rx) = mpsc::unbounded_channel::<cognis_core::Event>();
        let observer: Arc<dyn Observer> = Arc::new(ChannelObserver(tx));
        let mut cfg = config;
        cfg.observers.push(observer);

        let this = self.clone();
        tokio::spawn(async move {
            let _ = engine::run(&this, input, cfg).await;
        });

        let filtered = UnboundedReceiverStream::new(rx).filter(move |e| {
            let keep = modes.matches(e);
            async move { keep }
        });

        Ok(cognis_core::EventStream::new(filtered))
    }
}

#[async_trait]
impl<S> Runnable<S, S> for CompiledGraph<S>
where
    S: GraphState + Clone + Send + 'static,
    <S as GraphState>::Update: Clone,
{
    async fn invoke(&self, input: S, config: RunnableConfig) -> Result<S> {
        engine::run(self, input, config).await
    }

    fn name(&self) -> &str {
        "CompiledGraph"
    }

    /// Override the default `stream_events` to emit real per-node events as
    /// the engine runs (real-time, not synthetic OnEnd-only). Engine events
    /// embed `serde_json::Value::Null` so `S: Serialize` is not required.
    async fn stream_events(
        &self,
        input: S,
        config: RunnableConfig,
    ) -> Result<cognis_core::EventStream> {
        use cognis_core::Observer;
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        struct ChannelObserver(mpsc::UnboundedSender<cognis_core::Event>);
        impl Observer for ChannelObserver {
            fn on_event(&self, event: &cognis_core::Event) {
                let _ = self.0.send(event.clone());
            }
        }

        let (tx, rx) = mpsc::unbounded_channel::<cognis_core::Event>();
        let observer: Arc<dyn Observer> = Arc::new(ChannelObserver(tx));
        let mut cfg = config;
        cfg.observers.push(observer);

        let this = self.clone();
        tokio::spawn(async move {
            let _ = engine::run(&this, input, cfg).await;
        });

        Ok(cognis_core::EventStream::new(
            UnboundedReceiverStream::new(rx),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::node::{node_fn, NodeOut};

    #[derive(Default, Clone, Debug, PartialEq, serde::Serialize)]
    struct Counter {
        n: u32,
    }

    #[derive(Default, Clone)]
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
            cognis_core::CognisError::RecursionLimit { limit: 3 }
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
        let r1 = g
            .invoke(Counter::default(), RunnableConfig::default())
            .await
            .unwrap();
        let r2 = g2
            .invoke(Counter::default(), RunnableConfig::default())
            .await
            .unwrap();
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

    #[tokio::test]
    async fn stream_events_emits_per_node() {
        use cognis_core::Event;
        use futures::StreamExt;

        let g = Graph::<Counter>::new()
            .node(
                "a",
                node_fn::<Counter, _, _>("a", |_, _| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 1 },
                        goto: Goto::node("b"),
                    })
                }),
            )
            .node(
                "b",
                node_fn::<Counter, _, _>("b", |_, _| async move {
                    Ok(NodeOut {
                        update: CounterUpdate { n: 1 },
                        goto: Goto::end(),
                    })
                }),
            )
            .start_at("a")
            .compile()
            .unwrap();

        let mut s = g
            .stream_events(Counter::default(), RunnableConfig::default())
            .await
            .unwrap();
        let mut events = Vec::new();
        while let Some(e) = s.next().await {
            events.push(e);
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::OnNodeStart { node, .. } if node == "a")));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::OnNodeStart { node, .. } if node == "b")));
        assert!(events.iter().any(|e| matches!(e, Event::OnEnd { .. })));
    }
}
