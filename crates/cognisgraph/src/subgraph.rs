//! Run a child graph as a node inside a parent graph, with state
//! mapping in both directions.
//!
//! Already partly possible (`CompiledGraph<S>: Runnable<S, S>`), but the
//! parent and child usually have *different* state types. [`subgraph`]
//! adapts: parent state → child state → run → child state → parent
//! update.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Result, Runnable};

use crate::compiled::CompiledGraph;
use crate::goto::Goto;
use crate::node::{Node, NodeCtx, NodeOut};
use crate::state::GraphState;

/// Adapter `Node` that runs a child graph and projects its state into a
/// parent update.
#[allow(clippy::type_complexity)]
pub struct Subgraph<P, C>
where
    P: GraphState,
    C: GraphState + Clone,
    <C as GraphState>::Update: Clone,
{
    name: String,
    child: CompiledGraph<C>,
    project_in: Arc<dyn Fn(&P) -> C + Send + Sync>,
    project_out: Arc<dyn Fn(&C) -> P::Update + Send + Sync>,
    next: Goto,
}

impl<P, C> Subgraph<P, C>
where
    P: GraphState,
    C: GraphState + Clone + Send + 'static,
    <C as GraphState>::Update: Clone,
{
    /// Build a subgraph adapter.
    ///
    /// - `project_in`: parent state → fresh child state seed.
    /// - `project_out`: final child state → parent update.
    /// - `next`: where the parent goes after the subgraph returns.
    pub fn new(
        name: impl Into<String>,
        child: CompiledGraph<C>,
        project_in: impl Fn(&P) -> C + Send + Sync + 'static,
        project_out: impl Fn(&C) -> P::Update + Send + Sync + 'static,
        next: Goto,
    ) -> Self {
        Self {
            name: name.into(),
            child,
            project_in: Arc::new(project_in),
            project_out: Arc::new(project_out),
            next,
        }
    }
}

#[async_trait]
impl<P, C> Node<P> for Subgraph<P, C>
where
    P: GraphState + Send + Sync + 'static,
    C: GraphState + Clone + Send + Sync + 'static,
    <C as GraphState>::Update: Clone,
    <P as GraphState>::Update: Send + 'static,
{
    async fn execute(&self, parent: &P, ctx: &NodeCtx<'_>) -> Result<NodeOut<P>> {
        let seed = (self.project_in)(parent);
        let mut cfg = ctx.config.clone();
        // Fresh run_id for the child run so observers can correlate.
        cfg.run_id = uuid::Uuid::new_v4();
        let final_child = self.child.invoke(seed, cfg).await?;
        let update = (self.project_out)(&final_child);
        Ok(NodeOut {
            update,
            goto: self.next.clone(),
        })
    }
    fn name(&self) -> &str {
        &self.name
    }
}
