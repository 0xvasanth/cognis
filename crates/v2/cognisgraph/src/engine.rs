//! Pregel-style superstep executor. Slice 1 runs single-node-active.
//! Multi-node fan-out (Goto::Multiple, Send) is slice 2.

use std::sync::Arc;

use cognis2_core::{CognisError, Event, Result, RunnableConfig};

use crate::builder::Graph;
use crate::checkpoint::Checkpointer;
use crate::goto::Goto;
use crate::node::{Node, NodeCtx};
use crate::state::GraphState;

/// Run the graph from `initial_state` to termination.
pub(crate) async fn run<S>(
    graph: &Graph<S>,
    initial_state: S,
    config: RunnableConfig,
    checkpointer: Option<&Arc<dyn Checkpointer<S>>>,
) -> Result<S>
where
    S: GraphState + Clone,
{
    let start_name = graph
        .start
        .as_ref()
        .ok_or_else(|| CognisError::Configuration("graph has no start node".into()))?;
    let mut active: Arc<dyn Node<S>> = graph
        .nodes
        .get(start_name)
        .ok_or_else(|| {
            CognisError::Configuration(format!("start node `{start_name}` missing at runtime"))
        })?
        .clone();
    let mut active_name = start_name.clone();

    let mut state = initial_state;
    let recursion_limit = config.recursion_limit;
    let run_id = config.run_id;

    config.emit(&Event::OnStart {
        runnable: format!("graph[{start_name}]"),
        run_id,
        input: serde_json::Value::Null,
    });

    for step in 0..(recursion_limit as u64) {
        if config.is_cancelled() {
            return Err(CognisError::Cancelled);
        }
        if let Some(deadline) = config.deadline {
            if std::time::Instant::now() > deadline {
                return Err(CognisError::Timeout {
                    operation: format!("graph[{active_name}]"),
                    timeout_ms: 0,
                });
            }
        }

        let ctx = NodeCtx::new(run_id, step, &config);

        ctx.emit(&Event::OnNodeStart {
            node: active_name.clone(),
            step,
            run_id,
        });

        let out = match active.execute(&state, &ctx).await {
            Ok(o) => o,
            Err(e) => {
                ctx.emit(&Event::OnError {
                    error: e.to_string(),
                    run_id,
                });
                return Err(e);
            }
        };

        // Atomic merge.
        state.apply(out.update);

        // Persist checkpoint if configured.
        if let Some(cp) = checkpointer {
            cp.save(run_id, step, &state).await?;
        }

        ctx.emit(&Event::OnNodeEnd {
            node: active_name.clone(),
            step,
            output: serde_json::Value::Null,
            run_id,
        });

        // Route.
        match out.goto {
            Goto::End => {
                config.emit(&Event::OnEnd {
                    runnable: format!("graph[{start_name}]"),
                    run_id,
                    output: serde_json::Value::Null,
                });
                return Ok(state);
            }
            Goto::Node(next_name) => {
                let next = graph
                    .nodes
                    .get(&next_name)
                    .ok_or_else(|| {
                        CognisError::Configuration(format!(
                            "node `{active_name}` routed to unknown node `{next_name}`"
                        ))
                    })?
                    .clone();
                active = next;
                active_name = next_name;
            }
            Goto::Multiple(targets) => {
                // Slice 1: route to first target only. Slice 2 will run all
                // targets in parallel and merge their updates.
                if targets.is_empty() {
                    return Err(CognisError::Configuration(format!(
                        "node `{active_name}` returned Goto::Multiple([])"
                    )));
                }
                if targets.len() > 1 {
                    tracing::warn!(
                        ?targets,
                        node = %active_name,
                        "Goto::Multiple with >1 target falls back to first target in slice 1; \
                         true parallel fan-out lands in slice 2"
                    );
                }
                let next_name = targets.into_iter().next().unwrap();
                let next = graph
                    .nodes
                    .get(&next_name)
                    .ok_or_else(|| {
                        CognisError::Configuration(format!(
                            "node `{active_name}` routed to unknown node `{next_name}`"
                        ))
                    })?
                    .clone();
                active = next;
                active_name = next_name;
            }
        }
    }

    Err(CognisError::RecursionLimit {
        limit: recursion_limit,
    })
}
