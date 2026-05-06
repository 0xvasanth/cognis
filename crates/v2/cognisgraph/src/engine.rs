//! Pregel-style superstep executor. Slice 2c: true parallel multi-node
//! supersteps + Goto::Send per-target payloads + interrupt before/after
//! with checkpointed resume.

use std::sync::Arc;

use cognis2_core::{CognisError, Event, InterruptKind, Result, RunnableConfig};
use uuid::Uuid;

use crate::checkpoint::Checkpointer;
use crate::compiled::CompiledGraph;
use crate::goto::Goto;
use crate::node::{Node, NodeCtx, NodeOut};
use crate::state::GraphState;

struct ActiveTask<S: GraphState> {
    name: String,
    node: Arc<dyn Node<S>>,
    payload: Option<serde_json::Value>,
}

/// Run the graph from `initial_state` until termination, error, cancellation,
/// or interrupt.
pub(crate) async fn run<S>(
    compiled: &CompiledGraph<S>,
    initial_state: S,
    config: RunnableConfig,
) -> Result<S>
where
    S: GraphState + Clone,
    S::Update: Clone,
{
    validate_interrupt_names(compiled)?;

    let start_name = compiled
        .graph
        .start
        .clone()
        .ok_or_else(|| CognisError::Configuration("graph has no start node".into()))?;
    let start_node = compiled
        .graph
        .nodes
        .get(&start_name)
        .ok_or_else(|| {
            CognisError::Configuration(format!("start node `{start_name}` missing at runtime"))
        })?
        .clone();

    let initial_active = vec![ActiveTask {
        name: start_name.clone(),
        node: start_node,
        payload: None,
    }];

    config.emit(&Event::OnStart {
        runnable: format!("graph[{start_name}]"),
        run_id: config.run_id,
        input: serde_json::Value::Null,
    });

    superstep_loop(compiled, initial_state, &config, initial_active, 0).await
}

/// Resume an interrupted run. `start_step` is the step counter the original
/// run had reached when the interrupt fired; resume continues numbering from
/// there so checkpoint timeline stays linear.
///
/// Slice 2c: resume re-dispatches the start node with the caller-supplied state.
/// A future slice will persist the active set in the checkpoint for true
/// point-of-interrupt resume.
pub(crate) async fn resume<S>(
    compiled: &CompiledGraph<S>,
    state: S,
    config: RunnableConfig,
    start_step: u64,
) -> Result<S>
where
    S: GraphState + Clone,
    S::Update: Clone,
{
    validate_interrupt_names(compiled)?;

    let start_name = compiled
        .graph
        .start
        .clone()
        .ok_or_else(|| CognisError::Configuration("graph has no start node".into()))?;
    let start_node = compiled
        .graph
        .nodes
        .get(&start_name)
        .ok_or_else(|| {
            CognisError::Configuration(format!("start node `{start_name}` missing"))
        })?
        .clone();

    let active = vec![ActiveTask {
        name: start_name,
        node: start_node,
        payload: None,
    }];

    superstep_loop(compiled, state, &config, active, start_step).await
}

async fn superstep_loop<S>(
    compiled: &CompiledGraph<S>,
    initial_state: S,
    config: &RunnableConfig,
    initial_active: Vec<ActiveTask<S>>,
    start_step: u64,
) -> Result<S>
where
    S: GraphState + Clone,
    S::Update: Clone,
{
    let mut state = initial_state;
    let mut active = initial_active;
    let recursion_limit = config.recursion_limit;
    let run_id = config.run_id;

    let mut step = start_step;
    let max_step = start_step.saturating_add(recursion_limit as u64);

    while !active.is_empty() {
        if step >= max_step {
            return Err(CognisError::RecursionLimit { limit: recursion_limit });
        }
        if config.is_cancelled() {
            return Err(CognisError::Cancelled);
        }
        if let Some(deadline) = config.deadline {
            if std::time::Instant::now() > deadline {
                return Err(CognisError::Timeout {
                    operation: "graph".into(),
                    timeout_ms: 0,
                });
            }
        }

        // Check interrupt_before for any active task.
        for task in &active {
            if compiled.interrupt_before.contains(&task.name) {
                save_checkpoint(compiled, run_id, step, &state).await?;
                return Err(CognisError::GraphInterrupted {
                    run_id,
                    step,
                    node: task.name.clone(),
                    kind: InterruptKind::Before,
                });
            }
        }

        // Emit per-node OnNodeStart for each task.
        for task in &active {
            config.emit(&Event::OnNodeStart {
                node: task.name.clone(),
                step,
                run_id,
            });
        }

        // Run all tasks in parallel.
        let task_outputs = run_tasks_parallel(&active, &state, config, step).await?;

        // Atomic merge + OnNodeEnd.
        for (i, output) in task_outputs.iter().enumerate() {
            state.apply(output.clone_update());
            config.emit(&Event::OnNodeEnd {
                node: active[i].name.clone(),
                step,
                output: serde_json::Value::Null,
                run_id,
            });
        }

        // Check interrupt_after for any task that just ran.
        for task in &active {
            if compiled.interrupt_after.contains(&task.name) {
                save_checkpoint(compiled, run_id, step, &state).await?;
                return Err(CognisError::GraphInterrupted {
                    run_id,
                    step,
                    node: task.name.clone(),
                    kind: InterruptKind::After,
                });
            }
        }

        // Snapshot post-merge state.
        save_checkpoint(compiled, run_id, step, &state).await?;

        // Compute next_active. End anywhere terminates the whole graph.
        let mut next_active: Vec<ActiveTask<S>> = Vec::new();
        let mut should_end = false;
        for output in task_outputs {
            match output.goto {
                Goto::End => {
                    should_end = true;
                }
                Goto::Node(name) => {
                    let node = lookup_node(&compiled.graph, &name)?;
                    next_active.push(ActiveTask { name, node, payload: None });
                }
                Goto::Multiple(names) => {
                    for name in names {
                        let node = lookup_node(&compiled.graph, &name)?;
                        next_active.push(ActiveTask { name, node, payload: None });
                    }
                }
                Goto::Send(targets) => {
                    for (name, payload) in targets {
                        let node = lookup_node(&compiled.graph, &name)?;
                        next_active.push(ActiveTask {
                            name,
                            node,
                            payload: Some(payload),
                        });
                    }
                }
            }
        }

        if should_end {
            config.emit(&Event::OnEnd {
                runnable: "graph".into(),
                run_id,
                output: serde_json::Value::Null,
            });
            return Ok(state);
        }

        active = next_active;
        step += 1;
    }

    // No more active tasks but no End emitted — terminate gracefully.
    config.emit(&Event::OnEnd {
        runnable: "graph".into(),
        run_id,
        output: serde_json::Value::Null,
    });
    Ok(state)
}

/// Captured per-task output; decoupled from `NodeOut` so we can move the
/// `Goto` out without re-borrowing.
struct TaskOutput<S: GraphState> {
    update: S::Update,
    goto: Goto,
}

impl<S: GraphState> TaskOutput<S> {
    fn clone_update(&self) -> S::Update
    where
        S::Update: Clone,
    {
        self.update.clone()
    }
}

async fn run_tasks_parallel<S>(
    tasks: &[ActiveTask<S>],
    state: &S,
    config: &RunnableConfig,
    step: u64,
) -> Result<Vec<TaskOutput<S>>>
where
    S: GraphState + Clone,
    S::Update: Clone,
{
    use futures::future::try_join_all;

    let run_id = config.run_id;

    // Snapshot state + payload for each task before spawning. This ensures
    // each future is self-contained (no borrowed references crossing await).
    let task_futs: Vec<_> = tasks
        .iter()
        .map(|task| {
            let node = task.node.clone();
            let state_snap = state.clone();
            let payload_owned = task.payload.clone();
            // We can't hold a borrow to &RunnableConfig across an await that
            // might not complete immediately in the parallel path, so we clone
            // the lightweight config (observers are Arc, so clone is O(n_obs)).
            let config_snap = config.clone();

            async move {
                // NodeCtx borrows from the owned `payload_owned` which lives
                // for the duration of this async block.
                let ctx = NodeCtx::new(run_id, step, &config_snap);
                let ctx = if let Some(ref p) = payload_owned {
                    ctx.with_payload(p)
                } else {
                    ctx
                };
                let out: NodeOut<S> = node.execute(&state_snap, &ctx).await?;
                Ok::<TaskOutput<S>, CognisError>(TaskOutput {
                    update: out.update,
                    goto: out.goto,
                })
            }
        })
        .collect();

    // For small task counts try_join_all is simpler than buffer_unordered.
    // For max_concurrency control we could use buffer_unordered, but the
    // standard engine pattern is to run a full superstep in parallel.
    // max_concurrency primarily governs Runnable::batch; graph parallelism
    // in a superstep is always unbounded within the superstep.
    let results = try_join_all(task_futs).await?;
    Ok(results)
}

fn lookup_node<S: GraphState>(
    graph: &crate::builder::Graph<S>,
    name: &str,
) -> Result<Arc<dyn Node<S>>> {
    graph
        .nodes
        .get(name)
        .cloned()
        .ok_or_else(|| CognisError::Configuration(format!("node `{name}` not registered")))
}

fn validate_interrupt_names<S>(compiled: &CompiledGraph<S>) -> Result<()>
where
    S: GraphState,
{
    let interrupts_used =
        !compiled.interrupt_before.is_empty() || !compiled.interrupt_after.is_empty();
    if interrupts_used && compiled.checkpointer.is_none() {
        return Err(CognisError::Configuration(
            "interrupts require a checkpointer; attach via .with_checkpointer(...)".into(),
        ));
    }
    for name in &compiled.interrupt_before {
        if !compiled.graph.nodes.contains_key(name) {
            return Err(CognisError::Configuration(format!(
                "interrupt_before references unknown node `{name}`"
            )));
        }
    }
    for name in &compiled.interrupt_after {
        if !compiled.graph.nodes.contains_key(name) {
            return Err(CognisError::Configuration(format!(
                "interrupt_after references unknown node `{name}`"
            )));
        }
    }
    Ok(())
}

async fn save_checkpoint<S>(
    compiled: &CompiledGraph<S>,
    run_id: Uuid,
    step: u64,
    state: &S,
) -> Result<()>
where
    S: GraphState + Clone,
{
    if let Some(cp) = &compiled.checkpointer {
        cp.save(run_id, step, state).await?;
    }
    Ok(())
}
