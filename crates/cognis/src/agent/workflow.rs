//! Workflow — a typed sequence of agent / runnable invocations sharing
//! a [`WorkflowState`].
//!
//! Rides on the existing `cognis-graph` engine: a `Workflow` is just a
//! `Graph<WorkflowState>` with linear edges, but the API is friendlier
//! for the common "agent A → agent B → agent C" use case.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;

use cognis_core::{Result, Runnable, RunnableConfig};
use cognis_graph::{node_fn, Goto, Graph, GraphState, NodeOut};

/// Carrier for shared cross-step data. Append-only `outputs` map keyed
/// by step name; the latest emitted output is exposed as `last`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkflowState {
    /// Per-step output strings, keyed by step name.
    pub outputs: std::collections::HashMap<String, String>,
    /// Most-recent emitted output.
    pub last: String,
}

/// Manual `GraphState` impl: `outputs` keys overwrite; `last` replaces.
impl GraphState for WorkflowState {
    type Update = WorkflowStateUpdate;
    fn apply(&mut self, u: Self::Update) {
        for (k, v) in u.outputs {
            self.outputs.insert(k, v);
        }
        if let Some(last) = u.last {
            self.last = last;
        }
    }
}

/// Update shape for `WorkflowState`.
#[derive(Debug, Default, Clone)]
pub struct WorkflowStateUpdate {
    /// Map entries to insert / overwrite.
    pub outputs: std::collections::HashMap<String, String>,
    /// Replacement `last` value, if any.
    pub last: Option<String>,
}

/// Builds a linear workflow from named steps. Each step is a
/// `Runnable<String, String>` — the step takes the current `last` value
/// and emits its output.
pub struct Workflow {
    steps: Vec<(String, Arc<dyn Runnable<String, String>>)>,
}

impl Default for Workflow {
    fn default() -> Self {
        Self::new()
    }
}

impl Workflow {
    /// Empty workflow.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a named step.
    pub fn step(mut self, name: impl Into<String>, r: Arc<dyn Runnable<String, String>>) -> Self {
        self.steps.push((name.into(), r));
        self
    }

    /// Run the workflow with `initial` as the seed `last` value. Returns
    /// the final state.
    pub async fn run(&self, initial: impl Into<String>) -> Result<WorkflowState> {
        if self.steps.is_empty() {
            return Ok(WorkflowState {
                outputs: Default::default(),
                last: initial.into(),
            });
        }
        let mut g = Graph::<WorkflowState>::new();
        let n = self.steps.len();
        for (i, (name, runnable)) in self.steps.iter().enumerate() {
            let next_node = if i + 1 < n {
                Some(self.steps[i + 1].0.clone())
            } else {
                None
            };
            let r = runnable.clone();
            let step_name = name.clone();
            g = g.node(
                name.clone(),
                node_fn::<WorkflowState, _, _>(name.clone(), move |state, ctx| {
                    let r = r.clone();
                    let step_name = step_name.clone();
                    let next = next_node.clone();
                    let last = state.last.clone();
                    let cfg = ctx.config.clone();
                    async move {
                        let out = r.invoke(last, cfg).await?;
                        let mut outputs = std::collections::HashMap::new();
                        outputs.insert(step_name, out.clone());
                        Ok(NodeOut {
                            update: WorkflowStateUpdate {
                                outputs,
                                last: Some(out),
                            },
                            goto: match next {
                                Some(n) => Goto::node(n),
                                None => Goto::end(),
                            },
                        })
                    }
                }),
            );
        }
        let compiled = g.start_at(self.steps[0].0.clone()).compile()?;
        let initial_state = WorkflowState {
            outputs: Default::default(),
            last: initial.into(),
        };
        compiled
            .invoke(initial_state, RunnableConfig::default())
            .await
    }
}

#[async_trait]
impl Runnable<String, WorkflowState> for Workflow {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<WorkflowState> {
        self.run(input).await
    }
    fn name(&self) -> &str {
        "Workflow"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::compose::Lambda;

    #[tokio::test]
    async fn linear_workflow_passes_output_to_next() {
        let upper = Lambda::from_async(|s: String| async move { Ok(s.to_uppercase()) });
        let exclaim = Lambda::from_async(|s: String| async move { Ok(format!("{s}!")) });
        let wf = Workflow::new()
            .step("upper", Arc::new(upper))
            .step("exclaim", Arc::new(exclaim));
        let final_state = wf.run("hello").await.unwrap();
        assert_eq!(final_state.last, "HELLO!");
        assert_eq!(final_state.outputs["upper"], "HELLO");
        assert_eq!(final_state.outputs["exclaim"], "HELLO!");
    }
}
