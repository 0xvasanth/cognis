//! Declarative tool orchestration.
//!
//! Build an [`ExecutionPlan`] of [`ToolStep`]s with declared dependencies,
//! hand it to [`ToolOrchestrator::run`], and let the orchestrator
//! resolve a concurrent execution schedule. Independent steps run in
//! parallel (capped by `max_concurrency`); dependents wait for their
//! upstream results.
//!
//! Cycles are detected at planning time and surfaced as
//! `CognisError::Configuration`. Missing tools / missing dependency
//! references are also caught before any tool runs.
//!
//! Use case: an agent decides "fetch X, fetch Y, then summarize both".
//! Express it as three steps with the summary depending on the two
//! fetches; the orchestrator runs the fetches in parallel and summarizes
//! once both finish.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};

/// One step in an [`ExecutionPlan`].
#[derive(Clone)]
pub struct ToolStep {
    /// Unique step id within the plan. Used by other steps to declare
    /// dependencies and by [`OrchestratorResult`] to key the output.
    pub id: String,
    /// Name of the registered tool to invoke (matches [`Tool::name`]).
    pub tool: String,
    /// Static args for the tool. For dynamic interpolation from prior
    /// step outputs, use a custom step builder before calling `run`.
    pub args: ToolInput,
    /// Step ids whose results must be available before this step runs.
    /// An empty list means the step is eligible immediately.
    pub depends_on: Vec<String>,
}

impl ToolStep {
    /// New step with no dependencies.
    pub fn new(id: impl Into<String>, tool: impl Into<String>, args: ToolInput) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            args,
            depends_on: Vec::new(),
        }
    }

    /// Declare upstream dependencies. Builder-style.
    pub fn after<I, S>(mut self, deps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.depends_on.extend(deps.into_iter().map(Into::into));
        self
    }
}

/// Plan = ordered list of steps. Order doesn't drive execution — the
/// `depends_on` graph does — but it is the deterministic tiebreaker
/// when steps in the same batch are scheduled.
#[derive(Default, Clone)]
pub struct ExecutionPlan {
    /// All steps in the plan.
    pub steps: Vec<ToolStep>,
}

impl ExecutionPlan {
    /// Empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step. Builder-style.
    pub fn step(mut self, step: ToolStep) -> Self {
        self.steps.push(step);
        self
    }
}

/// Orchestrator: register tools, run plans.
pub struct ToolOrchestrator {
    tools: HashMap<String, Arc<dyn Tool>>,
    max_concurrency: usize,
}

impl Default for ToolOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolOrchestrator {
    /// Empty orchestrator with default concurrency = 8.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            max_concurrency: 8,
        }
    }

    /// Register a tool. Subsequent registrations under the same name
    /// overwrite the prior entry.
    pub fn register(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    /// Cap concurrent in-flight tool calls within a batch.
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// Validate the plan: every dep must point at a step in the plan,
    /// every step must reference a registered tool, the dependency
    /// graph must be a DAG.
    fn validate(&self, plan: &ExecutionPlan) -> Result<()> {
        let mut ids: HashSet<&str> = HashSet::with_capacity(plan.steps.len());
        for s in &plan.steps {
            if !ids.insert(s.id.as_str()) {
                return Err(CognisError::Configuration(format!(
                    "duplicate step id `{}`",
                    s.id
                )));
            }
        }
        for s in &plan.steps {
            if !self.tools.contains_key(&s.tool) {
                return Err(CognisError::Configuration(format!(
                    "step `{}` references unregistered tool `{}`",
                    s.id, s.tool
                )));
            }
            for d in &s.depends_on {
                if !ids.contains(d.as_str()) {
                    return Err(CognisError::Configuration(format!(
                        "step `{}` depends on unknown step `{}`",
                        s.id, d
                    )));
                }
            }
        }
        Ok(())
    }

    /// Topo-sort into batches. Each returned batch contains steps whose
    /// dependencies are satisfied by earlier batches; steps within a
    /// batch can run concurrently. Errors on cycle.
    fn batches(plan: &ExecutionPlan) -> Result<Vec<Vec<ToolStep>>> {
        let mut indeg: HashMap<String, usize> = plan
            .steps
            .iter()
            .map(|s| (s.id.clone(), s.depends_on.len()))
            .collect();
        let mut by_id: HashMap<String, ToolStep> =
            plan.steps.iter().map(|s| (s.id.clone(), s.clone())).collect();
        // reverse adjacency: dep_id -> list of step ids that depend on dep_id
        let mut rev: HashMap<String, Vec<String>> = HashMap::new();
        for s in &plan.steps {
            for d in &s.depends_on {
                rev.entry(d.clone()).or_default().push(s.id.clone());
            }
        }

        let mut batches: Vec<Vec<ToolStep>> = Vec::new();
        let mut ready: Vec<String> = indeg
            .iter()
            .filter(|(_, &n)| n == 0)
            .map(|(id, _)| id.clone())
            .collect();
        ready.sort();
        let mut consumed = 0usize;

        while !ready.is_empty() {
            let mut batch = Vec::with_capacity(ready.len());
            let current = std::mem::take(&mut ready);
            let mut next_ready: VecDeque<String> = VecDeque::new();
            for id in &current {
                let s = by_id
                    .remove(id)
                    .expect("ready id without step is impossible");
                batch.push(s);
                consumed += 1;
                if let Some(downstream) = rev.get(id) {
                    for d in downstream {
                        if let Some(n) = indeg.get_mut(d) {
                            *n -= 1;
                            if *n == 0 {
                                next_ready.push_back(d.clone());
                            }
                        }
                    }
                }
            }
            batches.push(batch);
            let mut nr: Vec<String> = next_ready.into();
            nr.sort();
            ready = nr;
        }

        if consumed != plan.steps.len() {
            return Err(CognisError::Configuration(
                "execution plan has a dependency cycle".into(),
            ));
        }
        Ok(batches)
    }

    /// Run the plan. Independent steps within a batch run concurrently
    /// (capped at `max_concurrency`); the next batch starts only after
    /// the prior one fully completes.
    ///
    /// On any tool error, that step's error is recorded and dependents
    /// are *not* cancelled — sibling steps in the same batch finish, and
    /// downstream batches proceed but skip steps whose ancestors errored.
    pub async fn run(&self, plan: ExecutionPlan) -> Result<OrchestratorResult> {
        self.validate(&plan)?;
        let batches = Self::batches(&plan)?;

        let mut results: HashMap<String, ToolOutput> = HashMap::new();
        let mut errors: HashMap<String, CognisError> = HashMap::new();
        let mut errored_ancestors: HashSet<String> = HashSet::new();

        for batch in batches {
            // Skip steps whose ancestors errored.
            let runnable: Vec<ToolStep> = batch
                .into_iter()
                .filter(|s| {
                    if s.depends_on.iter().any(|d| errored_ancestors.contains(d)) {
                        errored_ancestors.insert(s.id.clone());
                        false
                    } else {
                        true
                    }
                })
                .collect();

            type StepFut = BoxFuture<'static, (String, Result<ToolOutput>)>;
            let mut futs: FuturesUnordered<StepFut> = FuturesUnordered::new();
            let mut iter = runnable.into_iter();

            let spawn = |step: ToolStep, tools: &HashMap<String, Arc<dyn Tool>>| -> StepFut {
                let tool = tools.get(&step.tool).expect("validated").clone();
                let id = step.id.clone();
                let args = step.args.clone();
                Box::pin(async move { (id, tool._run(args).await) })
            };

            // Initial fill.
            while futs.len() < self.max_concurrency {
                let Some(step) = iter.next() else { break };
                futs.push(spawn(step, &self.tools));
            }

            // Drain + refill.
            while let Some((id, res)) = futs.next().await {
                match res {
                    Ok(out) => {
                        results.insert(id, out);
                    }
                    Err(e) => {
                        errored_ancestors.insert(id.clone());
                        errors.insert(id, e);
                    }
                }
                if let Some(step) = iter.next() {
                    futs.push(spawn(step, &self.tools));
                }
            }
        }

        Ok(OrchestratorResult { results, errors })
    }
}

/// Outcome of running an [`ExecutionPlan`]. Per-step success is in
/// `results`; per-step failure (or a skipped step whose ancestor
/// errored) is in `errors`.
#[derive(Debug, Default)]
pub struct OrchestratorResult {
    /// Step id → tool output for every successful step.
    pub results: HashMap<String, ToolOutput>,
    /// Step id → error for every failed step. Steps skipped because an
    /// ancestor errored are *not* in `errors` (they simply don't appear
    /// in `results`).
    pub errors: HashMap<String, CognisError>,
}

impl OrchestratorResult {
    /// `true` if every step in the plan produced a result.
    pub fn fully_succeeded(&self) -> bool {
        self.errors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Tool that returns its name + the input as JSON, optionally after
    /// a sleep, optionally erroring.
    struct ScriptedTool {
        name: &'static str,
        sleep_ms: u64,
        fail: bool,
        calls: Arc<AtomicUsize>,
    }

    impl ScriptedTool {
        fn new(name: &'static str) -> (Arc<Self>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            let t = Arc::new(Self {
                name,
                sleep_ms: 0,
                fail: false,
                calls: calls.clone(),
            });
            (t, calls)
        }

        fn slow(name: &'static str, sleep_ms: u64) -> (Arc<Self>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            let t = Arc::new(Self {
                name,
                sleep_ms,
                fail: false,
                calls: calls.clone(),
            });
            (t, calls)
        }

        fn failing(name: &'static str) -> (Arc<Self>, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            let t = Arc::new(Self {
                name,
                sleep_ms: 0,
                fail: true,
                calls: calls.clone(),
            });
            (t, calls)
        }
    }

    #[async_trait]
    impl Tool for ScriptedTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            None
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.sleep_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
            }
            if self.fail {
                return Err(CognisError::Internal(format!("{} failed", self.name)));
            }
            Ok(ToolOutput::Content(json!({
                "tool": self.name,
                "input": input.into_json(),
            })))
        }
    }

    fn args(text: &str) -> ToolInput {
        ToolInput::Text(text.to_string())
    }

    #[tokio::test]
    async fn runs_independent_steps_concurrently() {
        let (a, _) = ScriptedTool::slow("a", 60);
        let (b, _) = ScriptedTool::slow("b", 60);
        let orch = ToolOrchestrator::new()
            .register(a)
            .register(b)
            .with_max_concurrency(2);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("hi")))
            .step(ToolStep::new("s2", "b", args("hi")));
        let start = std::time::Instant::now();
        let r = orch.run(plan).await.unwrap();
        let elapsed = start.elapsed();
        assert!(r.fully_succeeded());
        assert!(
            elapsed < Duration::from_millis(110),
            "expected concurrent run, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn dependent_step_waits_for_ancestor() {
        let (a, a_calls) = ScriptedTool::new("a");
        let (b, b_calls) = ScriptedTool::new("b");
        let orch = ToolOrchestrator::new().register(a).register(b);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("first")))
            .step(ToolStep::new("s2", "b", args("second")).after(["s1"]));
        let r = orch.run(plan).await.unwrap();
        assert!(r.fully_succeeded());
        assert_eq!(a_calls.load(Ordering::Relaxed), 1);
        assert_eq!(b_calls.load(Ordering::Relaxed), 1);
        assert!(r.results.contains_key("s1"));
        assert!(r.results.contains_key("s2"));
    }

    #[tokio::test]
    async fn descendants_skipped_when_ancestor_errors() {
        let (a, _) = ScriptedTool::failing("a");
        let (b, b_calls) = ScriptedTool::new("b");
        let orch = ToolOrchestrator::new().register(a).register(b);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("x")))
            .step(ToolStep::new("s2", "b", args("y")).after(["s1"]));
        let r = orch.run(plan).await.unwrap();
        assert!(!r.fully_succeeded());
        assert!(r.errors.contains_key("s1"));
        assert!(!r.results.contains_key("s2"));
        assert_eq!(b_calls.load(Ordering::Relaxed), 0, "downstream skipped");
    }

    #[tokio::test]
    async fn cycle_is_rejected() {
        let (a, _) = ScriptedTool::new("a");
        let orch = ToolOrchestrator::new().register(a);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("x")).after(["s2"]))
            .step(ToolStep::new("s2", "a", args("y")).after(["s1"]));
        let err = orch.run(plan).await.unwrap_err();
        assert!(err.to_string().contains("cycle"), "got: {err}");
    }

    #[tokio::test]
    async fn unknown_tool_is_rejected() {
        let orch = ToolOrchestrator::new();
        let plan =
            ExecutionPlan::new().step(ToolStep::new("s1", "ghost", args("x")));
        let err = orch.run(plan).await.unwrap_err();
        assert!(err.to_string().contains("unregistered"), "got: {err}");
    }

    #[tokio::test]
    async fn unknown_dep_is_rejected() {
        let (a, _) = ScriptedTool::new("a");
        let orch = ToolOrchestrator::new().register(a);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("x")).after(["does-not-exist"]));
        let err = orch.run(plan).await.unwrap_err();
        assert!(err.to_string().contains("unknown step"), "got: {err}");
    }

    #[tokio::test]
    async fn diamond_runs_correctly() {
        // s1 → s2 ┐
        // s1 → s3 ┴ → s4
        let (t, calls) = ScriptedTool::new("t");
        let orch = ToolOrchestrator::new().register(t).with_max_concurrency(4);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "t", args("a")))
            .step(ToolStep::new("s2", "t", args("b")).after(["s1"]))
            .step(ToolStep::new("s3", "t", args("c")).after(["s1"]))
            .step(ToolStep::new("s4", "t", args("d")).after(["s2", "s3"]));
        let r = orch.run(plan).await.unwrap();
        assert!(r.fully_succeeded());
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        for id in ["s1", "s2", "s3", "s4"] {
            assert!(r.results.contains_key(id), "missing {id}");
        }
    }

    #[tokio::test]
    async fn duplicate_step_id_rejected() {
        let (a, _) = ScriptedTool::new("a");
        let orch = ToolOrchestrator::new().register(a);
        let plan = ExecutionPlan::new()
            .step(ToolStep::new("s1", "a", args("x")))
            .step(ToolStep::new("s1", "a", args("y")));
        let err = orch.run(plan).await.unwrap_err();
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }
}
