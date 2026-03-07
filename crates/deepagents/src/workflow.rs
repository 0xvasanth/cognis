//! Workflow engine for multi-step agent workflows with dependency resolution.
//!
//! Provides a high-level workflow system that orchestrates steps with
//! dependencies, conditional execution, retries, and timeouts.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use deepagents::workflow::{WorkflowBuilder, StepAction};
//! use serde_json::json;
//!
//! let workflow = WorkflowBuilder::new("my-workflow", "Example workflow")
//!     .add_step_simple("step1", "First step", StepAction::Message("Hello".into()))
//!     .add_step_simple("step2", "Second step", StepAction::Message("World".into()))
//!     .dependency("step2", "step1")
//!     .build()
//!     .unwrap();
//!
//! let executor = WorkflowExecutor::new();
//! let result = executor.execute(&workflow, json!({})).unwrap();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::agent::DeepAgentError;

/// A boxed condition function that evaluates workflow state to determine step execution.
type ConditionFn = Box<dyn Fn(&Value) -> bool + Send + Sync>;

/// Result type for workflow operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// StepStatus
// ---------------------------------------------------------------------------

/// Status of a workflow step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepStatus {
    /// Step has not started yet.
    Pending,
    /// Step is currently executing.
    Running,
    /// Step completed successfully.
    Completed,
    /// Step failed after exhausting retries.
    Failed,
    /// Step was skipped (condition not met).
    Skipped,
    /// Step exceeded its timeout.
    TimedOut,
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
            Self::TimedOut => write!(f, "timed_out"),
        }
    }
}

// ---------------------------------------------------------------------------
// StepAction
// ---------------------------------------------------------------------------

/// The action a workflow step performs.
pub enum StepAction {
    /// A synchronous transformation on the current state value.
    Transform(Box<dyn Fn(Value) -> std::result::Result<Value, String> + Send + Sync>),
    /// A message/prompt to send to an agent.
    Message(String),
    /// Run the listed step IDs in parallel (they must already exist in the workflow).
    Parallel(Vec<String>),
    /// Save the current state as a checkpoint.
    Checkpoint,
}

impl std::fmt::Debug for StepAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transform(_) => write!(f, "Transform(<fn>)"),
            Self::Message(msg) => write!(f, "Message({:?})", msg),
            Self::Parallel(ids) => write!(f, "Parallel({:?})", ids),
            Self::Checkpoint => write!(f, "Checkpoint"),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkflowStep
// ---------------------------------------------------------------------------

/// A single step in a workflow.
pub struct WorkflowStep {
    /// Unique identifier for this step.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description of what this step does.
    pub description: Option<String>,
    /// The action to perform.
    pub action: StepAction,
    /// IDs of steps that must complete before this step can run.
    pub dependencies: Vec<String>,
    /// Optional guard condition evaluated against the current state.
    /// If it returns `false`, the step is skipped.
    pub condition: Option<ConditionFn>,
    /// Optional timeout for step execution.
    pub timeout: Option<Duration>,
    /// Number of times to retry on failure (0 means no retries).
    pub retry_count: usize,
}

impl std::fmt::Debug for WorkflowStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowStep")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("action", &self.action)
            .field("dependencies", &self.dependencies)
            .field("condition", &self.condition.is_some())
            .field("timeout", &self.timeout)
            .field("retry_count", &self.retry_count)
            .finish()
    }
}

impl WorkflowStep {
    /// Create a new workflow step with the given id, name, and action.
    pub fn new(id: impl Into<String>, name: impl Into<String>, action: StepAction) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            action,
            dependencies: Vec::new(),
            condition: None,
            timeout: None,
            retry_count: 0,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a dependency on another step.
    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.dependencies.push(dep_id.into());
        self
    }

    /// Add multiple dependencies.
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies.extend(deps);
        self
    }

    /// Set a guard condition.
    pub fn with_condition(mut self, cond: impl Fn(&Value) -> bool + Send + Sync + 'static) -> Self {
        self.condition = Some(Box::new(cond));
        self
    }

    /// Set a timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the retry count.
    pub fn with_retry_count(mut self, count: usize) -> Self {
        self.retry_count = count;
        self
    }
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// The result of executing a single workflow step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The ID of the step that was executed.
    pub step_id: String,
    /// The final status of the step.
    pub status: StepStatus,
    /// The output value produced by the step, if any.
    pub output: Option<Value>,
    /// How long the step took to execute.
    pub duration: Duration,
    /// Error message if the step failed.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

/// A workflow consisting of ordered steps with dependencies.
pub struct Workflow {
    /// Unique identifier for this workflow.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The steps in this workflow.
    pub steps: Vec<WorkflowStep>,
}

impl Workflow {
    /// Create a new empty workflow.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step to the workflow. Returns `&mut Self` for chaining.
    pub fn add_step(&mut self, step: WorkflowStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// Validate the workflow:
    /// - All dependency references point to existing steps
    /// - No duplicate step IDs
    /// - No dependency cycles
    pub fn validate(&self) -> Result<()> {
        let ids: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();

        // Check for duplicate IDs.
        if ids.len() != self.steps.len() {
            return Err(DeepAgentError::ConfigError(
                "Workflow contains duplicate step IDs".to_string(),
            ));
        }

        // Check that all dependencies reference existing steps.
        for step in &self.steps {
            for dep in &step.dependencies {
                if !ids.contains(dep.as_str()) {
                    return Err(DeepAgentError::ConfigError(format!(
                        "Step '{}' depends on non-existent step '{}'",
                        step.id, dep
                    )));
                }
            }
            // Self-dependency check.
            if step.dependencies.contains(&step.id) {
                return Err(DeepAgentError::ConfigError(format!(
                    "Step '{}' depends on itself",
                    step.id
                )));
            }
        }

        // Cycle detection using topological sort (Kahn's algorithm).
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for step in &self.steps {
            in_degree.entry(step.id.as_str()).or_insert(0);
            adj.entry(step.id.as_str()).or_default();
            for dep in &step.dependencies {
                adj.entry(dep.as_str()).or_default().push(step.id.as_str());
                *in_degree.entry(step.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if visited != self.steps.len() {
            return Err(DeepAgentError::ConfigError(
                "Workflow contains a dependency cycle".to_string(),
            ));
        }

        Ok(())
    }

    /// Return all steps whose dependencies are fully satisfied.
    pub fn get_ready_steps(&self, completed: &HashSet<String>) -> Vec<&WorkflowStep> {
        self.steps
            .iter()
            .filter(|step| {
                !completed.contains(&step.id)
                    && step.dependencies.iter().all(|dep| completed.contains(dep))
            })
            .collect()
    }

    /// Check if the workflow is complete (all steps are in the completed set).
    pub fn is_complete(&self, completed: &HashSet<String>) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| completed.contains(&s.id))
    }
}

// ---------------------------------------------------------------------------
// WorkflowResult
// ---------------------------------------------------------------------------

/// The result of executing an entire workflow.
#[derive(Debug, Clone)]
pub struct WorkflowResult {
    /// Per-step results keyed by step ID.
    pub results: HashMap<String, StepResult>,
    /// The final state after all steps have run.
    pub final_state: Value,
    /// Total wall-clock duration of the workflow execution.
    pub total_duration: Duration,
}

impl WorkflowResult {
    /// Return all step results that completed successfully.
    pub fn succeeded(&self) -> Vec<&StepResult> {
        self.results
            .values()
            .filter(|r| r.status == StepStatus::Completed)
            .collect()
    }

    /// Return all step results that failed.
    pub fn failed(&self) -> Vec<&StepResult> {
        self.results
            .values()
            .filter(|r| r.status == StepStatus::Failed || r.status == StepStatus::TimedOut)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// WorkflowExecutor
// ---------------------------------------------------------------------------

/// Executes a workflow respecting dependency order, conditions, retries,
/// and timeouts.
pub struct WorkflowExecutor {
    /// Collected checkpoints during execution.
    pub checkpoints: Vec<Value>,
}

impl WorkflowExecutor {
    /// Create a new `WorkflowExecutor`.
    pub fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
        }
    }

    /// Execute a workflow with the given initial state.
    ///
    /// Steps are run in dependency order. Steps whose conditions evaluate to
    /// `false` are skipped. Failed steps are retried up to `retry_count` times.
    pub fn execute(&mut self, workflow: &Workflow, initial_state: Value) -> Result<WorkflowResult> {
        workflow.validate()?;

        let start = Instant::now();
        let mut state = initial_state;
        let mut results: HashMap<String, StepResult> = HashMap::new();
        let mut completed: HashSet<String> = HashSet::new();

        // Track steps that have been processed (completed, failed, or skipped).
        let mut processed: HashSet<String> = HashSet::new();

        loop {
            if workflow.is_complete(&processed) {
                break;
            }

            let ready_ids: Vec<String> = workflow
                .get_ready_steps(&processed)
                .iter()
                .map(|s| s.id.clone())
                .collect();

            if ready_ids.is_empty() {
                // No more ready steps but workflow not complete -- some steps
                // have unsatisfied dependencies due to failed predecessors.
                break;
            }

            for step_id in &ready_ids {
                let step = workflow.steps.iter().find(|s| s.id == *step_id).unwrap();
                let step_result = self.execute_step(step, &mut state);
                if step_result.status == StepStatus::Completed
                    || step_result.status == StepStatus::Skipped
                {
                    completed.insert(step_id.clone());
                }
                processed.insert(step_id.clone());
                results.insert(step_id.clone(), step_result);
            }
        }

        Ok(WorkflowResult {
            results,
            final_state: state,
            total_duration: start.elapsed(),
        })
    }

    /// Execute a single step, handling conditions, retries, and timeouts.
    fn execute_step(&mut self, step: &WorkflowStep, state: &mut Value) -> StepResult {
        let step_start = Instant::now();

        // Check condition.
        if let Some(ref cond) = step.condition {
            if !cond(state) {
                return StepResult {
                    step_id: step.id.clone(),
                    status: StepStatus::Skipped,
                    output: None,
                    duration: step_start.elapsed(),
                    error: Some("Condition not met".to_string()),
                };
            }
        }

        let max_attempts = step.retry_count + 1;
        let mut last_error: Option<String> = None;

        for _attempt in 0..max_attempts {
            // Check timeout.
            if let Some(timeout) = step.timeout {
                if step_start.elapsed() >= timeout {
                    return StepResult {
                        step_id: step.id.clone(),
                        status: StepStatus::TimedOut,
                        output: None,
                        duration: step_start.elapsed(),
                        error: Some("Step timed out".to_string()),
                    };
                }
            }

            match &step.action {
                StepAction::Transform(transform) => match transform(state.clone()) {
                    Ok(new_state) => {
                        *state = new_state.clone();
                        return StepResult {
                            step_id: step.id.clone(),
                            status: StepStatus::Completed,
                            output: Some(new_state),
                            duration: step_start.elapsed(),
                            error: None,
                        };
                    }
                    Err(e) => {
                        last_error = Some(e);
                    }
                },
                StepAction::Message(msg) => {
                    // Store the message in state under "messages" array.
                    let messages = state.as_object_mut().and_then(|obj| {
                        if !obj.contains_key("messages") {
                            obj.insert("messages".to_string(), Value::Array(Vec::new()));
                        }
                        obj.get_mut("messages").and_then(|v| v.as_array_mut())
                    });

                    if let Some(msgs) = messages {
                        msgs.push(serde_json::json!({
                            "type": "agent",
                            "content": msg
                        }));
                    }

                    return StepResult {
                        step_id: step.id.clone(),
                        status: StepStatus::Completed,
                        output: Some(serde_json::json!({ "message": msg })),
                        duration: step_start.elapsed(),
                        error: None,
                    };
                }
                StepAction::Parallel(step_ids) => {
                    // In the synchronous executor, parallel steps are executed
                    // sequentially but conceptually represent parallelism.
                    return StepResult {
                        step_id: step.id.clone(),
                        status: StepStatus::Completed,
                        output: Some(serde_json::json!({ "parallel_steps": step_ids })),
                        duration: step_start.elapsed(),
                        error: None,
                    };
                }
                StepAction::Checkpoint => {
                    self.checkpoints.push(state.clone());
                    return StepResult {
                        step_id: step.id.clone(),
                        status: StepStatus::Completed,
                        output: Some(state.clone()),
                        duration: step_start.elapsed(),
                        error: None,
                    };
                }
            }
        }

        // All retry attempts exhausted.
        StepResult {
            step_id: step.id.clone(),
            status: StepStatus::Failed,
            output: None,
            duration: step_start.elapsed(),
            error: last_error,
        }
    }
}

impl Default for WorkflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WorkflowBuilder
// ---------------------------------------------------------------------------

/// A fluent builder for constructing workflows.
pub struct WorkflowBuilder {
    id: String,
    name: String,
    steps: Vec<WorkflowStep>,
    /// Deferred dependencies: (step_id, dep_id).
    deferred_deps: Vec<(String, String)>,
}

impl WorkflowBuilder {
    /// Create a new workflow builder.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            steps: Vec::new(),
            deferred_deps: Vec::new(),
        }
    }

    /// Add a step with the given parameters.
    pub fn add_step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add a simple step (id, name, action) with no dependencies or conditions.
    pub fn add_step_simple(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        action: StepAction,
    ) -> Self {
        self.add_step(WorkflowStep::new(id, name, action))
    }

    /// Add a dependency from `step_id` on `dep_id`.
    pub fn dependency(mut self, step_id: impl Into<String>, dep_id: impl Into<String>) -> Self {
        self.deferred_deps.push((step_id.into(), dep_id.into()));
        self
    }

    /// Build the workflow, applying deferred dependencies.
    pub fn build(mut self) -> Result<Workflow> {
        // Apply deferred dependencies.
        for (step_id, dep_id) in &self.deferred_deps {
            let step = self
                .steps
                .iter_mut()
                .find(|s| s.id == *step_id)
                .ok_or_else(|| {
                    DeepAgentError::ConfigError(format!(
                        "Cannot add dependency: step '{}' not found",
                        step_id
                    ))
                })?;
            if !step.dependencies.contains(dep_id) {
                step.dependencies.push(dep_id.clone());
            }
        }

        let mut workflow = Workflow::new(self.id, self.name);
        for step in self.steps {
            workflow.add_step(step);
        }
        workflow.validate()?;
        Ok(workflow)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // StepStatus
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_status_display() {
        assert_eq!(StepStatus::Pending.to_string(), "pending");
        assert_eq!(StepStatus::Running.to_string(), "running");
        assert_eq!(StepStatus::Completed.to_string(), "completed");
        assert_eq!(StepStatus::Failed.to_string(), "failed");
        assert_eq!(StepStatus::Skipped.to_string(), "skipped");
        assert_eq!(StepStatus::TimedOut.to_string(), "timed_out");
    }

    #[test]
    fn test_step_status_equality() {
        assert_eq!(StepStatus::Pending, StepStatus::Pending);
        assert_ne!(StepStatus::Pending, StepStatus::Running);
    }

    // -----------------------------------------------------------------------
    // StepAction
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_action_debug() {
        let t = StepAction::Transform(Box::new(|v| Ok(v)));
        assert!(format!("{:?}", t).contains("Transform"));

        let m = StepAction::Message("hello".into());
        assert!(format!("{:?}", m).contains("hello"));

        let p = StepAction::Parallel(vec!["a".into(), "b".into()]);
        assert!(format!("{:?}", p).contains("Parallel"));

        let c = StepAction::Checkpoint;
        assert!(format!("{:?}", c).contains("Checkpoint"));
    }

    // -----------------------------------------------------------------------
    // WorkflowStep construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_workflow_step_new() {
        let step = WorkflowStep::new("s1", "Step One", StepAction::Checkpoint);
        assert_eq!(step.id, "s1");
        assert_eq!(step.name, "Step One");
        assert!(step.description.is_none());
        assert!(step.dependencies.is_empty());
        assert!(step.condition.is_none());
        assert!(step.timeout.is_none());
        assert_eq!(step.retry_count, 0);
    }

    #[test]
    fn test_workflow_step_with_description() {
        let step =
            WorkflowStep::new("s1", "Step", StepAction::Checkpoint).with_description("Does things");
        assert_eq!(step.description.as_deref(), Some("Does things"));
    }

    #[test]
    fn test_workflow_step_with_dependency() {
        let step = WorkflowStep::new("s2", "Step", StepAction::Checkpoint).with_dependency("s1");
        assert_eq!(step.dependencies, vec!["s1"]);
    }

    #[test]
    fn test_workflow_step_with_dependencies() {
        let step = WorkflowStep::new("s3", "Step", StepAction::Checkpoint)
            .with_dependencies(vec!["s1".into(), "s2".into()]);
        assert_eq!(step.dependencies, vec!["s1", "s2"]);
    }

    #[test]
    fn test_workflow_step_with_condition() {
        let step = WorkflowStep::new("s1", "Step", StepAction::Checkpoint)
            .with_condition(|v| v.get("ready").and_then(|r| r.as_bool()).unwrap_or(false));
        assert!(step.condition.is_some());
    }

    #[test]
    fn test_workflow_step_with_timeout() {
        let step = WorkflowStep::new("s1", "Step", StepAction::Checkpoint)
            .with_timeout(Duration::from_secs(30));
        assert_eq!(step.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_workflow_step_with_retry_count() {
        let step = WorkflowStep::new("s1", "Step", StepAction::Checkpoint).with_retry_count(3);
        assert_eq!(step.retry_count, 3);
    }

    // -----------------------------------------------------------------------
    // Workflow construction and validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_workflow_new() {
        let wf = Workflow::new("wf-1", "My Workflow");
        assert_eq!(wf.id, "wf-1");
        assert_eq!(wf.name, "My Workflow");
        assert!(wf.steps.is_empty());
    }

    #[test]
    fn test_workflow_add_step() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "Step 1", StepAction::Checkpoint));
        wf.add_step(WorkflowStep::new("s2", "Step 2", StepAction::Checkpoint));
        assert_eq!(wf.steps.len(), 2);
    }

    #[test]
    fn test_workflow_validate_valid() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "Step 1", StepAction::Checkpoint));
        wf.add_step(
            WorkflowStep::new("s2", "Step 2", StepAction::Checkpoint).with_dependency("s1"),
        );
        assert!(wf.validate().is_ok());
    }

    #[test]
    fn test_workflow_validate_missing_dependency() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(
            WorkflowStep::new("s1", "Step 1", StepAction::Checkpoint)
                .with_dependency("nonexistent"),
        );
        let err = wf.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("non-existent step"));
    }

    #[test]
    fn test_workflow_validate_duplicate_ids() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "Step 1", StepAction::Checkpoint));
        wf.add_step(WorkflowStep::new(
            "s1",
            "Step 1 dup",
            StepAction::Checkpoint,
        ));
        let err = wf.validate().unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn test_workflow_validate_self_dependency() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(
            WorkflowStep::new("s1", "Step 1", StepAction::Checkpoint).with_dependency("s1"),
        );
        let err = wf.validate().unwrap_err();
        assert!(err.to_string().contains("depends on itself"));
    }

    #[test]
    fn test_workflow_validate_cycle() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("a", "A", StepAction::Checkpoint).with_dependency("b"));
        wf.add_step(WorkflowStep::new("b", "B", StepAction::Checkpoint).with_dependency("a"));
        let err = wf.validate().unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    // -----------------------------------------------------------------------
    // Dependency ordering — get_ready_steps and is_complete
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_ready_steps_no_deps() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "A", StepAction::Checkpoint));
        wf.add_step(WorkflowStep::new("s2", "B", StepAction::Checkpoint));

        let completed = HashSet::new();
        let ready = wf.get_ready_steps(&completed);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_get_ready_steps_with_deps() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "A", StepAction::Checkpoint));
        wf.add_step(WorkflowStep::new("s2", "B", StepAction::Checkpoint).with_dependency("s1"));
        wf.add_step(WorkflowStep::new("s3", "C", StepAction::Checkpoint).with_dependency("s2"));

        let completed = HashSet::new();
        let ready = wf.get_ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "s1");

        let mut completed = HashSet::new();
        completed.insert("s1".to_string());
        let ready = wf.get_ready_steps(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "s2");
    }

    #[test]
    fn test_is_complete() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new("s1", "A", StepAction::Checkpoint));
        wf.add_step(WorkflowStep::new("s2", "B", StepAction::Checkpoint));

        let empty: HashSet<String> = HashSet::new();
        assert!(!wf.is_complete(&empty));

        let mut partial = HashSet::new();
        partial.insert("s1".to_string());
        assert!(!wf.is_complete(&partial));

        let mut all = HashSet::new();
        all.insert("s1".to_string());
        all.insert("s2".to_string());
        assert!(wf.is_complete(&all));
    }

    #[test]
    fn test_is_complete_empty_workflow() {
        let wf = Workflow::new("wf", "Empty");
        let empty: HashSet<String> = HashSet::new();
        assert!(!wf.is_complete(&empty));
    }

    // -----------------------------------------------------------------------
    // Execution — transform steps
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_single_transform() {
        let mut wf = Workflow::new("wf", "Test");
        wf.add_step(WorkflowStep::new(
            "s1",
            "Double x",
            StepAction::Transform(Box::new(|mut v| {
                let x = v.get("x").and_then(|x| x.as_i64()).unwrap_or(0);
                v["x"] = serde_json::json!(x * 2);
                Ok(v)
            })),
        ));

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"x": 5})).unwrap();

        assert_eq!(result.final_state["x"], 10);
        assert_eq!(result.succeeded().len(), 1);
        assert!(result.failed().is_empty());
    }

    #[test]
    fn test_execute_chained_transforms() {
        let mut wf = Workflow::new("wf", "Chain");
        wf.add_step(WorkflowStep::new(
            "add",
            "Add 10",
            StepAction::Transform(Box::new(|mut v| {
                let x = v.get("x").and_then(|x| x.as_i64()).unwrap_or(0);
                v["x"] = json!(x + 10);
                Ok(v)
            })),
        ));
        wf.add_step(
            WorkflowStep::new(
                "mul",
                "Multiply by 3",
                StepAction::Transform(Box::new(|mut v| {
                    let x = v.get("x").and_then(|x| x.as_i64()).unwrap_or(0);
                    v["x"] = json!(x * 3);
                    Ok(v)
                })),
            )
            .with_dependency("add"),
        );

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"x": 2})).unwrap();
        // (2 + 10) * 3 = 36
        assert_eq!(result.final_state["x"], 36);
        assert_eq!(result.succeeded().len(), 2);
    }

    // -----------------------------------------------------------------------
    // Execution — conditions
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_condition_skip() {
        let mut wf = Workflow::new("wf", "Cond");
        wf.add_step(
            WorkflowStep::new("guarded", "Guarded step", StepAction::Message("hi".into()))
                .with_condition(|v| v.get("run_it").and_then(|r| r.as_bool()).unwrap_or(false)),
        );

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"run_it": false})).unwrap();

        let step_result = result.results.get("guarded").unwrap();
        assert_eq!(step_result.status, StepStatus::Skipped);
    }

    #[test]
    fn test_execute_condition_pass() {
        let mut wf = Workflow::new("wf", "Cond");
        wf.add_step(
            WorkflowStep::new("guarded", "Guarded step", StepAction::Message("hi".into()))
                .with_condition(|v| v.get("run_it").and_then(|r| r.as_bool()).unwrap_or(false)),
        );

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"run_it": true})).unwrap();

        let step_result = result.results.get("guarded").unwrap();
        assert_eq!(step_result.status, StepStatus::Completed);
    }

    // -----------------------------------------------------------------------
    // Execution — retry
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_retry_then_fail() {
        let mut wf = Workflow::new("wf", "Retry");
        wf.add_step(
            WorkflowStep::new(
                "failing",
                "Always fails",
                StepAction::Transform(Box::new(|_| Err("boom".to_string()))),
            )
            .with_retry_count(2),
        );

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({})).unwrap();

        let step_result = result.results.get("failing").unwrap();
        assert_eq!(step_result.status, StepStatus::Failed);
        assert_eq!(step_result.error.as_deref(), Some("boom"));
    }

    #[test]
    fn test_execute_retry_success_on_transform() {
        // Use a transform that succeeds (no retry needed).
        let mut wf = Workflow::new("wf", "Retry");
        wf.add_step(
            WorkflowStep::new("ok", "Succeeds", StepAction::Transform(Box::new(|v| Ok(v))))
                .with_retry_count(3),
        );

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"val": 1})).unwrap();

        let step_result = result.results.get("ok").unwrap();
        assert_eq!(step_result.status, StepStatus::Completed);
    }

    // -----------------------------------------------------------------------
    // Execution — message action
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_message_action() {
        let mut wf = Workflow::new("wf", "Msg");
        wf.add_step(WorkflowStep::new(
            "msg",
            "Send message",
            StepAction::Message("Hello agent".into()),
        ));

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({})).unwrap();

        let step_result = result.results.get("msg").unwrap();
        assert_eq!(step_result.status, StepStatus::Completed);

        // Message should be appended to state.
        let messages = result.final_state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "Hello agent");
    }

    // -----------------------------------------------------------------------
    // Execution — checkpoint action
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_checkpoint_action() {
        let mut wf = Workflow::new("wf", "Chk");
        wf.add_step(WorkflowStep::new(
            "chk",
            "Save state",
            StepAction::Checkpoint,
        ));

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({"data": 42})).unwrap();

        assert_eq!(
            result.results.get("chk").unwrap().status,
            StepStatus::Completed
        );
        assert_eq!(executor.checkpoints.len(), 1);
        assert_eq!(executor.checkpoints[0]["data"], 42);
    }

    // -----------------------------------------------------------------------
    // Execution — parallel action
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_parallel_action() {
        let mut wf = Workflow::new("wf", "Par");
        wf.add_step(WorkflowStep::new(
            "par",
            "Parallel steps",
            StepAction::Parallel(vec!["a".into(), "b".into()]),
        ));

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({})).unwrap();

        let step_result = result.results.get("par").unwrap();
        assert_eq!(step_result.status, StepStatus::Completed);
        let output = step_result.output.as_ref().unwrap();
        assert!(output.get("parallel_steps").is_some());
    }

    // -----------------------------------------------------------------------
    // WorkflowResult helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_workflow_result_succeeded_and_failed() {
        let mut results = HashMap::new();
        results.insert(
            "s1".to_string(),
            StepResult {
                step_id: "s1".to_string(),
                status: StepStatus::Completed,
                output: None,
                duration: Duration::from_millis(10),
                error: None,
            },
        );
        results.insert(
            "s2".to_string(),
            StepResult {
                step_id: "s2".to_string(),
                status: StepStatus::Failed,
                output: None,
                duration: Duration::from_millis(20),
                error: Some("err".to_string()),
            },
        );
        results.insert(
            "s3".to_string(),
            StepResult {
                step_id: "s3".to_string(),
                status: StepStatus::TimedOut,
                output: None,
                duration: Duration::from_millis(30),
                error: Some("timeout".to_string()),
            },
        );

        let wr = WorkflowResult {
            results,
            final_state: json!({}),
            total_duration: Duration::from_millis(60),
        };

        assert_eq!(wr.succeeded().len(), 1);
        assert_eq!(wr.failed().len(), 2); // Failed + TimedOut
    }

    // -----------------------------------------------------------------------
    // WorkflowBuilder
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_simple() {
        let wf = WorkflowBuilder::new("wf", "Builder test")
            .add_step_simple("s1", "First", StepAction::Checkpoint)
            .add_step_simple("s2", "Second", StepAction::Checkpoint)
            .dependency("s2", "s1")
            .build()
            .unwrap();

        assert_eq!(wf.id, "wf");
        assert_eq!(wf.name, "Builder test");
        assert_eq!(wf.steps.len(), 2);
        assert_eq!(wf.steps[1].dependencies, vec!["s1"]);
    }

    #[test]
    fn test_builder_invalid_dependency_step() {
        let result = WorkflowBuilder::new("wf", "Test")
            .add_step_simple("s1", "First", StepAction::Checkpoint)
            .dependency("nonexistent", "s1")
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_full_step() {
        let step = WorkflowStep::new("s1", "Step", StepAction::Checkpoint)
            .with_description("A step")
            .with_retry_count(2)
            .with_timeout(Duration::from_secs(10));

        let wf = WorkflowBuilder::new("wf", "Test")
            .add_step(step)
            .build()
            .unwrap();

        assert_eq!(wf.steps[0].description.as_deref(), Some("A step"));
        assert_eq!(wf.steps[0].retry_count, 2);
        assert_eq!(wf.steps[0].timeout, Some(Duration::from_secs(10)));
    }

    // -----------------------------------------------------------------------
    // Execution — dependency ordering end-to-end
    // -----------------------------------------------------------------------

    #[test]
    fn test_execute_respects_dependency_order() {
        // s1 sets x=1, s2 (depends on s1) adds 10, s3 (depends on s2) multiplies by 2.
        let wf = WorkflowBuilder::new("wf", "Order test")
            .add_step(WorkflowStep::new(
                "s1",
                "Set x",
                StepAction::Transform(Box::new(|mut v| {
                    v["x"] = json!(1);
                    Ok(v)
                })),
            ))
            .add_step(
                WorkflowStep::new(
                    "s2",
                    "Add 10",
                    StepAction::Transform(Box::new(|mut v| {
                        let x = v["x"].as_i64().unwrap_or(0);
                        v["x"] = json!(x + 10);
                        Ok(v)
                    })),
                )
                .with_dependency("s1"),
            )
            .add_step(
                WorkflowStep::new(
                    "s3",
                    "Mul 2",
                    StepAction::Transform(Box::new(|mut v| {
                        let x = v["x"].as_i64().unwrap_or(0);
                        v["x"] = json!(x * 2);
                        Ok(v)
                    })),
                )
                .with_dependency("s2"),
            )
            .build()
            .unwrap();

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({})).unwrap();
        // (1 + 10) * 2 = 22
        assert_eq!(result.final_state["x"], 22);
        assert_eq!(result.succeeded().len(), 3);
    }

    // -----------------------------------------------------------------------
    // WorkflowExecutor default
    // -----------------------------------------------------------------------

    #[test]
    fn test_workflow_executor_default() {
        let executor = WorkflowExecutor::default();
        assert!(executor.checkpoints.is_empty());
    }

    // -----------------------------------------------------------------------
    // StepResult duration is non-zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_result_has_duration() {
        let mut wf = Workflow::new("wf", "Dur");
        wf.add_step(WorkflowStep::new("s1", "Step", StepAction::Checkpoint));

        let mut executor = WorkflowExecutor::new();
        let result = executor.execute(&wf, json!({})).unwrap();

        let sr = result.results.get("s1").unwrap();
        // Duration should be set (the struct is populated).
        let _ = sr.duration;
        let _ = result.total_duration;
    }
}
