//! Planning capabilities for deep agents.
//!
//! This module provides structured task decomposition, plan execution tracking,
//! and adaptive replanning. A [`Plan`] is an ordered collection of [`PlanStep`]s
//! with dependency tracking, allowing both sequential and parallel execution
//! patterns.
//!
//! # Overview
//!
//! - [`PlanStep`] — a single actionable step with status and dependencies
//! - [`Plan`] — ordered collection of steps with dependency-aware readiness
//! - [`PlanBuilder`] — ergonomic builder for constructing plans
//! - [`PlanExecutor`] — drives execution and records [`ExecutionEvent`]s
//! - [`Replanner`] — modifies plans mid-execution (add, remove, insert steps)

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// StepStatus
// ---------------------------------------------------------------------------

/// Status of a single plan step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepStatus {
    /// Step has not started yet.
    Pending,
    /// Step is currently being executed.
    InProgress,
    /// Step completed successfully.
    Completed,
    /// Step failed with the given reason.
    Failed(String),
    /// Step was skipped with the given reason.
    Skipped(String),
}

impl StepStatus {
    /// Returns `true` if the step is in a terminal state (completed, failed, or skipped).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StepStatus::Completed | StepStatus::Failed(_) | StepStatus::Skipped(_)
        )
    }

    /// Returns `true` if the step completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, StepStatus::Completed)
    }
}

impl fmt::Display for StepStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepStatus::Pending => write!(f, "Pending"),
            StepStatus::InProgress => write!(f, "InProgress"),
            StepStatus::Completed => write!(f, "Completed"),
            StepStatus::Failed(reason) => write!(f, "Failed: {}", reason),
            StepStatus::Skipped(reason) => write!(f, "Skipped: {}", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// PlanStep
// ---------------------------------------------------------------------------

/// A single step within a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Unique identifier for this step.
    pub id: String,
    /// Human-readable description of what this step does.
    pub description: String,
    /// Current execution status.
    pub status: StepStatus,
    /// IDs of steps that must complete before this step can run.
    pub dependencies: Vec<String>,
    /// Optional estimated token cost for executing this step.
    pub estimated_tokens: Option<usize>,
    /// Result value produced by this step upon completion.
    pub result: Option<Value>,
    /// Arbitrary metadata attached to the step.
    pub metadata: HashMap<String, Value>,
}

impl PlanStep {
    /// Create a new step with the given description and a generated id.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            status: StepStatus::Pending,
            dependencies: Vec::new(),
            estimated_tokens: None,
            result: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a dependency on another step by id. Returns `self` for chaining.
    pub fn with_dependency(mut self, step_id: impl Into<String>) -> Self {
        self.dependencies.push(step_id.into());
        self
    }

    /// Transition the step to [`StepStatus::InProgress`].
    pub fn mark_in_progress(&mut self) {
        self.status = StepStatus::InProgress;
    }

    /// Transition the step to [`StepStatus::Completed`] with an optional result.
    pub fn mark_complete(&mut self, result: Option<Value>) {
        self.status = StepStatus::Completed;
        self.result = result;
    }

    /// Transition the step to [`StepStatus::Failed`] with the given reason.
    pub fn mark_failed(&mut self, reason: impl Into<String>) {
        self.status = StepStatus::Failed(reason.into());
    }

    /// Transition the step to [`StepStatus::Skipped`] with the given reason.
    pub fn mark_skipped(&mut self, reason: impl Into<String>) {
        self.status = StepStatus::Skipped(reason.into());
    }

    /// Returns `true` if all dependencies are present in `completed_ids` and the
    /// step is still [`StepStatus::Pending`].
    pub fn is_ready(&self, completed_ids: &HashSet<String>) -> bool {
        self.status == StepStatus::Pending
            && self
                .dependencies
                .iter()
                .all(|dep| completed_ids.contains(dep))
    }

    /// Serialize the step to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// PlanProgress
// ---------------------------------------------------------------------------

/// Snapshot of plan execution progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanProgress {
    /// Total number of steps in the plan.
    pub total: usize,
    /// Number of steps that completed successfully.
    pub completed: usize,
    /// Number of steps that failed.
    pub failed: usize,
    /// Number of steps that were skipped.
    pub skipped: usize,
    /// Number of steps currently in progress.
    pub in_progress: usize,
    /// Number of steps still pending.
    pub pending: usize,
}

impl PlanProgress {
    /// Fraction of steps that have reached a terminal state.
    pub fn completion_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.completed + self.failed + self.skipped) as f64 / self.total as f64
    }

    /// Fraction of terminal steps that completed successfully.
    pub fn success_rate(&self) -> f64 {
        let terminal = self.completed + self.failed + self.skipped;
        if terminal == 0 {
            return 0.0;
        }
        self.completed as f64 / terminal as f64
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// An ordered collection of [`PlanStep`]s with dependency tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Name or title of the plan.
    pub name: String,
    /// Ordered list of steps.
    steps: Vec<PlanStep>,
}

impl Plan {
    /// Create an empty plan with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step and return its id.
    pub fn add_step(&mut self, step: PlanStep) -> String {
        let id = step.id.clone();
        self.steps.push(step);
        id
    }

    /// Get a reference to a step by id.
    pub fn get_step(&self, id: &str) -> Option<&PlanStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Get a mutable reference to a step by id.
    pub fn get_step_mut(&mut self, id: &str) -> Option<&mut PlanStep> {
        self.steps.iter_mut().find(|s| s.id == id)
    }

    /// Return all steps whose dependencies are satisfied and status is Pending.
    pub fn ready_steps(&self) -> Vec<&PlanStep> {
        let completed: HashSet<String> = self
            .steps
            .iter()
            .filter(|s| s.status.is_success())
            .map(|s| s.id.clone())
            .collect();
        self.steps
            .iter()
            .filter(|s| s.is_ready(&completed))
            .collect()
    }

    /// Return the first ready step (by insertion order).
    pub fn next_step(&self) -> Option<&PlanStep> {
        self.ready_steps().into_iter().next()
    }

    /// Compute current progress across all steps.
    pub fn progress(&self) -> PlanProgress {
        let mut p = PlanProgress {
            total: self.steps.len(),
            completed: 0,
            failed: 0,
            skipped: 0,
            in_progress: 0,
            pending: 0,
        };
        for s in &self.steps {
            match &s.status {
                StepStatus::Pending => p.pending += 1,
                StepStatus::InProgress => p.in_progress += 1,
                StepStatus::Completed => p.completed += 1,
                StepStatus::Failed(_) => p.failed += 1,
                StepStatus::Skipped(_) => p.skipped += 1,
            }
        }
        p
    }

    /// Returns `true` when every step is in a terminal state.
    pub fn is_complete(&self) -> bool {
        self.steps.iter().all(|s| s.status.is_terminal())
    }

    /// Returns `true` if any step has failed.
    pub fn is_failed(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s.status, StepStatus::Failed(_)))
    }

    /// Iterate over all steps.
    pub fn steps(&self) -> &[PlanStep] {
        &self.steps
    }

    /// Number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the plan has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Serialize the plan to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Remove a step by id and return it. Also removes this id from all
    /// dependency lists of other steps.
    pub fn remove_step(&mut self, step_id: &str) -> Option<PlanStep> {
        let pos = self.steps.iter().position(|s| s.id == step_id)?;
        let removed = self.steps.remove(pos);
        // Clean up dependency references.
        for step in &mut self.steps {
            step.dependencies.retain(|d| d != step_id);
        }
        Some(removed)
    }

    /// Insert a step immediately after `after_id`. Returns the id of the
    /// inserted step. The inserted step automatically gains a dependency on
    /// `after_id`.
    pub fn insert_after(&mut self, after_id: &str, mut step: PlanStep) -> Option<String> {
        let pos = self.steps.iter().position(|s| s.id == after_id)?;
        if !step.dependencies.contains(&after_id.to_string()) {
            step.dependencies.push(after_id.to_string());
        }
        let id = step.id.clone();
        self.steps.insert(pos + 1, step);
        Some(id)
    }
}

// ---------------------------------------------------------------------------
// PlanBuilder
// ---------------------------------------------------------------------------

/// Builder for ergonomic plan construction.
#[derive(Debug)]
pub struct PlanBuilder {
    name: String,
    steps: Vec<PlanStep>,
}

impl PlanBuilder {
    /// Create a new builder with the given plan name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
        }
    }

    /// Add a step with no dependencies.
    pub fn step(mut self, description: impl Into<String>) -> Self {
        self.steps.push(PlanStep::new(description));
        self
    }

    /// Add a step that depends on the given step ids.
    pub fn step_with_deps(mut self, description: impl Into<String>, deps: Vec<String>) -> Self {
        let mut s = PlanStep::new(description);
        s.dependencies = deps;
        self.steps.push(s);
        self
    }

    /// Add a chain of steps where each depends on the previous one.
    pub fn sequential(mut self, descriptions: Vec<String>) -> Self {
        let mut prev_id: Option<String> = None;
        for desc in descriptions {
            let mut step = PlanStep::new(desc);
            if let Some(pid) = prev_id {
                step.dependencies.push(pid);
            }
            prev_id = Some(step.id.clone());
            self.steps.push(step);
        }
        self
    }

    /// Add a set of independent steps that can run in parallel.
    pub fn parallel(mut self, descriptions: Vec<String>) -> Self {
        for desc in descriptions {
            self.steps.push(PlanStep::new(desc));
        }
        self
    }

    /// Consume the builder and produce a [`Plan`].
    pub fn build(self) -> Plan {
        let mut plan = Plan::new(self.name);
        for step in self.steps {
            plan.add_step(step);
        }
        plan
    }
}

// ---------------------------------------------------------------------------
// EventType / ExecutionEvent
// ---------------------------------------------------------------------------

/// Type of execution event recorded in the log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    /// A step started executing.
    Started,
    /// A step completed successfully.
    Completed,
    /// A step failed.
    Failed,
    /// A step was skipped.
    Skipped,
    /// The plan was modified by a replanner.
    Replanned,
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventType::Started => write!(f, "Started"),
            EventType::Completed => write!(f, "Completed"),
            EventType::Failed => write!(f, "Failed"),
            EventType::Skipped => write!(f, "Skipped"),
            EventType::Replanned => write!(f, "Replanned"),
        }
    }
}

/// A log entry recording a plan execution event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    /// The step this event relates to.
    pub step_id: String,
    /// What happened.
    pub event_type: EventType,
    /// ISO-8601 timestamp (or simple counter-based string in non-std environments).
    pub timestamp: String,
    /// Optional extra details.
    pub details: Option<Value>,
}

impl ExecutionEvent {
    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// PlanExecutor
// ---------------------------------------------------------------------------

/// Drives execution of a [`Plan`], recording events along the way.
pub struct PlanExecutor {
    plan: Plan,
    events: Vec<ExecutionEvent>,
    step_counter: usize,
}

impl PlanExecutor {
    /// Create an executor for the given plan.
    pub fn new(plan: Plan) -> Self {
        Self {
            plan,
            events: Vec::new(),
            step_counter: 0,
        }
    }

    /// Start the next ready step. Returns the step id if one was started.
    pub fn start_next(&mut self) -> Option<String> {
        let id = self.plan.next_step().map(|s| s.id.clone())?;
        if let Some(step) = self.plan.get_step_mut(&id) {
            step.mark_in_progress();
        }
        self.step_counter += 1;
        self.events.push(ExecutionEvent {
            step_id: id.clone(),
            event_type: EventType::Started,
            timestamp: self.timestamp(),
            details: None,
        });
        Some(id)
    }

    /// Record successful completion of a step.
    pub fn complete_step(&mut self, id: &str, result: Option<Value>) {
        if let Some(step) = self.plan.get_step_mut(id) {
            step.mark_complete(result.clone());
        }
        self.events.push(ExecutionEvent {
            step_id: id.to_string(),
            event_type: EventType::Completed,
            timestamp: self.timestamp(),
            details: result,
        });
    }

    /// Record failure of a step.
    pub fn fail_step(&mut self, id: &str, reason: impl Into<String>) {
        let reason = reason.into();
        if let Some(step) = self.plan.get_step_mut(id) {
            step.mark_failed(&reason);
        }
        self.events.push(ExecutionEvent {
            step_id: id.to_string(),
            event_type: EventType::Failed,
            timestamp: self.timestamp(),
            details: Some(Value::String(reason)),
        });
    }

    /// Record that a step was skipped.
    pub fn skip_step(&mut self, id: &str, reason: impl Into<String>) {
        let reason = reason.into();
        if let Some(step) = self.plan.get_step_mut(id) {
            step.mark_skipped(&reason);
        }
        self.events.push(ExecutionEvent {
            step_id: id.to_string(),
            event_type: EventType::Skipped,
            timestamp: self.timestamp(),
            details: Some(Value::String(reason)),
        });
    }

    /// Reference to the underlying plan.
    pub fn current_plan(&self) -> &Plan {
        &self.plan
    }

    /// Mutable reference to the underlying plan (used by [`Replanner`]).
    pub fn current_plan_mut(&mut self) -> &mut Plan {
        &mut self.plan
    }

    /// Full execution event log.
    pub fn execution_log(&self) -> &[ExecutionEvent] {
        &self.events
    }

    /// Number of steps that have been started.
    pub fn elapsed_steps(&self) -> usize {
        self.step_counter
    }

    /// Push a replanned event.
    #[allow(dead_code)]
    fn record_replan(&mut self, details: Option<Value>) {
        self.events.push(ExecutionEvent {
            step_id: String::new(),
            event_type: EventType::Replanned,
            timestamp: self.timestamp(),
            details,
        });
    }

    /// Simple monotonic timestamp based on the event count.
    fn timestamp(&self) -> String {
        format!("T{}", self.events.len())
    }
}

// ---------------------------------------------------------------------------
// Replanner
// ---------------------------------------------------------------------------

/// Modifies an in-flight plan by adding, removing, or inserting steps.
#[derive(Debug)]
pub struct Replanner {
    replan_count: usize,
}

impl Replanner {
    /// Create a new replanner.
    pub fn new() -> Self {
        Self { replan_count: 0 }
    }

    /// Append additional steps to an existing plan.
    pub fn add_steps(&mut self, plan: &mut Plan, steps: Vec<PlanStep>) {
        for step in steps {
            plan.add_step(step);
        }
        self.replan_count += 1;
    }

    /// Remove a step from the plan by id, returning it if found.
    pub fn remove_step(&mut self, plan: &mut Plan, step_id: &str) -> Option<PlanStep> {
        let removed = plan.remove_step(step_id);
        if removed.is_some() {
            self.replan_count += 1;
        }
        removed
    }

    /// Insert a step immediately after `after_id`, adding a dependency. Returns
    /// the new step's id.
    pub fn insert_after(&mut self, plan: &mut Plan, after_id: &str, step: PlanStep) -> String {
        let id = plan.insert_after(after_id, step).unwrap_or_default();
        if !id.is_empty() {
            self.replan_count += 1;
        }
        id
    }

    /// Number of replan operations performed.
    pub fn replan_count(&self) -> usize {
        self.replan_count
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({ "replan_count": self.replan_count })
    }
}

impl Default for Replanner {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- StepStatus -------------------------------------------------------

    #[test]
    fn test_step_status_pending_is_not_terminal() {
        assert!(!StepStatus::Pending.is_terminal());
    }

    #[test]
    fn test_step_status_in_progress_is_not_terminal() {
        assert!(!StepStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_step_status_completed_is_terminal() {
        assert!(StepStatus::Completed.is_terminal());
    }

    #[test]
    fn test_step_status_failed_is_terminal() {
        assert!(StepStatus::Failed("err".into()).is_terminal());
    }

    #[test]
    fn test_step_status_skipped_is_terminal() {
        assert!(StepStatus::Skipped("reason".into()).is_terminal());
    }

    #[test]
    fn test_step_status_completed_is_success() {
        assert!(StepStatus::Completed.is_success());
    }

    #[test]
    fn test_step_status_failed_is_not_success() {
        assert!(!StepStatus::Failed("err".into()).is_success());
    }

    #[test]
    fn test_step_status_display_pending() {
        assert_eq!(StepStatus::Pending.to_string(), "Pending");
    }

    #[test]
    fn test_step_status_display_in_progress() {
        assert_eq!(StepStatus::InProgress.to_string(), "InProgress");
    }

    #[test]
    fn test_step_status_display_completed() {
        assert_eq!(StepStatus::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_step_status_display_failed() {
        assert_eq!(
            StepStatus::Failed("oops".into()).to_string(),
            "Failed: oops"
        );
    }

    #[test]
    fn test_step_status_display_skipped() {
        assert_eq!(StepStatus::Skipped("na".into()).to_string(), "Skipped: na");
    }

    // ---- PlanStep ---------------------------------------------------------

    #[test]
    fn test_plan_step_new_has_pending_status() {
        let s = PlanStep::new("do something");
        assert_eq!(s.description, "do something");
        assert_eq!(s.status, StepStatus::Pending);
        assert!(s.dependencies.is_empty());
        assert!(s.result.is_none());
    }

    #[test]
    fn test_plan_step_with_dependency() {
        let s = PlanStep::new("step").with_dependency("dep-1");
        assert_eq!(s.dependencies, vec!["dep-1".to_string()]);
    }

    #[test]
    fn test_plan_step_chained_dependencies() {
        let s = PlanStep::new("step")
            .with_dependency("a")
            .with_dependency("b");
        assert_eq!(s.dependencies.len(), 2);
    }

    #[test]
    fn test_plan_step_mark_in_progress() {
        let mut s = PlanStep::new("s");
        s.mark_in_progress();
        assert_eq!(s.status, StepStatus::InProgress);
    }

    #[test]
    fn test_plan_step_mark_complete_with_result() {
        let mut s = PlanStep::new("s");
        s.mark_complete(Some(json!(42)));
        assert_eq!(s.status, StepStatus::Completed);
        assert_eq!(s.result, Some(json!(42)));
    }

    #[test]
    fn test_plan_step_mark_complete_without_result() {
        let mut s = PlanStep::new("s");
        s.mark_complete(None);
        assert!(s.status.is_success());
        assert!(s.result.is_none());
    }

    #[test]
    fn test_plan_step_mark_failed() {
        let mut s = PlanStep::new("s");
        s.mark_failed("boom");
        assert_eq!(s.status, StepStatus::Failed("boom".into()));
    }

    #[test]
    fn test_plan_step_mark_skipped() {
        let mut s = PlanStep::new("s");
        s.mark_skipped("not needed");
        assert_eq!(s.status, StepStatus::Skipped("not needed".into()));
    }

    #[test]
    fn test_plan_step_is_ready_no_deps() {
        let s = PlanStep::new("s");
        assert!(s.is_ready(&HashSet::new()));
    }

    #[test]
    fn test_plan_step_is_ready_deps_met() {
        let s = PlanStep::new("s").with_dependency("a");
        let mut completed = HashSet::new();
        completed.insert("a".to_string());
        assert!(s.is_ready(&completed));
    }

    #[test]
    fn test_plan_step_not_ready_deps_unmet() {
        let s = PlanStep::new("s").with_dependency("a");
        assert!(!s.is_ready(&HashSet::new()));
    }

    #[test]
    fn test_plan_step_not_ready_if_in_progress() {
        let mut s = PlanStep::new("s");
        s.mark_in_progress();
        assert!(!s.is_ready(&HashSet::new()));
    }

    #[test]
    fn test_plan_step_to_json() {
        let s = PlanStep::new("x");
        let v = s.to_json();
        assert_eq!(v["description"], "x");
    }

    #[test]
    fn test_plan_step_metadata() {
        let mut s = PlanStep::new("s");
        s.metadata.insert("key".into(), json!("value"));
        assert_eq!(s.metadata["key"], json!("value"));
    }

    #[test]
    fn test_plan_step_estimated_tokens() {
        let mut s = PlanStep::new("s");
        s.estimated_tokens = Some(500);
        assert_eq!(s.estimated_tokens, Some(500));
    }

    // ---- Plan -------------------------------------------------------------

    #[test]
    fn test_plan_new_is_empty() {
        let p = Plan::new("test plan");
        assert_eq!(p.name, "test plan");
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn test_plan_add_and_get_step() {
        let mut p = Plan::new("p");
        let s = PlanStep::new("a");
        let id = p.add_step(s);
        assert_eq!(p.len(), 1);
        assert!(p.get_step(&id).is_some());
    }

    #[test]
    fn test_plan_get_step_mut() {
        let mut p = Plan::new("p");
        let s = PlanStep::new("a");
        let id = p.add_step(s);
        p.get_step_mut(&id).unwrap().mark_in_progress();
        assert_eq!(p.get_step(&id).unwrap().status, StepStatus::InProgress);
    }

    #[test]
    fn test_plan_get_step_nonexistent() {
        let p = Plan::new("p");
        assert!(p.get_step("nope").is_none());
    }

    #[test]
    fn test_plan_ready_steps_all_independent() {
        let mut p = Plan::new("p");
        p.add_step(PlanStep::new("a"));
        p.add_step(PlanStep::new("b"));
        assert_eq!(p.ready_steps().len(), 2);
    }

    #[test]
    fn test_plan_ready_steps_with_dependency() {
        let mut p = Plan::new("p");
        let id_a = p.add_step(PlanStep::new("a"));
        p.add_step(PlanStep::new("b").with_dependency(id_a));
        // Only "a" is ready.
        assert_eq!(p.ready_steps().len(), 1);
        assert_eq!(p.ready_steps()[0].description, "a");
    }

    #[test]
    fn test_plan_next_step() {
        let mut p = Plan::new("p");
        p.add_step(PlanStep::new("first"));
        let ns = p.next_step().unwrap();
        assert_eq!(ns.description, "first");
    }

    #[test]
    fn test_plan_next_step_empty() {
        let p = Plan::new("p");
        assert!(p.next_step().is_none());
    }

    #[test]
    fn test_plan_progress() {
        let mut p = Plan::new("p");
        let id = p.add_step(PlanStep::new("a"));
        p.add_step(PlanStep::new("b"));
        p.get_step_mut(&id).unwrap().mark_complete(None);
        let prog = p.progress();
        assert_eq!(prog.total, 2);
        assert_eq!(prog.completed, 1);
        assert_eq!(prog.pending, 1);
    }

    #[test]
    fn test_plan_is_complete() {
        let mut p = Plan::new("p");
        let id = p.add_step(PlanStep::new("a"));
        assert!(!p.is_complete());
        p.get_step_mut(&id).unwrap().mark_complete(None);
        assert!(p.is_complete());
    }

    #[test]
    fn test_plan_is_failed() {
        let mut p = Plan::new("p");
        let id = p.add_step(PlanStep::new("a"));
        assert!(!p.is_failed());
        p.get_step_mut(&id).unwrap().mark_failed("err");
        assert!(p.is_failed());
    }

    #[test]
    fn test_plan_steps_iterator() {
        let mut p = Plan::new("p");
        p.add_step(PlanStep::new("a"));
        p.add_step(PlanStep::new("b"));
        assert_eq!(p.steps().len(), 2);
    }

    #[test]
    fn test_plan_to_json() {
        let p = Plan::new("my plan");
        let v = p.to_json();
        assert_eq!(v["name"], "my plan");
    }

    #[test]
    fn test_plan_remove_step() {
        let mut p = Plan::new("p");
        let id_a = p.add_step(PlanStep::new("a"));
        let _id_b = p.add_step(PlanStep::new("b").with_dependency(id_a.clone()));
        let removed = p.remove_step(&id_a);
        assert!(removed.is_some());
        assert_eq!(p.len(), 1);
        // Dependency on removed step should be cleaned up.
        assert!(p.steps()[0].dependencies.is_empty());
    }

    #[test]
    fn test_plan_remove_step_nonexistent() {
        let mut p = Plan::new("p");
        assert!(p.remove_step("nope").is_none());
    }

    #[test]
    fn test_plan_insert_after() {
        let mut p = Plan::new("p");
        let id_a = p.add_step(PlanStep::new("a"));
        p.add_step(PlanStep::new("c"));
        let new_step = PlanStep::new("b");
        let id_b = p.insert_after(&id_a, new_step).unwrap();
        assert_eq!(p.len(), 3);
        // "b" should be at index 1.
        assert_eq!(p.steps()[1].id, id_b);
        assert!(p.steps()[1].dependencies.contains(&id_a));
    }

    #[test]
    fn test_plan_insert_after_nonexistent() {
        let mut p = Plan::new("p");
        let result = p.insert_after("nope", PlanStep::new("x"));
        assert!(result.is_none());
    }

    // ---- PlanProgress -----------------------------------------------------

    #[test]
    fn test_plan_progress_completion_rate() {
        let prog = PlanProgress {
            total: 4,
            completed: 2,
            failed: 1,
            skipped: 0,
            in_progress: 0,
            pending: 1,
        };
        assert!((prog.completion_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress_completion_rate_empty() {
        let prog = PlanProgress {
            total: 0,
            completed: 0,
            failed: 0,
            skipped: 0,
            in_progress: 0,
            pending: 0,
        };
        assert!((prog.completion_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress_success_rate() {
        let prog = PlanProgress {
            total: 4,
            completed: 2,
            failed: 1,
            skipped: 1,
            in_progress: 0,
            pending: 0,
        };
        assert!((prog.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress_success_rate_no_terminal() {
        let prog = PlanProgress {
            total: 2,
            completed: 0,
            failed: 0,
            skipped: 0,
            in_progress: 2,
            pending: 0,
        };
        assert!((prog.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_progress_to_json() {
        let prog = PlanProgress {
            total: 1,
            completed: 1,
            failed: 0,
            skipped: 0,
            in_progress: 0,
            pending: 0,
        };
        let v = prog.to_json();
        assert_eq!(v["total"], 1);
        assert_eq!(v["completed"], 1);
    }

    // ---- PlanBuilder ------------------------------------------------------

    #[test]
    fn test_builder_step() {
        let plan = PlanBuilder::new("bp").step("a").step("b").build();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.name, "bp");
    }

    #[test]
    fn test_builder_step_with_deps() {
        // Build a plan with "a", then add "b" that depends on "a".
        let plan = PlanBuilder::new("bp").step("a").build();
        let dep_id = plan.steps()[0].id.clone();
        let plan = PlanBuilder::new("bp")
            .step_with_deps("b", vec![dep_id.clone()])
            .build();
        assert_eq!(plan.steps()[0].dependencies, vec![dep_id]);
    }

    #[test]
    fn test_builder_sequential() {
        let plan = PlanBuilder::new("seq")
            .sequential(vec!["a".into(), "b".into(), "c".into()])
            .build();
        assert_eq!(plan.len(), 3);
        // Second step depends on first, third depends on second.
        assert!(plan.steps()[0].dependencies.is_empty());
        assert_eq!(plan.steps()[1].dependencies.len(), 1);
        assert_eq!(plan.steps()[2].dependencies.len(), 1);
        assert_eq!(plan.steps()[1].dependencies[0], plan.steps()[0].id);
        assert_eq!(plan.steps()[2].dependencies[0], plan.steps()[1].id);
    }

    #[test]
    fn test_builder_parallel() {
        let plan = PlanBuilder::new("par")
            .parallel(vec!["a".into(), "b".into(), "c".into()])
            .build();
        assert_eq!(plan.len(), 3);
        for s in plan.steps() {
            assert!(s.dependencies.is_empty());
        }
    }

    #[test]
    fn test_builder_mixed() {
        let plan = PlanBuilder::new("mix")
            .step("independent")
            .sequential(vec!["s1".into(), "s2".into()])
            .parallel(vec!["p1".into(), "p2".into()])
            .build();
        assert_eq!(plan.len(), 5);
    }

    #[test]
    fn test_builder_empty() {
        let plan = PlanBuilder::new("empty").build();
        assert!(plan.is_empty());
    }

    // ---- EventType --------------------------------------------------------

    #[test]
    fn test_event_type_display() {
        assert_eq!(EventType::Started.to_string(), "Started");
        assert_eq!(EventType::Completed.to_string(), "Completed");
        assert_eq!(EventType::Failed.to_string(), "Failed");
        assert_eq!(EventType::Skipped.to_string(), "Skipped");
        assert_eq!(EventType::Replanned.to_string(), "Replanned");
    }

    // ---- ExecutionEvent ---------------------------------------------------

    #[test]
    fn test_execution_event_to_json() {
        let ev = ExecutionEvent {
            step_id: "s1".into(),
            event_type: EventType::Started,
            timestamp: "T0".into(),
            details: None,
        };
        let v = ev.to_json();
        assert_eq!(v["step_id"], "s1");
        assert_eq!(v["event_type"], "Started");
    }

    // ---- PlanExecutor -----------------------------------------------------

    #[test]
    fn test_executor_start_next() {
        let plan = PlanBuilder::new("e").step("a").step("b").build();
        let mut exec = PlanExecutor::new(plan);
        let id = exec.start_next().unwrap();
        assert_eq!(
            exec.current_plan().get_step(&id).unwrap().status,
            StepStatus::InProgress
        );
        assert_eq!(exec.elapsed_steps(), 1);
    }

    #[test]
    fn test_executor_complete_step() {
        let plan = PlanBuilder::new("e").step("a").build();
        let mut exec = PlanExecutor::new(plan);
        let id = exec.start_next().unwrap();
        exec.complete_step(&id, Some(json!("done")));
        assert!(exec
            .current_plan()
            .get_step(&id)
            .unwrap()
            .status
            .is_success());
    }

    #[test]
    fn test_executor_fail_step() {
        let plan = PlanBuilder::new("e").step("a").build();
        let mut exec = PlanExecutor::new(plan);
        let id = exec.start_next().unwrap();
        exec.fail_step(&id, "oops");
        assert!(exec
            .current_plan()
            .get_step(&id)
            .unwrap()
            .status
            .is_terminal());
        assert!(exec.current_plan().is_failed());
    }

    #[test]
    fn test_executor_skip_step() {
        let plan = PlanBuilder::new("e").step("a").build();
        let mut exec = PlanExecutor::new(plan);
        let id = exec.start_next().unwrap();
        exec.skip_step(&id, "not needed");
        let status = &exec.current_plan().get_step(&id).unwrap().status;
        assert!(matches!(status, StepStatus::Skipped(_)));
    }

    #[test]
    fn test_executor_execution_log() {
        let plan = PlanBuilder::new("e").step("a").build();
        let mut exec = PlanExecutor::new(plan);
        let id = exec.start_next().unwrap();
        exec.complete_step(&id, None);
        assert_eq!(exec.execution_log().len(), 2); // Started + Completed
    }

    #[test]
    fn test_executor_sequential_flow() {
        let plan = PlanBuilder::new("e")
            .sequential(vec!["a".into(), "b".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);
        // Start and complete first step.
        let id_a = exec.start_next().unwrap();
        exec.complete_step(&id_a, None);
        // Now second step should be ready.
        let id_b = exec.start_next().unwrap();
        assert_ne!(id_a, id_b);
        exec.complete_step(&id_b, None);
        assert!(exec.current_plan().is_complete());
    }

    #[test]
    fn test_executor_no_ready_step() {
        let plan = PlanBuilder::new("e")
            .sequential(vec!["a".into(), "b".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);
        let id_a = exec.start_next().unwrap();
        // "a" is in progress, "b" depends on "a" — no ready steps.
        assert!(exec.start_next().is_none());
        exec.complete_step(&id_a, None);
        // Now "b" is ready.
        assert!(exec.start_next().is_some());
    }

    #[test]
    fn test_executor_empty_plan() {
        let plan = Plan::new("empty");
        let mut exec = PlanExecutor::new(plan);
        assert!(exec.start_next().is_none());
        assert_eq!(exec.elapsed_steps(), 0);
    }

    #[test]
    fn test_executor_elapsed_steps() {
        let plan = PlanBuilder::new("e")
            .parallel(vec!["a".into(), "b".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);
        let id1 = exec.start_next().unwrap();
        exec.complete_step(&id1, None);
        let id2 = exec.start_next().unwrap();
        exec.complete_step(&id2, None);
        assert_eq!(exec.elapsed_steps(), 2);
    }

    // ---- Replanner --------------------------------------------------------

    #[test]
    fn test_replanner_add_steps() {
        let mut plan = PlanBuilder::new("p").step("a").build();
        let mut rp = Replanner::new();
        rp.add_steps(&mut plan, vec![PlanStep::new("b"), PlanStep::new("c")]);
        assert_eq!(plan.len(), 3);
        assert_eq!(rp.replan_count(), 1);
    }

    #[test]
    fn test_replanner_remove_step() {
        let mut plan = PlanBuilder::new("p").step("a").step("b").build();
        let id = plan.steps()[0].id.clone();
        let mut rp = Replanner::new();
        let removed = rp.remove_step(&mut plan, &id);
        assert!(removed.is_some());
        assert_eq!(plan.len(), 1);
        assert_eq!(rp.replan_count(), 1);
    }

    #[test]
    fn test_replanner_remove_nonexistent() {
        let mut plan = Plan::new("p");
        let mut rp = Replanner::new();
        assert!(rp.remove_step(&mut plan, "nope").is_none());
        assert_eq!(rp.replan_count(), 0);
    }

    #[test]
    fn test_replanner_insert_after() {
        let mut plan = PlanBuilder::new("p").step("a").step("c").build();
        let id_a = plan.steps()[0].id.clone();
        let mut rp = Replanner::new();
        let id_b = rp.insert_after(&mut plan, &id_a, PlanStep::new("b"));
        assert!(!id_b.is_empty());
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.steps()[1].description, "b");
        assert_eq!(rp.replan_count(), 1);
    }

    #[test]
    fn test_replanner_to_json() {
        let rp = Replanner::new();
        let v = rp.to_json();
        assert_eq!(v["replan_count"], 0);
    }

    #[test]
    fn test_replanner_default() {
        let rp = Replanner::default();
        assert_eq!(rp.replan_count(), 0);
    }

    #[test]
    fn test_replanner_multiple_operations() {
        let mut plan = PlanBuilder::new("p").step("a").build();
        let mut rp = Replanner::new();
        rp.add_steps(&mut plan, vec![PlanStep::new("b")]);
        rp.add_steps(&mut plan, vec![PlanStep::new("c")]);
        let id = plan.steps()[1].id.clone();
        rp.remove_step(&mut plan, &id);
        assert_eq!(rp.replan_count(), 3);
    }

    // ---- Integration / end-to-end ----------------------------------------

    #[test]
    fn test_full_plan_execution() {
        let plan = PlanBuilder::new("deploy")
            .sequential(vec!["build".into(), "test".into(), "deploy".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);

        for _ in 0..3 {
            let id = exec.start_next().unwrap();
            exec.complete_step(&id, Some(json!("ok")));
        }

        assert!(exec.current_plan().is_complete());
        assert!(!exec.current_plan().is_failed());
        let prog = exec.current_plan().progress();
        assert_eq!(prog.completed, 3);
        assert!((prog.completion_rate() - 1.0).abs() < f64::EPSILON);
        assert!((prog.success_rate() - 1.0).abs() < f64::EPSILON);
        assert_eq!(exec.execution_log().len(), 6); // 3 started + 3 completed
    }

    #[test]
    fn test_plan_with_failure_and_replan() {
        let plan = PlanBuilder::new("p")
            .sequential(vec!["a".into(), "b".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);

        // Start and fail first step.
        let id_a = exec.start_next().unwrap();
        exec.fail_step(&id_a, "timeout");

        // Replan: add a retry step.
        let mut rp = Replanner::new();
        let retry = PlanStep::new("retry-a");
        rp.add_steps(exec.current_plan_mut(), vec![retry]);

        // The second original step still depends on "a" which failed, so it
        // won't be ready. But the retry step (no deps) is ready.
        let id_retry = exec.start_next().unwrap();
        exec.complete_step(&id_retry, None);

        assert!(exec.current_plan().is_failed()); // original "a" still failed
        assert_eq!(rp.replan_count(), 1);
    }

    #[test]
    fn test_parallel_execution_pattern() {
        let plan = PlanBuilder::new("p")
            .parallel(vec!["fetch-a".into(), "fetch-b".into(), "fetch-c".into()])
            .build();
        let mut exec = PlanExecutor::new(plan);

        // All three should be startable (we start them one by one here).
        let mut ids = Vec::new();
        for _ in 0..3 {
            if let Some(id) = exec.start_next() {
                ids.push(id);
            }
        }
        assert_eq!(ids.len(), 3);
        for id in &ids {
            exec.complete_step(id, None);
        }
        assert!(exec.current_plan().is_complete());
    }

    #[test]
    fn test_diamond_dependency() {
        // A -> B, A -> C, B+C -> D
        let mut plan = Plan::new("diamond");
        let step_a = PlanStep::new("A");
        let id_a = step_a.id.clone();
        plan.add_step(step_a);
        let step_b = PlanStep::new("B").with_dependency(id_a.clone());
        let id_b = step_b.id.clone();
        plan.add_step(step_b);
        let step_c = PlanStep::new("C").with_dependency(id_a.clone());
        let id_c = step_c.id.clone();
        plan.add_step(step_c);
        plan.add_step(
            PlanStep::new("D")
                .with_dependency(id_b.clone())
                .with_dependency(id_c.clone()),
        );

        let mut exec = PlanExecutor::new(plan);

        // Only A is ready.
        let a = exec.start_next().unwrap();
        assert!(exec.start_next().is_none());
        exec.complete_step(&a, None);

        // Now B and C are ready.
        let b = exec.start_next().unwrap();
        let c = exec.start_next().unwrap();
        assert!(exec.start_next().is_none()); // D not ready yet
        exec.complete_step(&b, None);
        exec.complete_step(&c, None);

        // Now D is ready.
        let d = exec.start_next().unwrap();
        exec.complete_step(&d, None);
        assert!(exec.current_plan().is_complete());
    }

    #[test]
    fn test_step_status_serialization_roundtrip() {
        let statuses = vec![
            StepStatus::Pending,
            StepStatus::InProgress,
            StepStatus::Completed,
            StepStatus::Failed("err".into()),
            StepStatus::Skipped("reason".into()),
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let deserialized: StepStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, deserialized);
        }
    }

    #[test]
    fn test_plan_serialization_roundtrip() {
        let plan = PlanBuilder::new("test")
            .sequential(vec!["a".into(), "b".into()])
            .build();
        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: Plan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.len(), 2);
    }
}
