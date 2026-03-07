//! Planning middleware — plan-then-execute support for Deep Agents.
//!
//! Provides structured plan creation, step tracking with dependencies,
//! and middleware hooks that inject plan status into the agent context.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::middleware::{AgentState, Middleware, Result};

// ---------------------------------------------------------------------------
// PlanStepStatus
// ---------------------------------------------------------------------------

/// Status of a single plan step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    /// Step has not started yet.
    Pending,
    /// Step is currently being worked on.
    InProgress,
    /// Step finished successfully.
    Completed,
    /// Step failed.
    Failed,
    /// Step was skipped (e.g. no longer relevant).
    Skipped,
}

impl std::fmt::Display for PlanStepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

// ---------------------------------------------------------------------------
// PlanStep
// ---------------------------------------------------------------------------

/// A single step within a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Unique identifier within the plan.
    pub id: usize,
    /// Human-readable description of what this step accomplishes.
    pub description: String,
    /// Current status of this step.
    pub status: PlanStepStatus,
    /// IDs of steps that must be completed before this step can start.
    pub dependencies: Vec<usize>,
    /// The result or output of this step once completed.
    pub result: Option<String>,
    /// When this step was created.
    #[serde(skip)]
    pub created_at: Option<Instant>,
    /// When this step was completed.
    #[serde(skip)]
    pub completed_at: Option<Instant>,
}

impl PlanStep {
    /// Create a new plan step with the given id and description.
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            status: PlanStepStatus::Pending,
            dependencies: Vec::new(),
            result: None,
            created_at: Some(Instant::now()),
            completed_at: None,
        }
    }

    /// Add a dependency on another step.
    pub fn with_dependency(mut self, dep_id: usize) -> Self {
        self.dependencies.push(dep_id);
        self
    }

    /// Add multiple dependencies.
    pub fn with_dependencies(mut self, deps: Vec<usize>) -> Self {
        self.dependencies.extend(deps);
        self
    }

    /// Whether this step has finished (completed, failed, or skipped).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            PlanStepStatus::Completed | PlanStepStatus::Failed | PlanStepStatus::Skipped
        )
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A structured plan consisting of ordered steps with dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Unique identifier for this plan.
    pub id: String,
    /// The high-level goal this plan aims to achieve.
    pub goal: String,
    /// The ordered list of steps.
    pub steps: Vec<PlanStep>,
    /// When this plan was created.
    #[serde(skip)]
    pub created_at: Option<Instant>,
}

impl Plan {
    /// Create a new empty plan for the given goal.
    pub fn new(id: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            goal: goal.into(),
            steps: Vec::new(),
            created_at: Some(Instant::now()),
        }
    }

    /// Add a step to the plan. Returns the id assigned to the step.
    pub fn add_step(&mut self, mut step: PlanStep) -> usize {
        let id = if self.steps.is_empty() {
            0
        } else {
            self.steps.iter().map(|s| s.id).max().unwrap_or(0) + 1
        };
        step.id = id;
        self.steps.push(step);
        id
    }

    /// Get a reference to a step by id.
    pub fn get_step(&self, id: usize) -> Option<&PlanStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Get a mutable reference to a step by id.
    pub fn get_step_mut(&mut self, id: usize) -> Option<&mut PlanStep> {
        self.steps.iter_mut().find(|s| s.id == id)
    }

    /// Update the status of a step. Returns `true` if the step was found.
    pub fn update_step_status(
        &mut self,
        id: usize,
        status: PlanStepStatus,
        result: Option<String>,
    ) -> bool {
        if let Some(step) = self.get_step_mut(id) {
            step.status = status.clone();
            if result.is_some() {
                step.result = result;
            }
            if step.is_terminal() {
                step.completed_at = Some(Instant::now());
            }
            true
        } else {
            false
        }
    }

    /// Return the next step that is pending and whose dependencies are all completed.
    pub fn next_actionable_step(&self) -> Option<&PlanStep> {
        self.get_ready_steps().into_iter().next()
    }

    /// Return all steps whose dependencies are fully met and that are still pending.
    pub fn get_ready_steps(&self) -> Vec<&PlanStep> {
        self.steps
            .iter()
            .filter(|step| {
                step.status == PlanStepStatus::Pending
                    && step.dependencies.iter().all(|dep_id| {
                        self.get_step(*dep_id)
                            .map(|d| d.status == PlanStepStatus::Completed)
                            .unwrap_or(false)
                    })
            })
            .collect()
    }

    /// Whether all steps are in a terminal state.
    pub fn is_complete(&self) -> bool {
        !self.steps.is_empty() && self.steps.iter().all(|s| s.is_terminal())
    }

    /// Percentage of steps that are in a terminal state (0.0 to 100.0).
    pub fn progress_percentage(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let done = self.steps.iter().filter(|s| s.is_terminal()).count();
        (done as f64 / self.steps.len() as f64) * 100.0
    }

    /// Produce a human-readable summary of the plan status.
    pub fn status_summary(&self) -> String {
        let mut lines = vec![format!("Plan: {} (goal: {})", self.id, self.goal)];
        lines.push(format!(
            "Progress: {:.0}% ({}/{})",
            self.progress_percentage(),
            self.steps.iter().filter(|s| s.is_terminal()).count(),
            self.steps.len()
        ));
        for step in &self.steps {
            let deps = if step.dependencies.is_empty() {
                String::new()
            } else {
                format!(" [deps: {:?}]", step.dependencies)
            };
            lines.push(format!(
                "  [{}] Step {}: {}{}",
                step.status, step.id, step.description, deps
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// PlanningStrategy trait
// ---------------------------------------------------------------------------

/// Strategy for creating and revising plans.
#[async_trait]
pub trait PlanningStrategy: Send + Sync {
    /// Create a plan to achieve the given goal, with optional context.
    async fn create_plan(&self, goal: &str, context: &Value) -> Result<Plan>;

    /// Revise an existing plan based on feedback.
    async fn revise_plan(&self, plan: &mut Plan, feedback: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// SimplePlanningStrategy
// ---------------------------------------------------------------------------

/// A rule-based planning strategy that creates plans from structured goal descriptions.
///
/// The strategy parses goals that contain numbered steps (lines starting with
/// a digit or "- ") and creates a sequential plan from them. If no structure
/// is detected, a single-step plan is created.
pub struct SimplePlanningStrategy;

impl SimplePlanningStrategy {
    /// Create a new `SimplePlanningStrategy`.
    pub fn new() -> Self {
        Self
    }

    /// Parse structured steps from a goal string.
    fn parse_steps(goal: &str) -> Vec<String> {
        let mut steps = Vec::new();
        for line in goal.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Match lines that start with a number+period, number+paren, or dash/bullet.
            let is_step = trimmed
                .chars()
                .next()
                .map(|c| c.is_ascii_digit() || c == '-' || c == '*')
                .unwrap_or(false);
            if is_step {
                // Strip the prefix marker.
                let content = trimmed
                    .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '-' || c == '*')
                    .trim();
                if !content.is_empty() {
                    steps.push(content.to_string());
                }
            }
        }
        steps
    }
}

impl Default for SimplePlanningStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlanningStrategy for SimplePlanningStrategy {
    async fn create_plan(&self, goal: &str, _context: &Value) -> Result<Plan> {
        let parsed_steps = Self::parse_steps(goal);

        let mut plan = Plan::new(uuid::Uuid::new_v4().to_string(), goal);

        if parsed_steps.is_empty() {
            // No structure found — create a single-step plan.
            plan.add_step(PlanStep::new(0, goal));
        } else {
            let mut prev_id: Option<usize> = None;
            for desc in parsed_steps {
                let mut step = PlanStep::new(0, desc);
                if let Some(pid) = prev_id {
                    step = step.with_dependency(pid);
                }
                let id = plan.add_step(step);
                prev_id = Some(id);
            }
        }

        Ok(plan)
    }

    async fn revise_plan(&self, plan: &mut Plan, feedback: &str) -> Result<()> {
        // Simple strategy: mark failed steps as skipped and add new steps from feedback.
        for step in &mut plan.steps {
            if step.status == PlanStepStatus::Failed {
                step.status = PlanStepStatus::Skipped;
                step.completed_at = Some(Instant::now());
            }
        }

        let new_steps = Self::parse_steps(feedback);
        if new_steps.is_empty() {
            // Add feedback as a single new step.
            plan.add_step(PlanStep::new(0, feedback));
        } else {
            for desc in new_steps {
                plan.add_step(PlanStep::new(0, desc));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ReplanningStrategy
// ---------------------------------------------------------------------------

/// A strategy wrapper that can revise plans when steps fail.
///
/// It delegates initial plan creation to an inner strategy and adds
/// replanning logic on failure.
pub struct ReplanningStrategy {
    inner: Arc<dyn PlanningStrategy>,
    max_retries: usize,
}

impl ReplanningStrategy {
    /// Create a new `ReplanningStrategy` wrapping the given inner strategy.
    pub fn new(inner: Arc<dyn PlanningStrategy>, max_retries: usize) -> Self {
        Self { inner, max_retries }
    }

    /// Maximum number of retries allowed per step.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }
}

#[async_trait]
impl PlanningStrategy for ReplanningStrategy {
    async fn create_plan(&self, goal: &str, context: &Value) -> Result<Plan> {
        self.inner.create_plan(goal, context).await
    }

    async fn revise_plan(&self, plan: &mut Plan, feedback: &str) -> Result<()> {
        // Count how many times we have revised (proxy: number of skipped steps).
        let revision_count = plan
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Skipped)
            .count();

        if revision_count >= self.max_retries {
            return Err(crate::agent::DeepAgentError::MiddlewareError(format!(
                "Maximum replanning retries ({}) exceeded",
                self.max_retries
            )));
        }

        self.inner.revise_plan(plan, feedback).await
    }
}

// ---------------------------------------------------------------------------
// PlanningMiddleware
// ---------------------------------------------------------------------------

/// Middleware that manages a plan and injects plan status into agent context.
///
/// Before each model call, the current plan summary is injected as a system
/// message. After each model call, the middleware checks for plan-related
/// updates in the model response.
pub struct PlanningMiddleware {
    /// The current plan, if any.
    plan: Arc<Mutex<Option<Plan>>>,
    /// The planning strategy to use.
    strategy: Arc<dyn PlanningStrategy>,
}

impl PlanningMiddleware {
    /// Create a new `PlanningMiddleware` with the given strategy.
    pub fn new(strategy: Arc<dyn PlanningStrategy>) -> Self {
        Self {
            plan: Arc::new(Mutex::new(None)),
            strategy,
        }
    }

    /// Set the current plan.
    pub async fn set_plan(&self, plan: Plan) {
        let mut current = self.plan.lock().await;
        *current = Some(plan);
    }

    /// Get a clone of the current plan, if any.
    pub async fn get_plan(&self) -> Option<Plan> {
        let current = self.plan.lock().await;
        current.clone()
    }

    /// Get a reference-guarded access to the current plan.
    /// Returns a clone for convenience.
    pub async fn current_plan(&self) -> Option<Plan> {
        self.get_plan().await
    }

    /// Create a plan for the given goal using the configured strategy.
    pub async fn create_plan(&self, goal: &str, context: &Value) -> Result<Plan> {
        let plan = self.strategy.create_plan(goal, context).await?;
        self.set_plan(plan.clone()).await;
        Ok(plan)
    }

    /// Revise the current plan with the given feedback.
    pub async fn revise_plan(&self, feedback: &str) -> Result<()> {
        let mut guard = self.plan.lock().await;
        if let Some(ref mut plan) = *guard {
            self.strategy.revise_plan(plan, feedback).await
        } else {
            Err(crate::agent::DeepAgentError::MiddlewareError(
                "No active plan to revise".to_string(),
            ))
        }
    }

    /// Update a step's status in the current plan.
    pub async fn update_step(
        &self,
        step_id: usize,
        status: PlanStepStatus,
        result: Option<String>,
    ) -> bool {
        let mut guard = self.plan.lock().await;
        if let Some(ref mut plan) = *guard {
            plan.update_step_status(step_id, status, result)
        } else {
            false
        }
    }
}

#[async_trait]
impl Middleware for PlanningMiddleware {
    fn name(&self) -> &str {
        "planning"
    }

    /// Before the model is called, inject the current plan status as context.
    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let guard = self.plan.lock().await;
        let plan = match guard.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };

        let summary = plan.status_summary();
        let next_step_info = if let Some(step) = plan.next_actionable_step() {
            format!("\n\nNext step to execute: Step {} - {}", step.id, step.description)
        } else if plan.is_complete() {
            "\n\nAll steps are complete.".to_string()
        } else {
            "\n\nNo actionable steps available (dependencies not met).".to_string()
        };

        let plan_context = format!("## Current Plan\n{}{}", summary, next_step_info);

        if let Some(messages) = state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            let plan_msg = json!({
                "type": "system",
                "content": plan_context
            });
            // Insert after any existing system prompt.
            let insert_pos = if messages
                .first()
                .and_then(|m| m.get("type"))
                .and_then(|t| t.as_str())
                == Some("system")
            {
                1
            } else {
                0
            };
            messages.insert(insert_pos, plan_msg);
        }

        Ok(())
    }

    /// After the model responds, check for plan update signals in the response.
    async fn after_model(&self, state: &mut AgentState) -> Result<()> {
        let messages = match state.get("messages").and_then(|v| v.as_array()) {
            Some(msgs) => msgs,
            None => return Ok(()),
        };

        let last_msg = match messages.last() {
            Some(msg) => msg,
            None => return Ok(()),
        };

        // Look for plan_update metadata in the response.
        if let Some(plan_update) = last_msg.get("plan_update") {
            let mut guard = self.plan.lock().await;
            if let Some(ref mut plan) = *guard {
                if let Some(step_id) = plan_update.get("step_id").and_then(|v| v.as_u64()) {
                    let status = match plan_update
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("completed")
                    {
                        "completed" => PlanStepStatus::Completed,
                        "failed" => PlanStepStatus::Failed,
                        "in_progress" => PlanStepStatus::InProgress,
                        "skipped" => PlanStepStatus::Skipped,
                        _ => PlanStepStatus::Pending,
                    };
                    let result = plan_update
                        .get("result")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    plan.update_step_status(step_id as usize, status, result);
                }
            }
        }

        Ok(())
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
    // Plan step basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_step_new() {
        let step = PlanStep::new(0, "Do something");
        assert_eq!(step.id, 0);
        assert_eq!(step.description, "Do something");
        assert_eq!(step.status, PlanStepStatus::Pending);
        assert!(step.dependencies.is_empty());
        assert!(step.result.is_none());
        assert!(step.created_at.is_some());
        assert!(step.completed_at.is_none());
    }

    #[test]
    fn test_plan_step_with_dependencies() {
        let step = PlanStep::new(2, "Final step")
            .with_dependency(0)
            .with_dependency(1);
        assert_eq!(step.dependencies, vec![0, 1]);
    }

    #[test]
    fn test_plan_step_with_dependencies_vec() {
        let step = PlanStep::new(3, "Step").with_dependencies(vec![0, 1, 2]);
        assert_eq!(step.dependencies, vec![0, 1, 2]);
    }

    #[test]
    fn test_plan_step_is_terminal() {
        let mut step = PlanStep::new(0, "s");
        assert!(!step.is_terminal());

        step.status = PlanStepStatus::InProgress;
        assert!(!step.is_terminal());

        step.status = PlanStepStatus::Completed;
        assert!(step.is_terminal());

        step.status = PlanStepStatus::Failed;
        assert!(step.is_terminal());

        step.status = PlanStepStatus::Skipped;
        assert!(step.is_terminal());
    }

    // -----------------------------------------------------------------------
    // Plan basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_new() {
        let plan = Plan::new("plan-1", "Build a house");
        assert_eq!(plan.id, "plan-1");
        assert_eq!(plan.goal, "Build a house");
        assert!(plan.steps.is_empty());
        assert!(plan.created_at.is_some());
    }

    #[test]
    fn test_plan_add_step() {
        let mut plan = Plan::new("p", "g");
        let id0 = plan.add_step(PlanStep::new(0, "Step A"));
        let id1 = plan.add_step(PlanStep::new(0, "Step B"));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].description, "Step A");
        assert_eq!(plan.steps[1].description, "Step B");
    }

    #[test]
    fn test_plan_get_step() {
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B"));

        assert_eq!(plan.get_step(0).unwrap().description, "A");
        assert_eq!(plan.get_step(1).unwrap().description, "B");
        assert!(plan.get_step(99).is_none());
    }

    #[test]
    fn test_plan_update_step_status() {
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));

        let updated = plan.update_step_status(0, PlanStepStatus::Completed, Some("done".into()));
        assert!(updated);
        assert_eq!(plan.get_step(0).unwrap().status, PlanStepStatus::Completed);
        assert_eq!(plan.get_step(0).unwrap().result.as_deref(), Some("done"));
        assert!(plan.get_step(0).unwrap().completed_at.is_some());

        let not_found = plan.update_step_status(99, PlanStepStatus::Failed, None);
        assert!(!not_found);
    }

    // -----------------------------------------------------------------------
    // Dependencies and ready steps
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_get_ready_steps_no_deps() {
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B"));

        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_plan_get_ready_steps_with_deps() {
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B").with_dependency(0));
        plan.add_step(PlanStep::new(0, "C").with_dependency(1));

        // Only step 0 is ready initially.
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 0);

        // Complete step 0 — step 1 becomes ready.
        plan.update_step_status(0, PlanStepStatus::Completed, None);
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 1);

        // Complete step 1 — step 2 becomes ready.
        plan.update_step_status(1, PlanStepStatus::Completed, None);
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, 2);
    }

    #[test]
    fn test_plan_next_actionable_step() {
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B").with_dependency(0));

        assert_eq!(plan.next_actionable_step().unwrap().id, 0);

        plan.update_step_status(0, PlanStepStatus::Completed, None);
        assert_eq!(plan.next_actionable_step().unwrap().id, 1);

        plan.update_step_status(1, PlanStepStatus::Completed, None);
        assert!(plan.next_actionable_step().is_none());
    }

    // -----------------------------------------------------------------------
    // Completion and progress
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_is_complete() {
        let mut plan = Plan::new("p", "g");
        // Empty plan is not complete.
        assert!(!plan.is_complete());

        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B"));
        assert!(!plan.is_complete());

        plan.update_step_status(0, PlanStepStatus::Completed, None);
        assert!(!plan.is_complete());

        plan.update_step_status(1, PlanStepStatus::Skipped, None);
        assert!(plan.is_complete());
    }

    #[test]
    fn test_plan_progress_percentage() {
        let mut plan = Plan::new("p", "g");
        assert_eq!(plan.progress_percentage(), 0.0);

        plan.add_step(PlanStep::new(0, "A"));
        plan.add_step(PlanStep::new(0, "B"));
        plan.add_step(PlanStep::new(0, "C"));
        plan.add_step(PlanStep::new(0, "D"));
        assert_eq!(plan.progress_percentage(), 0.0);

        plan.update_step_status(0, PlanStepStatus::Completed, None);
        assert_eq!(plan.progress_percentage(), 25.0);

        plan.update_step_status(1, PlanStepStatus::Completed, None);
        assert_eq!(plan.progress_percentage(), 50.0);

        plan.update_step_status(2, PlanStepStatus::Failed, None);
        assert_eq!(plan.progress_percentage(), 75.0);

        plan.update_step_status(3, PlanStepStatus::Skipped, None);
        assert_eq!(plan.progress_percentage(), 100.0);
    }

    #[test]
    fn test_plan_status_summary() {
        let mut plan = Plan::new("p1", "Build something");
        plan.add_step(PlanStep::new(0, "Step A"));
        plan.add_step(PlanStep::new(0, "Step B").with_dependency(0));

        let summary = plan.status_summary();
        assert!(summary.contains("p1"));
        assert!(summary.contains("Build something"));
        assert!(summary.contains("Step A"));
        assert!(summary.contains("Step B"));
        assert!(summary.contains("0%"));
    }

    // -----------------------------------------------------------------------
    // SimplePlanningStrategy
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_simple_strategy_structured_goal() {
        let strategy = SimplePlanningStrategy::new();
        let goal = "Build a website:\n1. Design the layout\n2. Implement the frontend\n3. Deploy";
        let plan = strategy.create_plan(goal, &json!({})).await.unwrap();

        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].description, "Design the layout");
        assert_eq!(plan.steps[1].description, "Implement the frontend");
        assert_eq!(plan.steps[2].description, "Deploy");
        // Sequential deps.
        assert!(plan.steps[0].dependencies.is_empty());
        assert_eq!(plan.steps[1].dependencies, vec![0]);
        assert_eq!(plan.steps[2].dependencies, vec![1]);
    }

    #[tokio::test]
    async fn test_simple_strategy_unstructured_goal() {
        let strategy = SimplePlanningStrategy::new();
        let goal = "Just do the thing";
        let plan = strategy.create_plan(goal, &json!({})).await.unwrap();

        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].description, goal);
    }

    #[tokio::test]
    async fn test_simple_strategy_revise_plan() {
        let strategy = SimplePlanningStrategy::new();
        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "Step A"));
        plan.update_step_status(0, PlanStepStatus::Failed, Some("error".into()));

        strategy.revise_plan(&mut plan, "- Fix the error\n- Retry").await.unwrap();

        // Failed step should be skipped.
        assert_eq!(plan.get_step(0).unwrap().status, PlanStepStatus::Skipped);
        // New steps added.
        assert!(plan.steps.len() >= 3);
    }

    // -----------------------------------------------------------------------
    // ReplanningStrategy
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_replanning_strategy_max_retries() {
        let inner = Arc::new(SimplePlanningStrategy::new());
        let strategy = ReplanningStrategy::new(inner, 1);

        assert_eq!(strategy.max_retries(), 1);

        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        plan.update_step_status(0, PlanStepStatus::Failed, None);

        // First revision: skips the failed step.
        strategy.revise_plan(&mut plan, "try again").await.unwrap();

        // Now we have 1 skipped step — next revision should fail.
        plan.update_step_status(1, PlanStepStatus::Failed, None);
        let result = strategy.revise_plan(&mut plan, "try once more").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // PlanningMiddleware
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_middleware_set_get_plan() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        assert!(mw.get_plan().await.is_none());

        let plan = Plan::new("test", "Test goal");
        mw.set_plan(plan.clone()).await;

        let retrieved = mw.get_plan().await.unwrap();
        assert_eq!(retrieved.id, "test");
        assert_eq!(retrieved.goal, "Test goal");
    }

    #[tokio::test]
    async fn test_middleware_current_plan() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        assert!(mw.current_plan().await.is_none());

        mw.set_plan(Plan::new("x", "y")).await;
        assert!(mw.current_plan().await.is_some());
    }

    #[tokio::test]
    async fn test_middleware_create_plan() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        let plan = mw.create_plan("1. A\n2. B", &json!({})).await.unwrap();
        assert_eq!(plan.steps.len(), 2);

        // Plan should be stored.
        assert!(mw.get_plan().await.is_some());
    }

    #[tokio::test]
    async fn test_middleware_update_step() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        // No plan — update should return false.
        assert!(!mw.update_step(0, PlanStepStatus::Completed, None).await);

        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        mw.set_plan(plan).await;

        assert!(mw.update_step(0, PlanStepStatus::Completed, Some("ok".into())).await);

        let p = mw.get_plan().await.unwrap();
        assert_eq!(p.get_step(0).unwrap().status, PlanStepStatus::Completed);
    }

    #[tokio::test]
    async fn test_middleware_before_model_no_plan() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        let mut state = json!({"messages": [{"type": "human", "content": "hi"}]});
        mw.before_model(&mut state).await.unwrap();

        // No plan — messages unchanged.
        assert_eq!(state["messages"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_middleware_before_model_injects_plan() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        let mut plan = Plan::new("p1", "Do things");
        plan.add_step(PlanStep::new(0, "First"));
        plan.add_step(PlanStep::new(0, "Second").with_dependency(0));
        mw.set_plan(plan).await;

        let mut state = json!({"messages": [{"type": "human", "content": "go"}]});
        mw.before_model(&mut state).await.unwrap();

        let messages = state["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        let injected = &messages[0];
        assert_eq!(injected["type"], "system");
        let content = injected["content"].as_str().unwrap();
        assert!(content.contains("Current Plan"));
        assert!(content.contains("First"));
        assert!(content.contains("Next step to execute"));
    }

    #[tokio::test]
    async fn test_middleware_after_model_with_plan_update() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);

        let mut plan = Plan::new("p", "g");
        plan.add_step(PlanStep::new(0, "A"));
        mw.set_plan(plan).await;

        let mut state = json!({
            "messages": [
                {"type": "human", "content": "go"},
                {
                    "type": "ai",
                    "content": "Done with step 0",
                    "plan_update": {
                        "step_id": 0,
                        "status": "completed",
                        "result": "success"
                    }
                }
            ]
        });

        mw.after_model(&mut state).await.unwrap();

        let p = mw.get_plan().await.unwrap();
        assert_eq!(p.get_step(0).unwrap().status, PlanStepStatus::Completed);
        assert_eq!(p.get_step(0).unwrap().result.as_deref(), Some("success"));
    }

    #[tokio::test]
    async fn test_middleware_name() {
        let strategy = Arc::new(SimplePlanningStrategy::new());
        let mw = PlanningMiddleware::new(strategy);
        assert_eq!(mw.name(), "planning");
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_step_status_serialization() {
        let status = PlanStepStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");

        let deserialized: PlanStepStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, PlanStepStatus::InProgress);
    }

    #[test]
    fn test_plan_serialization_roundtrip() {
        let mut plan = Plan::new("p1", "Test goal");
        plan.add_step(PlanStep::new(0, "Step A"));
        plan.add_step(PlanStep::new(0, "Step B").with_dependency(0));

        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: Plan = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "p1");
        assert_eq!(deserialized.goal, "Test goal");
        assert_eq!(deserialized.steps.len(), 2);
        assert_eq!(deserialized.steps[1].dependencies, vec![0]);
    }
}
