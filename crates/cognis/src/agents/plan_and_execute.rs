//! Plan-and-Execute agent implementation.
//!
//! Implements the Plan-and-Execute pattern:
//! 1. Create a plan from a high-level goal (via a [`Planner`])
//! 2. Execute each step sequentially (via a [`StepExecutor`])
//! 3. Optionally replan on failure (up to `max_replans` times)
//!
//! This mirrors the Python LangChain Plan-and-Execute agent pattern.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::error::{CognisError, Result};
use cognis_core::tools::base::BaseTool;

/// A thread-safe function that generates plan text from a template string.
type GeneratorFn = Box<dyn Fn(&str) -> Result<String> + Send + Sync>;

// ---------------------------------------------------------------------------
// PlanStepStatus
// ---------------------------------------------------------------------------

/// Status of a single step in a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStepStatus {
    /// Step has not been started yet.
    Pending,
    /// Step is currently being executed.
    InProgress,
    /// Step completed successfully.
    Completed,
    /// Step failed during execution.
    Failed,
    /// Step was skipped (e.g., due to replanning).
    Skipped,
}

impl fmt::Display for PlanStepStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanStepStatus::Pending => write!(f, "Pending"),
            PlanStepStatus::InProgress => write!(f, "InProgress"),
            PlanStepStatus::Completed => write!(f, "Completed"),
            PlanStepStatus::Failed => write!(f, "Failed"),
            PlanStepStatus::Skipped => write!(f, "Skipped"),
        }
    }
}

// ---------------------------------------------------------------------------
// PlanStep
// ---------------------------------------------------------------------------

/// A single step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Zero-based index of the step.
    pub index: usize,
    /// Human-readable description of what this step should accomplish.
    pub description: String,
    /// Current status of the step.
    pub status: PlanStepStatus,
    /// Result of executing this step, if available.
    pub result: Option<String>,
}

impl PlanStep {
    /// Create a new pending plan step.
    pub fn new(index: usize, description: impl Into<String>) -> Self {
        Self {
            index,
            description: description.into(),
            status: PlanStepStatus::Pending,
            result: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A structured plan consisting of ordered steps to achieve a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// The high-level goal this plan is designed to achieve.
    pub goal: String,
    /// The ordered list of steps.
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Create a new plan for the given goal.
    pub fn new(goal: impl Into<String>, steps: Vec<PlanStep>) -> Self {
        Self {
            goal: goal.into(),
            steps,
        }
    }

    /// Returns a mutable reference to the next step that is [`PlanStepStatus::Pending`],
    /// or `None` if all steps have been processed.
    pub fn next_step(&mut self) -> Option<&mut PlanStep> {
        self.steps
            .iter_mut()
            .find(|s| s.status == PlanStepStatus::Pending)
    }

    /// Returns `true` if every step has a terminal status (not `Pending` or `InProgress`).
    pub fn is_complete(&self) -> bool {
        !self.steps.is_empty()
            && self.steps.iter().all(|s| {
                matches!(
                    s.status,
                    PlanStepStatus::Completed | PlanStepStatus::Skipped | PlanStepStatus::Failed
                )
            })
    }

    /// Returns `(completed_or_skipped, total)` as a progress indicator.
    pub fn progress(&self) -> (usize, usize) {
        let done = self
            .steps
            .iter()
            .filter(|s| {
                s.status == PlanStepStatus::Completed || s.status == PlanStepStatus::Skipped
            })
            .count();
        (done, self.steps.len())
    }
}

// ---------------------------------------------------------------------------
// Planner trait
// ---------------------------------------------------------------------------

/// A planner generates a [`Plan`] from a high-level goal string.
pub trait Planner: Send + Sync {
    /// Create a plan to achieve the given goal.
    fn create_plan(&self, goal: &str) -> Result<Plan>;
}

// ---------------------------------------------------------------------------
// SimplePlanner
// ---------------------------------------------------------------------------

/// A simple planner that splits a goal into numbered steps by looking for
/// lines matching patterns like `1. Step description` or `1) Step description`.
///
/// If no numbered steps are found, the entire goal is used as a single step.
pub struct SimplePlanner;

impl SimplePlanner {
    /// Create a new `SimplePlanner`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimplePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner for SimplePlanner {
    fn create_plan(&self, goal: &str) -> Result<Plan> {
        let re = Regex::new(r"(?m)^\s*\d+[\.\)]\s*(.+)$")
            .map_err(|e| CognisError::Other(format!("Regex error: {}", e)))?;

        let steps: Vec<PlanStep> = re
            .captures_iter(goal)
            .enumerate()
            .map(|(i, cap)| PlanStep::new(i, cap[1].trim()))
            .collect();

        if steps.is_empty() {
            // Treat the entire goal as a single step.
            Ok(Plan::new(goal, vec![PlanStep::new(0, goal)]))
        } else {
            Ok(Plan::new(goal, steps))
        }
    }
}

// ---------------------------------------------------------------------------
// TemplatePlanner
// ---------------------------------------------------------------------------

/// A planner that uses a configurable prompt template for plan generation.
///
/// The template must contain the placeholder `{goal}` which will be replaced
/// with the actual goal. The result is then parsed for numbered steps
/// (same logic as [`SimplePlanner`]).
pub struct TemplatePlanner {
    /// The prompt template containing `{goal}`.
    template: String,
    /// Optional function that generates plan text from the expanded template.
    /// If `None`, the expanded template itself is parsed for numbered steps.
    generator: Option<GeneratorFn>,
}

impl TemplatePlanner {
    /// Create a new `TemplatePlanner` with the given template string.
    ///
    /// The template must contain `{goal}`.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            generator: None,
        }
    }

    /// Set a generator function that produces plan text from the expanded prompt.
    pub fn with_generator(
        mut self,
        gen: impl Fn(&str) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        self.generator = Some(Box::new(gen));
        self
    }

    /// Expand the template by replacing `{goal}` with the actual goal.
    pub fn expand_template(&self, goal: &str) -> String {
        self.template.replace("{goal}", goal)
    }
}

impl Planner for TemplatePlanner {
    fn create_plan(&self, goal: &str) -> Result<Plan> {
        let expanded = self.expand_template(goal);

        let plan_text = if let Some(ref gen) = self.generator {
            gen(&expanded)?
        } else {
            expanded.clone()
        };

        // Parse numbered steps from the plan text.
        let re = Regex::new(r"(?m)^\s*\d+[\.\)]\s*(.+)$")
            .map_err(|e| CognisError::Other(format!("Regex error: {}", e)))?;

        let steps: Vec<PlanStep> = re
            .captures_iter(&plan_text)
            .enumerate()
            .map(|(i, cap)| PlanStep::new(i, cap[1].trim()))
            .collect();

        if steps.is_empty() {
            Ok(Plan::new(goal, vec![PlanStep::new(0, goal)]))
        } else {
            Ok(Plan::new(goal, steps))
        }
    }
}

// ---------------------------------------------------------------------------
// StepExecutor trait
// ---------------------------------------------------------------------------

/// Executes a single [`PlanStep`], optionally using the provided context.
#[async_trait]
pub trait StepExecutor: Send + Sync {
    /// Execute the given step and return a result string.
    async fn execute_step(&self, step: &PlanStep, context: &Value) -> Result<String>;
}

// ---------------------------------------------------------------------------
// ToolStepExecutor
// ---------------------------------------------------------------------------

/// A step executor that matches step descriptions to available tools and
/// invokes the best-matching tool.
///
/// Matching is done by finding the tool whose name appears in the step
/// description (case-insensitive). If no tool matches, the step description
/// is returned as a passthrough result.
pub struct ToolStepExecutor {
    /// Available tools.
    tools: Vec<Arc<dyn BaseTool>>,
}

impl ToolStepExecutor {
    /// Create a new `ToolStepExecutor` with the given tools.
    pub fn new(tools: Vec<Arc<dyn BaseTool>>) -> Self {
        Self { tools }
    }

    /// Find the best matching tool for the given step description.
    fn find_tool(&self, description: &str) -> Option<Arc<dyn BaseTool>> {
        let desc_lower = description.to_lowercase();
        self.tools
            .iter()
            .find(|t| desc_lower.contains(&t.name().to_lowercase()))
            .cloned()
    }
}

#[async_trait]
impl StepExecutor for ToolStepExecutor {
    async fn execute_step(&self, step: &PlanStep, context: &Value) -> Result<String> {
        if let Some(tool) = self.find_tool(&step.description) {
            let input = if context.is_null() {
                Value::String(step.description.clone())
            } else {
                context.clone()
            };
            let result = tool.run_json(&input).await?;
            match result {
                Value::String(s) => Ok(s),
                other => Ok(serde_json::to_string(&other).unwrap_or_default()),
            }
        } else {
            // No matching tool — return description as a passthrough.
            Ok(format!(
                "No matching tool found. Step: {}",
                step.description
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// PlanAndExecuteResult
// ---------------------------------------------------------------------------

/// The result of running a [`PlanAndExecuteAgent`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAndExecuteResult {
    /// The final plan (including step statuses and results).
    pub plan: Plan,
    /// The final output string, typically the result of the last step.
    pub output: String,
    /// Total number of steps executed.
    pub total_steps: usize,
    /// Number of times the agent replanned.
    pub replans: usize,
    /// Pairs of `(step_description, step_result)` for each executed step.
    pub step_results: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent builder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`PlanAndExecuteAgent`].
pub struct PlanAndExecuteAgentBuilder {
    planner: Option<Box<dyn Planner>>,
    executor: Option<Box<dyn StepExecutor>>,
    max_replans: usize,
}

impl PlanAndExecuteAgentBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            planner: None,
            executor: None,
            max_replans: 3,
        }
    }

    /// Set the planner.
    pub fn planner(mut self, planner: impl Planner + 'static) -> Self {
        self.planner = Some(Box::new(planner));
        self
    }

    /// Set the step executor.
    pub fn executor(mut self, executor: impl StepExecutor + 'static) -> Self {
        self.executor = Some(Box::new(executor));
        self
    }

    /// Set the maximum number of replanning attempts on failure.
    pub fn max_replans(mut self, max: usize) -> Self {
        self.max_replans = max;
        self
    }

    /// Build the [`PlanAndExecuteAgent`].
    pub fn build(self) -> Result<PlanAndExecuteAgent> {
        let planner = self
            .planner
            .ok_or_else(|| CognisError::Other("PlanAndExecuteAgent requires a planner".into()))?;
        let executor = self
            .executor
            .ok_or_else(|| CognisError::Other("PlanAndExecuteAgent requires an executor".into()))?;
        Ok(PlanAndExecuteAgent {
            planner,
            executor,
            max_replans: self.max_replans,
        })
    }
}

impl Default for PlanAndExecuteAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent
// ---------------------------------------------------------------------------

/// An agent that first creates a plan, then executes each step sequentially.
///
/// On step failure, the agent can optionally replan (up to `max_replans` times)
/// to recover and continue execution.
pub struct PlanAndExecuteAgent {
    /// The planner that generates a structured plan from a goal.
    planner: Box<dyn Planner>,
    /// The executor that runs individual plan steps.
    executor: Box<dyn StepExecutor>,
    /// Maximum number of times the agent will replan on failure.
    max_replans: usize,
}

impl PlanAndExecuteAgent {
    /// Create a new builder for constructing a `PlanAndExecuteAgent`.
    pub fn builder() -> PlanAndExecuteAgentBuilder {
        PlanAndExecuteAgentBuilder::new()
    }

    /// Run the plan-and-execute loop on the given goal.
    pub async fn run(&self, goal: &str) -> Result<PlanAndExecuteResult> {
        self.run_with_callback(goal, |_| {}).await
    }

    /// Run the plan-and-execute loop, invoking `on_step` for each step
    /// before it is executed.
    pub async fn run_with_callback(
        &self,
        goal: &str,
        on_step: impl Fn(&PlanStep),
    ) -> Result<PlanAndExecuteResult> {
        let mut plan = self.planner.create_plan(goal)?;
        let mut replans = 0;
        let mut step_results: Vec<(String, String)> = Vec::new();
        let mut total_steps = 0;
        let context = Value::Null;

        loop {
            // Process all pending steps in the current plan.
            while let Some(step) = plan.next_step() {
                on_step(step);
                step.status = PlanStepStatus::InProgress;
                let desc = step.description.clone();
                let idx = step.index;

                match self.executor.execute_step(step, &context).await {
                    Ok(result) => {
                        // Re-borrow after the await.
                        plan.steps[idx].status = PlanStepStatus::Completed;
                        plan.steps[idx].result = Some(result.clone());
                        step_results.push((desc, result));
                        total_steps += 1;
                    }
                    Err(e) => {
                        plan.steps[idx].status = PlanStepStatus::Failed;
                        plan.steps[idx].result = Some(format!("Error: {}", e));
                        step_results.push((desc, format!("Error: {}", e)));
                        total_steps += 1;

                        // Try replanning if we haven't exhausted replans.
                        if replans < self.max_replans {
                            replans += 1;
                            // Skip remaining steps in the failed plan.
                            for s in plan.steps.iter_mut() {
                                if s.status == PlanStepStatus::Pending {
                                    s.status = PlanStepStatus::Skipped;
                                }
                            }
                            // Create a new plan for the same goal.
                            plan = self.planner.create_plan(goal)?;
                            break;
                        } else {
                            // No more replans — mark remaining as skipped and return.
                            for s in plan.steps.iter_mut() {
                                if s.status == PlanStepStatus::Pending {
                                    s.status = PlanStepStatus::Skipped;
                                }
                            }
                        }
                    }
                }
            }

            if plan.is_complete() {
                break;
            }
        }

        let output = step_results
            .last()
            .map(|(_, r)| r.clone())
            .unwrap_or_default();

        Ok(PlanAndExecuteResult {
            plan,
            output,
            total_steps,
            replans,
            step_results,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- Mock planner that returns a fixed list of steps ---------------------

    struct MockPlanner {
        steps: Vec<String>,
    }

    impl MockPlanner {
        fn new(steps: Vec<&str>) -> Self {
            Self {
                steps: steps.into_iter().map(String::from).collect(),
            }
        }
    }

    impl Planner for MockPlanner {
        fn create_plan(&self, goal: &str) -> Result<Plan> {
            let plan_steps = self
                .steps
                .iter()
                .enumerate()
                .map(|(i, desc)| PlanStep::new(i, desc.as_str()))
                .collect();
            Ok(Plan::new(goal, plan_steps))
        }
    }

    // -- Mock executor that returns a fixed result --------------------------

    struct MockExecutor {
        result: String,
    }

    impl MockExecutor {
        fn new(result: &str) -> Self {
            Self {
                result: result.to_string(),
            }
        }
    }

    #[async_trait]
    impl StepExecutor for MockExecutor {
        async fn execute_step(&self, _step: &PlanStep, _context: &Value) -> Result<String> {
            Ok(self.result.clone())
        }
    }

    // -- Mock executor that fails on specific steps -------------------------

    struct FailingExecutor {
        fail_on_indices: Vec<usize>,
        call_count: AtomicUsize,
    }

    impl FailingExecutor {
        fn new(fail_on_indices: Vec<usize>) -> Self {
            Self {
                fail_on_indices,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl StepExecutor for FailingExecutor {
        async fn execute_step(&self, step: &PlanStep, _context: &Value) -> Result<String> {
            let call = self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.fail_on_indices.contains(&call) {
                Err(CognisError::Other(format!("Step {} failed", step.index)))
            } else {
                Ok(format!("Result for step {}", step.index))
            }
        }
    }

    // -- Mock executor that records descriptions ----------------------------

    struct RecordingExecutor {
        recorded: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingExecutor {
        fn new() -> Self {
            Self {
                recorded: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn get_recorded(&self) -> Vec<String> {
            self.recorded.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl StepExecutor for RecordingExecutor {
        async fn execute_step(&self, step: &PlanStep, _context: &Value) -> Result<String> {
            self.recorded.lock().unwrap().push(step.description.clone());
            Ok(format!("Done: {}", step.description))
        }
    }

    // -----------------------------------------------------------------------
    // PlanStepStatus tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_step_status_display() {
        assert_eq!(PlanStepStatus::Pending.to_string(), "Pending");
        assert_eq!(PlanStepStatus::InProgress.to_string(), "InProgress");
        assert_eq!(PlanStepStatus::Completed.to_string(), "Completed");
        assert_eq!(PlanStepStatus::Failed.to_string(), "Failed");
        assert_eq!(PlanStepStatus::Skipped.to_string(), "Skipped");
    }

    #[test]
    fn plan_step_status_equality() {
        assert_eq!(PlanStepStatus::Pending, PlanStepStatus::Pending);
        assert_ne!(PlanStepStatus::Pending, PlanStepStatus::Completed);
    }

    // -----------------------------------------------------------------------
    // PlanStep tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_step_new_defaults() {
        let step = PlanStep::new(0, "Do something");
        assert_eq!(step.index, 0);
        assert_eq!(step.description, "Do something");
        assert_eq!(step.status, PlanStepStatus::Pending);
        assert!(step.result.is_none());
    }

    #[test]
    fn plan_step_serialization() {
        let step = PlanStep::new(1, "Test step");
        let json = serde_json::to_string(&step).unwrap();
        assert!(json.contains("\"index\":1"));
        assert!(json.contains("Test step"));
        let deserialized: PlanStep = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.index, 1);
        assert_eq!(deserialized.description, "Test step");
    }

    // -----------------------------------------------------------------------
    // Plan tests
    // -----------------------------------------------------------------------

    #[test]
    fn plan_new_with_steps() {
        let plan = Plan::new(
            "test goal",
            vec![PlanStep::new(0, "step 1"), PlanStep::new(1, "step 2")],
        );
        assert_eq!(plan.goal, "test goal");
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn plan_next_step_returns_first_pending() {
        let mut plan = Plan::new(
            "goal",
            vec![PlanStep::new(0, "first"), PlanStep::new(1, "second")],
        );
        let step = plan.next_step().unwrap();
        assert_eq!(step.index, 0);
    }

    #[test]
    fn plan_next_step_skips_completed() {
        let mut plan = Plan::new(
            "goal",
            vec![PlanStep::new(0, "first"), PlanStep::new(1, "second")],
        );
        plan.steps[0].status = PlanStepStatus::Completed;
        let step = plan.next_step().unwrap();
        assert_eq!(step.index, 1);
    }

    #[test]
    fn plan_next_step_none_when_all_done() {
        let mut plan = Plan::new("goal", vec![PlanStep::new(0, "first")]);
        plan.steps[0].status = PlanStepStatus::Completed;
        assert!(plan.next_step().is_none());
    }

    #[test]
    fn plan_is_complete_all_completed() {
        let mut plan = Plan::new("goal", vec![PlanStep::new(0, "a"), PlanStep::new(1, "b")]);
        plan.steps[0].status = PlanStepStatus::Completed;
        plan.steps[1].status = PlanStepStatus::Completed;
        assert!(plan.is_complete());
    }

    #[test]
    fn plan_is_complete_with_skipped() {
        let mut plan = Plan::new("goal", vec![PlanStep::new(0, "a"), PlanStep::new(1, "b")]);
        plan.steps[0].status = PlanStepStatus::Completed;
        plan.steps[1].status = PlanStepStatus::Skipped;
        assert!(plan.is_complete());
    }

    #[test]
    fn plan_is_not_complete_with_pending() {
        let mut plan = Plan::new("goal", vec![PlanStep::new(0, "a"), PlanStep::new(1, "b")]);
        plan.steps[0].status = PlanStepStatus::Completed;
        assert!(!plan.is_complete());
    }

    #[test]
    fn plan_is_not_complete_when_empty() {
        let plan = Plan::new("goal", vec![]);
        assert!(!plan.is_complete());
    }

    #[test]
    fn plan_progress_all_pending() {
        let plan = Plan::new(
            "goal",
            vec![
                PlanStep::new(0, "a"),
                PlanStep::new(1, "b"),
                PlanStep::new(2, "c"),
            ],
        );
        assert_eq!(plan.progress(), (0, 3));
    }

    #[test]
    fn plan_progress_some_completed() {
        let mut plan = Plan::new(
            "goal",
            vec![
                PlanStep::new(0, "a"),
                PlanStep::new(1, "b"),
                PlanStep::new(2, "c"),
            ],
        );
        plan.steps[0].status = PlanStepStatus::Completed;
        plan.steps[1].status = PlanStepStatus::Skipped;
        assert_eq!(plan.progress(), (2, 3));
    }

    // -----------------------------------------------------------------------
    // SimplePlanner tests
    // -----------------------------------------------------------------------

    #[test]
    fn simple_planner_parses_numbered_steps_dot() {
        let planner = SimplePlanner::new();
        let goal =
            "Do the following:\n1. Research the topic\n2. Write an outline\n3. Draft the article";
        let plan = planner.create_plan(goal).unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].description, "Research the topic");
        assert_eq!(plan.steps[1].description, "Write an outline");
        assert_eq!(plan.steps[2].description, "Draft the article");
    }

    #[test]
    fn simple_planner_parses_numbered_steps_paren() {
        let planner = SimplePlanner::new();
        let goal = "1) First step\n2) Second step";
        let plan = planner.create_plan(goal).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].description, "First step");
        assert_eq!(plan.steps[1].description, "Second step");
    }

    #[test]
    fn simple_planner_single_step_fallback() {
        let planner = SimplePlanner::new();
        let goal = "Just do this one thing";
        let plan = planner.create_plan(goal).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].description, "Just do this one thing");
    }

    #[test]
    fn simple_planner_default() {
        let planner = SimplePlanner::default();
        let plan = planner.create_plan("test").unwrap();
        assert_eq!(plan.steps.len(), 1);
    }

    // -----------------------------------------------------------------------
    // TemplatePlanner tests
    // -----------------------------------------------------------------------

    #[test]
    fn template_planner_expands_template() {
        let planner = TemplatePlanner::new("Plan for: {goal}");
        let expanded = planner.expand_template("build a house");
        assert_eq!(expanded, "Plan for: build a house");
    }

    #[test]
    fn template_planner_with_numbered_template() {
        let template = "Steps to achieve {goal}:\n1. Analyze requirements\n2. Implement solution\n3. Test and verify";
        let planner = TemplatePlanner::new(template);
        let plan = planner.create_plan("the goal").unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].description, "Analyze requirements");
    }

    #[test]
    fn template_planner_with_generator() {
        let planner = TemplatePlanner::new("Goal: {goal}")
            .with_generator(|_prompt| Ok("1. Generated step A\n2. Generated step B".to_string()));
        let plan = planner.create_plan("anything").unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].description, "Generated step A");
        assert_eq!(plan.steps[1].description, "Generated step B");
    }

    #[test]
    fn template_planner_generator_fallback() {
        let planner =
            TemplatePlanner::new("Goal: {goal}").with_generator(|_| Ok("no numbers here".into()));
        let plan = planner.create_plan("test").unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].description, "test");
    }

    // -----------------------------------------------------------------------
    // PlanAndExecuteAgent builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_requires_planner() {
        let result = PlanAndExecuteAgentBuilder::new()
            .executor(MockExecutor::new("ok"))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_requires_executor() {
        let result = PlanAndExecuteAgentBuilder::new()
            .planner(MockPlanner::new(vec!["step"]))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_default_max_replans() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["step"]))
            .executor(MockExecutor::new("ok"))
            .build()
            .unwrap();
        assert_eq!(agent.max_replans, 3);
    }

    #[test]
    fn builder_custom_max_replans() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["step"]))
            .executor(MockExecutor::new("ok"))
            .max_replans(5)
            .build()
            .unwrap();
        assert_eq!(agent.max_replans, 5);
    }

    // -----------------------------------------------------------------------
    // PlanAndExecuteAgent run tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn run_simple_plan() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["step 1", "step 2"]))
            .executor(MockExecutor::new("done"))
            .build()
            .unwrap();

        let result = agent.run("test goal").await.unwrap();
        assert_eq!(result.total_steps, 2);
        assert_eq!(result.replans, 0);
        assert_eq!(result.output, "done");
        assert!(result.plan.is_complete());
    }

    #[tokio::test]
    async fn run_single_step_plan() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["only step"]))
            .executor(MockExecutor::new("single result"))
            .build()
            .unwrap();

        let result = agent.run("simple").await.unwrap();
        assert_eq!(result.total_steps, 1);
        assert_eq!(result.output, "single result");
        assert_eq!(result.step_results.len(), 1);
        assert_eq!(result.step_results[0].0, "only step");
    }

    #[tokio::test]
    async fn run_with_callback_invoked() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["a", "b", "c"]))
            .executor(MockExecutor::new("ok"))
            .build()
            .unwrap();

        let callback_count = std::sync::Arc::new(AtomicUsize::new(0));
        let count_clone = callback_count.clone();

        let result = agent
            .run_with_callback("goal", move |_step| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            })
            .await
            .unwrap();

        assert_eq!(result.total_steps, 3);
        assert_eq!(callback_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn run_replan_on_failure() {
        // The executor fails on call index 0 (first step of first plan),
        // then succeeds on subsequent calls (replanned steps).
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["step 1", "step 2"]))
            .executor(FailingExecutor::new(vec![0]))
            .max_replans(2)
            .build()
            .unwrap();

        let result = agent.run("goal").await.unwrap();
        assert_eq!(result.replans, 1);
        assert!(result.plan.is_complete());
    }

    #[tokio::test]
    async fn run_exhaust_replans() {
        // Fails on every call — should exhaust all replans.
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["step"]))
            .executor(FailingExecutor::new(vec![0, 1, 2, 3]))
            .max_replans(2)
            .build()
            .unwrap();

        let result = agent.run("goal").await.unwrap();
        // 2 replans + 1 original = 3 attempts, but last one also fails.
        assert_eq!(result.replans, 2);
    }

    #[tokio::test]
    async fn run_step_results_collected() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["alpha", "beta", "gamma"]))
            .executor(MockExecutor::new("result"))
            .build()
            .unwrap();

        let result = agent.run("goal").await.unwrap();
        assert_eq!(result.step_results.len(), 3);
        assert_eq!(
            result.step_results[0],
            ("alpha".to_string(), "result".to_string())
        );
        assert_eq!(
            result.step_results[1],
            ("beta".to_string(), "result".to_string())
        );
        assert_eq!(
            result.step_results[2],
            ("gamma".to_string(), "result".to_string())
        );
    }

    #[tokio::test]
    async fn run_executor_sees_all_steps_in_order() {
        let recording = Arc::new(RecordingExecutor::new());
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["first", "second", "third"]))
            .executor(RecordingExecutorWrapper(recording.clone()))
            .build()
            .unwrap();

        agent.run("goal").await.unwrap();

        let recorded = recording.get_recorded();
        assert_eq!(recorded, vec!["first", "second", "third"]);
    }

    // Wrapper to use Arc<RecordingExecutor> as StepExecutor.
    struct RecordingExecutorWrapper(Arc<RecordingExecutor>);

    #[async_trait]
    impl StepExecutor for RecordingExecutorWrapper {
        async fn execute_step(&self, step: &PlanStep, context: &Value) -> Result<String> {
            self.0.execute_step(step, context).await
        }
    }

    #[tokio::test]
    async fn run_max_replans_zero_no_replan() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["fail step"]))
            .executor(FailingExecutor::new(vec![0]))
            .max_replans(0)
            .build()
            .unwrap();

        let result = agent.run("goal").await.unwrap();
        assert_eq!(result.replans, 0);
        assert!(result.output.contains("Error"));
    }

    // -----------------------------------------------------------------------
    // PlanAndExecuteResult tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn result_output_is_last_step_result() {
        let agent = PlanAndExecuteAgent::builder()
            .planner(MockPlanner::new(vec!["a", "b"]))
            .executor(MockExecutor::new("final"))
            .build()
            .unwrap();

        let result = agent.run("goal").await.unwrap();
        assert_eq!(result.output, "final");
    }

    #[test]
    fn result_serialization() {
        let result = PlanAndExecuteResult {
            plan: Plan::new("goal", vec![]),
            output: "output".to_string(),
            total_steps: 2,
            replans: 1,
            step_results: vec![("s1".to_string(), "r1".to_string())],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"output\":\"output\""));
        assert!(json.contains("\"total_steps\":2"));
        assert!(json.contains("\"replans\":1"));
    }

    // -----------------------------------------------------------------------
    // ToolStepExecutor tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn tool_executor_matches_tool_by_name() {
        use cognis_core::tools::SimpleTool;

        let tool: Arc<dyn BaseTool> = Arc::new(SimpleTool::new(
            "search",
            "Search for information",
            |q: &str| Ok(format!("Found: {}", q)),
        ));
        let executor = ToolStepExecutor::new(vec![tool]);

        let step = PlanStep::new(0, "search for rust documentation");
        let result = executor.execute_step(&step, &Value::Null).await.unwrap();
        assert!(result.contains("Found:"));
    }

    #[tokio::test]
    async fn tool_executor_no_match_passthrough() {
        let executor = ToolStepExecutor::new(vec![]);
        let step = PlanStep::new(0, "do something");
        let result = executor.execute_step(&step, &Value::Null).await.unwrap();
        assert!(result.contains("No matching tool found"));
    }

    #[tokio::test]
    async fn tool_executor_uses_context_when_provided() {
        use cognis_core::tools::SimpleTool;

        let tool: Arc<dyn BaseTool> = Arc::new(SimpleTool::new("calc", "Calculator", |q: &str| {
            Ok(format!("Calculated: {}", q))
        }));
        let executor = ToolStepExecutor::new(vec![tool]);

        let step = PlanStep::new(0, "calc something");
        let context = Value::String("2+2".to_string());
        let result = executor.execute_step(&step, &context).await.unwrap();
        assert!(result.contains("Calculated:"));
    }
}
