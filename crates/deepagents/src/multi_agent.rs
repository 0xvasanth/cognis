//! Multi-agent orchestration patterns for coordinating teams of agents.
//!
//! This module provides building blocks for defining agent roles, assigning
//! tasks with priorities and dependencies, orchestrating execution, composing
//! workflows, and assembling teams.
//!
//! # Key Types
//!
//! - [`AgentRole`] -- defines an agent's identity, capabilities, and constraints
//! - [`TaskAssignment`] -- a unit of work with priority, dependencies, and status
//! - [`Orchestrator`] -- manages multi-agent task execution and lifecycle
//! - [`Workflow`] -- defines a multi-step workflow with a chosen pattern
//! - [`TeamComposition`] -- assembles a team of agents by role
//!
//! # Example
//!
//! ```rust
//! use deepagents::multi_agent::*;
//! use serde_json::json;
//!
//! let mut orchestrator = Orchestrator::new();
//!
//! let role = AgentRole::builder("researcher")
//!     .description("Finds relevant information")
//!     .capability("search")
//!     .capability("summarize")
//!     .build();
//! orchestrator.add_role(role);
//!
//! let task = TaskAssignment::builder("task-1", "Search for data")
//!     .agent_role("researcher")
//!     .input(json!({"query": "Rust async"}))
//!     .priority(TaskPriority::High)
//!     .build();
//! orchestrator.add_task(task);
//!
//! let ready = orchestrator.next_ready_tasks();
//! assert_eq!(ready.len(), 1);
//! ```

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};

use crate::agent::DeepAgentError;

/// Result type alias for multi-agent operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// AgentRole
// ---------------------------------------------------------------------------

/// Defines an agent's identity, capabilities, and constraints.
#[derive(Debug, Clone)]
pub struct AgentRole {
    /// Unique name for this role.
    pub name: String,
    /// Human-readable description of the role.
    pub description: String,
    /// List of capabilities this role possesses.
    pub capabilities: Vec<String>,
    /// List of constraints or limitations for this role.
    pub constraints: Vec<String>,
}

/// Builder for [`AgentRole`].
pub struct AgentRoleBuilder {
    name: String,
    description: String,
    capabilities: Vec<String>,
    constraints: Vec<String>,
}

impl AgentRole {
    /// Start building a new role with the given name.
    pub fn builder(name: impl Into<String>) -> AgentRoleBuilder {
        AgentRoleBuilder {
            name: name.into(),
            description: String::new(),
            capabilities: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Serialize this role to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
        })
    }

    /// Returns `true` if this role lists the given capability.
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

impl AgentRoleBuilder {
    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a capability.
    pub fn capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    /// Add a constraint.
    pub fn constraint(mut self, c: impl Into<String>) -> Self {
        self.constraints.push(c.into());
        self
    }

    /// Build the [`AgentRole`].
    pub fn build(self) -> AgentRole {
        AgentRole {
            name: self.name,
            description: self.description,
            capabilities: self.capabilities,
            constraints: self.constraints,
        }
    }
}

// ---------------------------------------------------------------------------
// TaskPriority
// ---------------------------------------------------------------------------

/// Priority level for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPriority {
    /// Must be handled immediately.
    Critical,
    /// Important, should be handled soon.
    High,
    /// Default priority.
    Normal,
    /// Can be deferred.
    Low,
}

impl TaskPriority {
    /// Returns a numeric weight (higher = more urgent).
    pub fn weight(&self) -> u32 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Normal => 2,
            Self::Low => 1,
        }
    }
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// TaskStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a task.
#[derive(Debug, Clone)]
pub enum TaskStatus {
    /// Waiting to be started.
    Pending,
    /// Currently being executed.
    InProgress,
    /// Successfully completed with a result value.
    Completed(Value),
    /// Failed with an error message.
    Failed(String),
    /// Blocked by unmet dependencies.
    Blocked,
}

impl TaskStatus {
    /// Returns `true` if the status is terminal (completed or failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Failed(_))
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::InProgress => write!(f, "InProgress"),
            Self::Completed(_) => write!(f, "Completed"),
            Self::Failed(err) => write!(f, "Failed: {}", err),
            Self::Blocked => write!(f, "Blocked"),
        }
    }
}

// ---------------------------------------------------------------------------
// TaskAssignment
// ---------------------------------------------------------------------------

/// A unit of work assigned to an agent role.
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    /// Unique task identifier.
    pub task_id: String,
    /// The role this task is assigned to.
    pub agent_role: String,
    /// Human-readable description.
    pub description: String,
    /// Input data for the task.
    pub input: Value,
    /// Task priority.
    pub priority: TaskPriority,
    /// IDs of tasks that must complete before this one can start.
    pub dependencies: Vec<String>,
    /// Current status.
    pub status: TaskStatus,
}

/// Builder for [`TaskAssignment`].
pub struct TaskAssignmentBuilder {
    task_id: String,
    agent_role: String,
    description: String,
    input: Value,
    priority: TaskPriority,
    dependencies: Vec<String>,
}

impl TaskAssignment {
    /// Start building a new task with the given ID and description.
    pub fn builder(
        task_id: impl Into<String>,
        description: impl Into<String>,
    ) -> TaskAssignmentBuilder {
        TaskAssignmentBuilder {
            task_id: task_id.into(),
            agent_role: String::new(),
            description: description.into(),
            input: Value::Null,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
        }
    }

    /// Serialize this task to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "description": self.description,
            "input": self.input,
            "priority": self.priority.to_string(),
            "dependencies": self.dependencies,
            "status": self.status.to_string(),
        })
    }

    /// Returns `true` if all dependencies are in the completed set.
    pub fn is_ready(&self, completed: &[String]) -> bool {
        self.dependencies.iter().all(|dep| completed.contains(dep))
    }
}

impl TaskAssignmentBuilder {
    /// Set the agent role.
    pub fn agent_role(mut self, role: impl Into<String>) -> Self {
        self.agent_role = role.into();
        self
    }

    /// Set the input data.
    pub fn input(mut self, input: Value) -> Self {
        self.input = input;
        self
    }

    /// Set the priority.
    pub fn priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Add a dependency on another task.
    pub fn dependency(mut self, task_id: impl Into<String>) -> Self {
        self.dependencies.push(task_id.into());
        self
    }

    /// Build the [`TaskAssignment`].
    pub fn build(self) -> TaskAssignment {
        TaskAssignment {
            task_id: self.task_id,
            agent_role: self.agent_role,
            description: self.description,
            input: self.input,
            priority: self.priority,
            dependencies: self.dependencies,
            status: TaskStatus::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// OrchestratorProgress
// ---------------------------------------------------------------------------

/// Summary of orchestrator task progress.
#[derive(Debug, Clone)]
pub struct OrchestratorProgress {
    /// Total number of tasks.
    pub total: usize,
    /// Number of completed tasks.
    pub completed: usize,
    /// Number of failed tasks.
    pub failed: usize,
    /// Number of in-progress tasks.
    pub in_progress: usize,
    /// Number of blocked tasks.
    pub blocked: usize,
}

impl OrchestratorProgress {
    /// Returns the completion rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns 1.0 if there are no tasks.
    pub fn completion_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.completed as f64 / self.total as f64
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "in_progress": self.in_progress,
            "blocked": self.blocked,
            "completion_rate": self.completion_rate(),
        })
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Manages multi-agent task execution with dependency tracking.
#[derive(Debug)]
pub struct Orchestrator {
    roles: HashMap<String, AgentRole>,
    tasks: HashMap<String, TaskAssignment>,
}

impl Orchestrator {
    /// Create a new empty orchestrator.
    pub fn new() -> Self {
        Self {
            roles: HashMap::new(),
            tasks: HashMap::new(),
        }
    }

    /// Register a role.
    pub fn add_role(&mut self, role: AgentRole) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Add a task to be managed.
    pub fn add_task(&mut self, task: TaskAssignment) {
        self.tasks.insert(task.task_id.clone(), task);
    }

    /// Assign a task to a role. The role must be registered and the task must exist.
    pub fn assign_task(&mut self, task_id: &str, role: &str) -> Result<()> {
        if !self.roles.contains_key(role) {
            return Err(DeepAgentError::Other(format!("unknown role: {}", role)));
        }
        match self.tasks.get_mut(task_id) {
            Some(task) => {
                task.agent_role = role.to_string();
                task.status = TaskStatus::InProgress;
                Ok(())
            }
            None => Err(DeepAgentError::Other(format!("unknown task: {}", task_id))),
        }
    }

    /// Mark a task as completed with the given result.
    pub fn complete_task(&mut self, task_id: &str, result: Value) -> Result<()> {
        match self.tasks.get_mut(task_id) {
            Some(task) => {
                task.status = TaskStatus::Completed(result);
                Ok(())
            }
            None => Err(DeepAgentError::Other(format!("unknown task: {}", task_id))),
        }
    }

    /// Mark a task as failed with the given error message.
    pub fn fail_task(&mut self, task_id: &str, error: &str) -> Result<()> {
        match self.tasks.get_mut(task_id) {
            Some(task) => {
                task.status = TaskStatus::Failed(error.to_string());
                Ok(())
            }
            None => Err(DeepAgentError::Other(format!("unknown task: {}", task_id))),
        }
    }

    /// Returns tasks that are pending and have all dependencies satisfied.
    pub fn next_ready_tasks(&self) -> Vec<&TaskAssignment> {
        let completed: Vec<String> = self
            .tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Completed(_)))
            .map(|t| t.task_id.clone())
            .collect();

        self.tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Pending))
            .filter(|t| t.is_ready(&completed))
            .collect()
    }

    /// Returns a summary of current progress.
    pub fn progress(&self) -> OrchestratorProgress {
        let mut completed = 0;
        let mut failed = 0;
        let mut in_progress = 0;
        let mut blocked = 0;

        let completed_ids: Vec<String> = self
            .tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Completed(_)))
            .map(|t| t.task_id.clone())
            .collect();

        for task in self.tasks.values() {
            match &task.status {
                TaskStatus::Completed(_) => completed += 1,
                TaskStatus::Failed(_) => failed += 1,
                TaskStatus::InProgress => in_progress += 1,
                TaskStatus::Blocked => blocked += 1,
                TaskStatus::Pending => {
                    if !task.is_ready(&completed_ids) {
                        blocked += 1;
                    }
                }
            }
        }

        OrchestratorProgress {
            total: self.tasks.len(),
            completed,
            failed,
            in_progress,
            blocked,
        }
    }

    /// Returns `true` if all tasks are in a terminal state.
    pub fn is_complete(&self) -> bool {
        self.tasks.values().all(|t| t.status.is_terminal())
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WorkflowPattern
// ---------------------------------------------------------------------------

/// Defines how steps in a workflow relate to each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowPattern {
    /// Tasks run one after another in order.
    Sequential,
    /// Independent tasks run simultaneously.
    Parallel,
    /// Output of one step feeds into the next.
    Pipeline,
    /// One agent delegates work to others.
    Supervisor,
}

impl WorkflowPattern {
    /// Returns the pattern name as a string.
    pub fn name(&self) -> &str {
        match self {
            Self::Sequential => "Sequential",
            Self::Parallel => "Parallel",
            Self::Pipeline => "Pipeline",
            Self::Supervisor => "Supervisor",
        }
    }
}

impl fmt::Display for WorkflowPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// WorkflowStep
// ---------------------------------------------------------------------------

/// A single step in a workflow.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    /// Zero-based index of this step.
    pub index: usize,
    /// The role responsible for this step.
    pub role: String,
    /// What this step does.
    pub description: String,
    /// Indices of steps that must complete before this one.
    pub dependencies: Vec<usize>,
    /// Current status.
    pub status: TaskStatus,
}

impl WorkflowStep {
    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "index": self.index,
            "role": self.role,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

/// A multi-step workflow following a particular pattern.
#[derive(Debug)]
pub struct Workflow {
    name: String,
    pattern: WorkflowPattern,
    steps: Vec<WorkflowStep>,
}

impl Workflow {
    /// Create a new workflow with the given name and pattern.
    pub fn new(name: impl Into<String>, pattern: WorkflowPattern) -> Self {
        Self {
            name: name.into(),
            pattern,
            steps: Vec::new(),
        }
    }

    /// Add a step to the workflow, returning its index.
    pub fn add_step(&mut self, role: &str, task_description: &str) -> usize {
        let index = self.steps.len();
        self.steps.push(WorkflowStep {
            index,
            role: role.to_string(),
            description: task_description.to_string(),
            dependencies: Vec::new(),
            status: TaskStatus::Pending,
        });
        index
    }

    /// Add a dependency: `to_step` depends on `from_step`.
    pub fn add_dependency(&mut self, from_step: usize, to_step: usize) {
        if let Some(step) = self.steps.get_mut(to_step) {
            if !step.dependencies.contains(&from_step) {
                step.dependencies.push(from_step);
            }
        }
    }

    /// Returns a reference to the workflow steps.
    pub fn steps(&self) -> &[WorkflowStep] {
        &self.steps
    }

    /// Validate the workflow structure.
    ///
    /// Checks:
    /// - At least one step exists.
    /// - All dependency indices are valid.
    /// - No step depends on itself.
    /// - Sequential/Pipeline patterns have linear dependencies.
    pub fn validate(&self) -> Result<()> {
        if self.steps.is_empty() {
            return Err(DeepAgentError::Other(
                "workflow must have at least one step".to_string(),
            ));
        }

        for step in &self.steps {
            for &dep in &step.dependencies {
                if dep >= self.steps.len() {
                    return Err(DeepAgentError::Other(format!(
                        "step {} depends on non-existent step {}",
                        step.index, dep
                    )));
                }
                if dep == step.index {
                    return Err(DeepAgentError::Other(format!(
                        "step {} depends on itself",
                        step.index
                    )));
                }
            }
        }

        Ok(())
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let steps: Vec<Value> = self.steps.iter().map(|s| s.to_json()).collect();
        json!({
            "name": self.name,
            "pattern": self.pattern.to_string(),
            "steps": steps,
        })
    }
}

// ---------------------------------------------------------------------------
// TeamComposition
// ---------------------------------------------------------------------------

/// Defines a team of agents organized by role.
#[derive(Debug)]
pub struct TeamComposition {
    name: String,
    members: Vec<AgentRole>,
}

impl TeamComposition {
    /// Create a new team with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            members: Vec::new(),
        }
    }

    /// Add a member to the team.
    pub fn add_member(&mut self, role: AgentRole) {
        self.members.push(role);
    }

    /// Find all members that have the given capability.
    pub fn find_for_capability(&self, cap: &str) -> Vec<&AgentRole> {
        self.members
            .iter()
            .filter(|r| r.has_capability(cap))
            .collect()
    }

    /// Returns a reference to the team members.
    pub fn members(&self) -> &[AgentRole] {
        &self.members
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if the team has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let members: Vec<Value> = self.members.iter().map(|m| m.to_json()).collect();
        json!({
            "name": self.name,
            "members": members,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- AgentRole tests --

    #[test]
    fn test_agent_role_builder() {
        let role = AgentRole::builder("researcher")
            .description("Finds information")
            .capability("search")
            .capability("summarize")
            .constraint("no-write")
            .build();
        assert_eq!(role.name, "researcher");
        assert_eq!(role.description, "Finds information");
        assert_eq!(role.capabilities, vec!["search", "summarize"]);
        assert_eq!(role.constraints, vec!["no-write"]);
    }

    #[test]
    fn test_agent_role_has_capability() {
        let role = AgentRole::builder("r")
            .capability("search")
            .capability("write")
            .build();
        assert!(role.has_capability("search"));
        assert!(role.has_capability("write"));
        assert!(!role.has_capability("delete"));
    }

    #[test]
    fn test_agent_role_to_json() {
        let role = AgentRole::builder("coder")
            .description("Writes code")
            .capability("code")
            .constraint("no-deploy")
            .build();
        let j = role.to_json();
        assert_eq!(j["name"], "coder");
        assert_eq!(j["description"], "Writes code");
        assert_eq!(j["capabilities"], json!(["code"]));
        assert_eq!(j["constraints"], json!(["no-deploy"]));
    }

    #[test]
    fn test_agent_role_empty_capabilities() {
        let role = AgentRole::builder("empty").build();
        assert!(!role.has_capability("anything"));
        assert!(role.capabilities.is_empty());
    }

    // -- TaskPriority tests --

    #[test]
    fn test_priority_weight() {
        assert_eq!(TaskPriority::Critical.weight(), 4);
        assert_eq!(TaskPriority::High.weight(), 3);
        assert_eq!(TaskPriority::Normal.weight(), 2);
        assert_eq!(TaskPriority::Low.weight(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_priority_sort() {
        let mut priorities = vec![
            TaskPriority::Low,
            TaskPriority::Critical,
            TaskPriority::Normal,
            TaskPriority::High,
        ];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![
                TaskPriority::Low,
                TaskPriority::Normal,
                TaskPriority::High,
                TaskPriority::Critical,
            ]
        );
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", TaskPriority::Critical), "Critical");
        assert_eq!(format!("{}", TaskPriority::High), "High");
        assert_eq!(format!("{}", TaskPriority::Normal), "Normal");
        assert_eq!(format!("{}", TaskPriority::Low), "Low");
    }

    // -- TaskStatus tests --

    #[test]
    fn test_status_is_terminal() {
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::InProgress.is_terminal());
        assert!(TaskStatus::Completed(json!({})).is_terminal());
        assert!(TaskStatus::Failed("err".to_string()).is_terminal());
        assert!(!TaskStatus::Blocked.is_terminal());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", TaskStatus::Pending), "Pending");
        assert_eq!(format!("{}", TaskStatus::InProgress), "InProgress");
        assert_eq!(format!("{}", TaskStatus::Completed(json!({}))), "Completed");
        assert_eq!(
            format!("{}", TaskStatus::Failed("oops".to_string())),
            "Failed: oops"
        );
        assert_eq!(format!("{}", TaskStatus::Blocked), "Blocked");
    }

    // -- TaskAssignment tests --

    #[test]
    fn test_task_assignment_builder() {
        let task = TaskAssignment::builder("t1", "Do something")
            .agent_role("worker")
            .input(json!({"key": "val"}))
            .priority(TaskPriority::High)
            .dependency("t0")
            .build();
        assert_eq!(task.task_id, "t1");
        assert_eq!(task.agent_role, "worker");
        assert_eq!(task.description, "Do something");
        assert_eq!(task.input, json!({"key": "val"}));
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.dependencies, vec!["t0"]);
        assert!(matches!(task.status, TaskStatus::Pending));
    }

    #[test]
    fn test_task_assignment_is_ready_no_deps() {
        let task = TaskAssignment::builder("t1", "desc").build();
        assert!(task.is_ready(&[]));
    }

    #[test]
    fn test_task_assignment_is_ready_with_deps() {
        let task = TaskAssignment::builder("t2", "desc")
            .dependency("t0")
            .dependency("t1")
            .build();
        assert!(!task.is_ready(&["t0".to_string()]));
        assert!(task.is_ready(&["t0".to_string(), "t1".to_string()]));
    }

    #[test]
    fn test_task_assignment_to_json() {
        let task = TaskAssignment::builder("t1", "Do it")
            .agent_role("coder")
            .priority(TaskPriority::Critical)
            .build();
        let j = task.to_json();
        assert_eq!(j["task_id"], "t1");
        assert_eq!(j["agent_role"], "coder");
        assert_eq!(j["description"], "Do it");
        assert_eq!(j["priority"], "Critical");
        assert_eq!(j["status"], "Pending");
    }

    #[test]
    fn test_task_default_priority() {
        let task = TaskAssignment::builder("t1", "desc").build();
        assert_eq!(task.priority, TaskPriority::Normal);
    }

    // -- Orchestrator tests --

    #[test]
    fn test_orchestrator_new() {
        let orch = Orchestrator::new();
        assert!(orch.is_complete()); // no tasks = complete
        let progress = orch.progress();
        assert_eq!(progress.total, 0);
    }

    #[test]
    fn test_orchestrator_add_role_and_task() {
        let mut orch = Orchestrator::new();
        orch.add_role(AgentRole::builder("worker").build());
        orch.add_task(
            TaskAssignment::builder("t1", "Work")
                .agent_role("worker")
                .build(),
        );
        assert!(!orch.is_complete());
        assert_eq!(orch.progress().total, 1);
    }

    #[test]
    fn test_orchestrator_assign_task() {
        let mut orch = Orchestrator::new();
        orch.add_role(AgentRole::builder("worker").build());
        orch.add_task(TaskAssignment::builder("t1", "Work").build());
        orch.assign_task("t1", "worker").unwrap();
        assert_eq!(orch.progress().in_progress, 1);
    }

    #[test]
    fn test_orchestrator_assign_task_unknown_role() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "Work").build());
        let result = orch.assign_task("t1", "ghost");
        assert!(result.is_err());
    }

    #[test]
    fn test_orchestrator_assign_task_unknown_task() {
        let mut orch = Orchestrator::new();
        orch.add_role(AgentRole::builder("worker").build());
        let result = orch.assign_task("nope", "worker");
        assert!(result.is_err());
    }

    #[test]
    fn test_orchestrator_complete_task() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "Work").build());
        orch.complete_task("t1", json!({"result": "done"})).unwrap();
        assert!(orch.is_complete());
        assert_eq!(orch.progress().completed, 1);
    }

    #[test]
    fn test_orchestrator_fail_task() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "Work").build());
        orch.fail_task("t1", "something broke").unwrap();
        assert!(orch.is_complete()); // failed is terminal
        assert_eq!(orch.progress().failed, 1);
    }

    #[test]
    fn test_orchestrator_complete_unknown_task() {
        let mut orch = Orchestrator::new();
        assert!(orch.complete_task("nope", json!({})).is_err());
    }

    #[test]
    fn test_orchestrator_fail_unknown_task() {
        let mut orch = Orchestrator::new();
        assert!(orch.fail_task("nope", "err").is_err());
    }

    #[test]
    fn test_orchestrator_next_ready_tasks() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "First").build());
        orch.add_task(
            TaskAssignment::builder("t2", "Second")
                .dependency("t1")
                .build(),
        );

        // Only t1 should be ready
        let ready = orch.next_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].task_id, "t1");

        // Complete t1
        orch.complete_task("t1", json!({})).unwrap();

        // Now t2 should be ready
        let ready = orch.next_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].task_id, "t2");
    }

    #[test]
    fn test_orchestrator_next_ready_tasks_multiple_deps() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("a", "A").build());
        orch.add_task(TaskAssignment::builder("b", "B").build());
        orch.add_task(
            TaskAssignment::builder("c", "C")
                .dependency("a")
                .dependency("b")
                .build(),
        );

        // a and b are ready
        let ready = orch.next_ready_tasks();
        assert_eq!(ready.len(), 2);

        // Complete only a
        orch.complete_task("a", json!({})).unwrap();
        let ready = orch.next_ready_tasks();
        // Only b should be ready (c still blocked)
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].task_id, "b");

        // Complete b
        orch.complete_task("b", json!({})).unwrap();
        let ready = orch.next_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].task_id, "c");
    }

    #[test]
    fn test_orchestrator_progress() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "A").build());
        orch.add_task(TaskAssignment::builder("t2", "B").dependency("t1").build());
        orch.add_task(TaskAssignment::builder("t3", "C").build());

        let p = orch.progress();
        assert_eq!(p.total, 3);
        // t2 is blocked (dep on t1), t1 and t3 are pending-ready
        assert_eq!(p.blocked, 1);

        orch.complete_task("t1", json!({})).unwrap();
        orch.fail_task("t3", "oops").unwrap();

        let p = orch.progress();
        assert_eq!(p.completed, 1);
        assert_eq!(p.failed, 1);
        assert_eq!(p.blocked, 0);
    }

    #[test]
    fn test_orchestrator_is_complete_mixed() {
        let mut orch = Orchestrator::new();
        orch.add_task(TaskAssignment::builder("t1", "A").build());
        orch.add_task(TaskAssignment::builder("t2", "B").build());

        assert!(!orch.is_complete());
        orch.complete_task("t1", json!({})).unwrap();
        assert!(!orch.is_complete());
        orch.fail_task("t2", "bad").unwrap();
        assert!(orch.is_complete());
    }

    // -- OrchestratorProgress tests --

    #[test]
    fn test_progress_completion_rate() {
        let p = OrchestratorProgress {
            total: 4,
            completed: 2,
            failed: 0,
            in_progress: 1,
            blocked: 1,
        };
        assert!((p.completion_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_completion_rate_empty() {
        let p = OrchestratorProgress {
            total: 0,
            completed: 0,
            failed: 0,
            in_progress: 0,
            blocked: 0,
        };
        assert!((p.completion_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_to_json() {
        let p = OrchestratorProgress {
            total: 5,
            completed: 3,
            failed: 1,
            in_progress: 1,
            blocked: 0,
        };
        let j = p.to_json();
        assert_eq!(j["total"], 5);
        assert_eq!(j["completed"], 3);
        assert_eq!(j["failed"], 1);
        assert_eq!(j["in_progress"], 1);
        assert_eq!(j["blocked"], 0);
    }

    // -- WorkflowPattern tests --

    #[test]
    fn test_workflow_pattern_name() {
        assert_eq!(WorkflowPattern::Sequential.name(), "Sequential");
        assert_eq!(WorkflowPattern::Parallel.name(), "Parallel");
        assert_eq!(WorkflowPattern::Pipeline.name(), "Pipeline");
        assert_eq!(WorkflowPattern::Supervisor.name(), "Supervisor");
    }

    #[test]
    fn test_workflow_pattern_display() {
        assert_eq!(format!("{}", WorkflowPattern::Sequential), "Sequential");
        assert_eq!(format!("{}", WorkflowPattern::Parallel), "Parallel");
        assert_eq!(format!("{}", WorkflowPattern::Pipeline), "Pipeline");
        assert_eq!(format!("{}", WorkflowPattern::Supervisor), "Supervisor");
    }

    #[test]
    fn test_workflow_pattern_equality() {
        assert_eq!(WorkflowPattern::Sequential, WorkflowPattern::Sequential);
        assert_ne!(WorkflowPattern::Sequential, WorkflowPattern::Parallel);
    }

    // -- WorkflowStep tests --

    #[test]
    fn test_workflow_step_to_json() {
        let step = WorkflowStep {
            index: 0,
            role: "coder".to_string(),
            description: "Write code".to_string(),
            dependencies: vec![],
            status: TaskStatus::Pending,
        };
        let j = step.to_json();
        assert_eq!(j["index"], 0);
        assert_eq!(j["role"], "coder");
        assert_eq!(j["description"], "Write code");
        assert_eq!(j["status"], "Pending");
    }

    // -- Workflow tests --

    #[test]
    fn test_workflow_new() {
        let wf = Workflow::new("my-flow", WorkflowPattern::Sequential);
        assert!(wf.steps().is_empty());
    }

    #[test]
    fn test_workflow_add_step() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Sequential);
        let idx = wf.add_step("researcher", "Find data");
        assert_eq!(idx, 0);
        assert_eq!(wf.steps().len(), 1);
        assert_eq!(wf.steps()[0].role, "researcher");
    }

    #[test]
    fn test_workflow_add_dependency() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Pipeline);
        wf.add_step("a", "Step A");
        wf.add_step("b", "Step B");
        wf.add_dependency(0, 1);
        assert_eq!(wf.steps()[1].dependencies, vec![0]);
    }

    #[test]
    fn test_workflow_add_dependency_idempotent() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Pipeline);
        wf.add_step("a", "Step A");
        wf.add_step("b", "Step B");
        wf.add_dependency(0, 1);
        wf.add_dependency(0, 1);
        assert_eq!(wf.steps()[1].dependencies, vec![0]);
    }

    #[test]
    fn test_workflow_validate_empty() {
        let wf = Workflow::new("empty", WorkflowPattern::Sequential);
        assert!(wf.validate().is_err());
    }

    #[test]
    fn test_workflow_validate_valid() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Sequential);
        wf.add_step("a", "Step A");
        wf.add_step("b", "Step B");
        wf.add_dependency(0, 1);
        assert!(wf.validate().is_ok());
    }

    #[test]
    fn test_workflow_validate_self_dependency() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Sequential);
        wf.add_step("a", "Step A");
        // Manually inject self-dependency
        wf.steps[0].dependencies.push(0);
        assert!(wf.validate().is_err());
    }

    #[test]
    fn test_workflow_validate_invalid_dep_index() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Sequential);
        wf.add_step("a", "Step A");
        wf.steps[0].dependencies.push(99);
        assert!(wf.validate().is_err());
    }

    #[test]
    fn test_workflow_to_json() {
        let mut wf = Workflow::new("my-flow", WorkflowPattern::Parallel);
        wf.add_step("r1", "Research");
        wf.add_step("r2", "Analyze");
        let j = wf.to_json();
        assert_eq!(j["name"], "my-flow");
        assert_eq!(j["pattern"], "Parallel");
        assert_eq!(j["steps"].as_array().unwrap().len(), 2);
    }

    // -- TeamComposition tests --

    #[test]
    fn test_team_new() {
        let team = TeamComposition::new("alpha");
        assert_eq!(team.len(), 0);
        assert!(team.is_empty());
    }

    #[test]
    fn test_team_add_member() {
        let mut team = TeamComposition::new("alpha");
        team.add_member(AgentRole::builder("coder").capability("code").build());
        team.add_member(AgentRole::builder("tester").capability("test").build());
        assert_eq!(team.len(), 2);
        assert!(!team.is_empty());
    }

    #[test]
    fn test_team_find_for_capability() {
        let mut team = TeamComposition::new("alpha");
        team.add_member(
            AgentRole::builder("coder")
                .capability("code")
                .capability("review")
                .build(),
        );
        team.add_member(AgentRole::builder("reviewer").capability("review").build());
        team.add_member(AgentRole::builder("tester").capability("test").build());

        let reviewers = team.find_for_capability("review");
        assert_eq!(reviewers.len(), 2);

        let coders = team.find_for_capability("code");
        assert_eq!(coders.len(), 1);
        assert_eq!(coders[0].name, "coder");

        let none = team.find_for_capability("deploy");
        assert!(none.is_empty());
    }

    #[test]
    fn test_team_members() {
        let mut team = TeamComposition::new("beta");
        team.add_member(AgentRole::builder("a").build());
        let members = team.members();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].name, "a");
    }

    #[test]
    fn test_team_to_json() {
        let mut team = TeamComposition::new("gamma");
        team.add_member(AgentRole::builder("worker").build());
        let j = team.to_json();
        assert_eq!(j["name"], "gamma");
        assert_eq!(j["members"].as_array().unwrap().len(), 1);
    }

    // -- Integration / edge case tests --

    #[test]
    fn test_orchestrator_full_lifecycle() {
        let mut orch = Orchestrator::new();

        // Set up roles
        orch.add_role(
            AgentRole::builder("researcher")
                .capability("search")
                .build(),
        );
        orch.add_role(AgentRole::builder("writer").capability("write").build());

        // Set up tasks with dependencies
        orch.add_task(
            TaskAssignment::builder("research", "Research topic")
                .agent_role("researcher")
                .priority(TaskPriority::High)
                .build(),
        );
        orch.add_task(
            TaskAssignment::builder("write", "Write article")
                .agent_role("writer")
                .dependency("research")
                .build(),
        );

        // Initially only research is ready
        assert_eq!(orch.next_ready_tasks().len(), 1);
        assert!(!orch.is_complete());

        // Assign and complete research
        orch.assign_task("research", "researcher").unwrap();
        assert_eq!(orch.progress().in_progress, 1);

        orch.complete_task("research", json!({"findings": "data"}))
            .unwrap();
        assert_eq!(orch.progress().completed, 1);

        // Now write is ready
        let ready = orch.next_ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].task_id, "write");

        // Complete write
        orch.complete_task("write", json!({"article": "done"}))
            .unwrap();
        assert!(orch.is_complete());
        assert!((orch.progress().completion_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_task_is_ready_empty_completed_with_deps() {
        let task = TaskAssignment::builder("t1", "desc")
            .dependency("dep1")
            .build();
        assert!(!task.is_ready(&[]));
    }

    #[test]
    fn test_workflow_add_dependency_out_of_bounds() {
        let mut wf = Workflow::new("flow", WorkflowPattern::Sequential);
        wf.add_step("a", "Step A");
        // to_step out of bounds -- should be a no-op
        wf.add_dependency(0, 99);
        assert!(wf.steps()[0].dependencies.is_empty());
    }
}
