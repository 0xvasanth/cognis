//! Lifecycle management for deep agents.
//!
//! This module provides types and logic for controlling agent startup, shutdown,
//! health monitoring, and graceful degradation. It includes state machines for
//! tracking agent lifecycle, health checks, restart policies, and a monitor for
//! overseeing multiple agents simultaneously.

use serde_json::json;
use std::fmt;

// ---------------------------------------------------------------------------
// AgentState
// ---------------------------------------------------------------------------

/// Represents the current state of an agent in its lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    /// Agent is being initialized.
    Initializing,
    /// Agent is initialized and ready to run.
    Ready,
    /// Agent is actively running.
    Running,
    /// Agent execution is paused.
    Paused,
    /// Agent is in the process of stopping.
    Stopping,
    /// Agent has stopped normally.
    Stopped,
    /// Agent has entered a failed state with a reason.
    Failed(String),
}

impl AgentState {
    /// Returns `true` if the agent is in an active (non-terminal) state where
    /// it may still perform work.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            AgentState::Initializing
                | AgentState::Ready
                | AgentState::Running
                | AgentState::Paused
                | AgentState::Stopping
        )
    }

    /// Returns `true` if the agent is in a terminal state from which it will
    /// not transition without an external restart.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentState::Stopped | AgentState::Failed(_))
    }

    /// Determines whether a transition from the current state to `target` is
    /// valid according to the lifecycle state machine.
    pub fn can_transition_to(&self, target: &AgentState) -> bool {
        match self {
            AgentState::Initializing => matches!(target, AgentState::Ready | AgentState::Failed(_)),
            AgentState::Ready => matches!(
                target,
                AgentState::Running | AgentState::Stopped | AgentState::Failed(_)
            ),
            AgentState::Running => matches!(
                target,
                AgentState::Paused | AgentState::Stopping | AgentState::Failed(_)
            ),
            AgentState::Paused => matches!(
                target,
                AgentState::Running | AgentState::Stopping | AgentState::Failed(_)
            ),
            AgentState::Stopping => matches!(target, AgentState::Stopped | AgentState::Failed(_)),
            AgentState::Stopped => false,
            AgentState::Failed(_) => false,
        }
    }

    fn name(&self) -> &str {
        match self {
            AgentState::Initializing => "Initializing",
            AgentState::Ready => "Ready",
            AgentState::Running => "Running",
            AgentState::Paused => "Paused",
            AgentState::Stopping => "Stopping",
            AgentState::Stopped => "Stopped",
            AgentState::Failed(_) => "Failed",
        }
    }
}

impl fmt::Display for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentState::Failed(reason) => write!(f, "Failed({})", reason),
            other => write!(f, "{}", other.name()),
        }
    }
}

// ---------------------------------------------------------------------------
// LifecycleEvent
// ---------------------------------------------------------------------------

/// An event that can occur during an agent's lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleEvent {
    /// Agent was created.
    Created,
    /// Agent was started.
    Started,
    /// Agent was paused.
    Paused,
    /// Agent was resumed from a pause.
    Resumed,
    /// A processing step was completed.
    StepCompleted { step: usize },
    /// An error occurred.
    Error { message: String },
    /// Agent is shutting down.
    ShuttingDown,
    /// Agent was terminated.
    Terminated,
}

impl LifecycleEvent {
    /// Returns the short name of this event.
    pub fn name(&self) -> &str {
        match self {
            LifecycleEvent::Created => "Created",
            LifecycleEvent::Started => "Started",
            LifecycleEvent::Paused => "Paused",
            LifecycleEvent::Resumed => "Resumed",
            LifecycleEvent::StepCompleted { .. } => "StepCompleted",
            LifecycleEvent::Error { .. } => "Error",
            LifecycleEvent::ShuttingDown => "ShuttingDown",
            LifecycleEvent::Terminated => "Terminated",
        }
    }

    /// Serializes this event to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            LifecycleEvent::StepCompleted { step } => json!({
                "event": "StepCompleted",
                "step": step
            }),
            LifecycleEvent::Error { message } => json!({
                "event": "Error",
                "message": message
            }),
            other => json!({ "event": other.name() }),
        }
    }
}

impl fmt::Display for LifecycleEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleEvent::StepCompleted { step } => {
                write!(f, "StepCompleted(step={})", step)
            }
            LifecycleEvent::Error { message } => {
                write!(f, "Error({})", message)
            }
            _ => write!(f, "{}", self.name()),
        }
    }
}

// ---------------------------------------------------------------------------
// LifecycleError
// ---------------------------------------------------------------------------

/// Errors that may occur during lifecycle transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleError {
    /// The requested state transition is not valid.
    InvalidTransition { from: String, to: String },
    /// The agent is already running.
    AlreadyRunning,
    /// The agent has not been started yet.
    NotStarted,
    /// The agent has already been stopped.
    AlreadyStopped,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleError::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {} to {}", from, to)
            }
            LifecycleError::AlreadyRunning => write!(f, "agent is already running"),
            LifecycleError::NotStarted => write!(f, "agent has not been started"),
            LifecycleError::AlreadyStopped => write!(f, "agent has already been stopped"),
        }
    }
}

impl std::error::Error for LifecycleError {}

// ---------------------------------------------------------------------------
// LifecycleTransition
// ---------------------------------------------------------------------------

/// Records a single state transition in an agent's lifecycle.
#[derive(Debug, Clone)]
pub struct LifecycleTransition {
    /// The state before the transition.
    pub from: AgentState,
    /// The state after the transition.
    pub to: AgentState,
    /// ISO-style timestamp when the transition occurred.
    pub timestamp: String,
    /// Optional human-readable reason for the transition.
    pub reason: Option<String>,
}

impl LifecycleTransition {
    /// Creates a new transition record with the current timestamp.
    pub fn new(from: AgentState, to: AgentState) -> Self {
        Self {
            from,
            to,
            timestamp: Self::now(),
            reason: None,
        }
    }

    /// Attaches a reason to this transition, consuming and returning `self`.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Serializes this transition to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "from": self.from.to_string(),
            "to": self.to.to_string(),
            "timestamp": self.timestamp,
            "reason": self.reason,
        })
    }

    fn now() -> String {
        // Simple monotonic-ish timestamp for ordering. In production this would
        // use a real clock, but we only depend on std + serde here.
        use std::time::{SystemTime, UNIX_EPOCH};
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}.{:09}", d.as_secs(), d.subsec_nanos())
    }
}

// ---------------------------------------------------------------------------
// AgentLifecycle
// ---------------------------------------------------------------------------

/// Manages the lifecycle state machine for a single agent.
pub struct AgentLifecycle {
    agent_id: String,
    state: AgentState,
    history: Vec<LifecycleTransition>,
    step_count: usize,
}

impl AgentLifecycle {
    /// Creates a new lifecycle manager in the `Initializing` state.
    pub fn new(agent_id: String) -> Self {
        let mut lifecycle = Self {
            agent_id,
            state: AgentState::Initializing,
            history: Vec::new(),
            step_count: 0,
        };
        // Record the initial creation transition.
        lifecycle.history.push(
            LifecycleTransition::new(AgentState::Initializing, AgentState::Initializing)
                .with_reason("created"),
        );
        lifecycle
    }

    /// Returns the agent identifier.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns the current agent state.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Transitions the agent to `Running` (moving through `Ready` if needed).
    pub fn start(&mut self) -> Result<(), LifecycleError> {
        if self.state == AgentState::Running {
            return Err(LifecycleError::AlreadyRunning);
        }
        if self.state.is_terminal() {
            return Err(LifecycleError::AlreadyStopped);
        }
        // Move through Ready if currently Initializing.
        if self.state == AgentState::Initializing {
            self.transition(AgentState::Ready, None)?;
        }
        self.transition(AgentState::Running, Some("started"))
    }

    /// Pauses a running agent.
    pub fn pause(&mut self) -> Result<(), LifecycleError> {
        if self.state != AgentState::Running {
            if self.state == AgentState::Paused {
                return Err(LifecycleError::InvalidTransition {
                    from: "Paused".into(),
                    to: "Paused".into(),
                });
            }
            return Err(LifecycleError::NotStarted);
        }
        self.transition(AgentState::Paused, Some("paused"))
    }

    /// Resumes a paused agent back to `Running`.
    pub fn resume(&mut self) -> Result<(), LifecycleError> {
        if self.state != AgentState::Paused {
            if self.state == AgentState::Running {
                return Err(LifecycleError::AlreadyRunning);
            }
            return Err(LifecycleError::NotStarted);
        }
        self.transition(AgentState::Running, Some("resumed"))
    }

    /// Stops the agent (moving through `Stopping` then `Stopped`).
    pub fn stop(&mut self) -> Result<(), LifecycleError> {
        if self.state.is_terminal() {
            return Err(LifecycleError::AlreadyStopped);
        }
        if self.state == AgentState::Initializing {
            return Err(LifecycleError::NotStarted);
        }
        if self.state != AgentState::Stopping {
            self.transition(AgentState::Stopping, Some("stopping"))?;
        }
        self.transition(AgentState::Stopped, Some("stopped"))
    }

    /// Moves the agent into the `Failed` state with a reason.
    pub fn fail(&mut self, reason: String) {
        let from = self.state.clone();
        let to = AgentState::Failed(reason.clone());
        self.history
            .push(LifecycleTransition::new(from, to.clone()).with_reason(reason));
        self.state = to;
    }

    /// Returns the full transition history.
    pub fn history(&self) -> &[LifecycleTransition] {
        &self.history
    }

    /// Returns the number of steps completed while this agent was running.
    pub fn uptime_steps(&self) -> usize {
        self.step_count
    }

    /// Records a completed step (used by the execution engine).
    pub fn record_step(&mut self) {
        self.step_count += 1;
    }

    // Internal helper to validate and apply a transition.
    fn transition(
        &mut self,
        target: AgentState,
        reason: Option<&str>,
    ) -> Result<(), LifecycleError> {
        if !self.state.can_transition_to(&target) {
            return Err(LifecycleError::InvalidTransition {
                from: self.state.to_string(),
                to: target.to_string(),
            });
        }
        let from = self.state.clone();
        let mut t = LifecycleTransition::new(from, target.clone());
        if let Some(r) = reason {
            t = t.with_reason(r);
        }
        self.history.push(t);
        self.state = target;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HealthCheck
// ---------------------------------------------------------------------------

/// Monitors the health of an agent by tracking heartbeats, errors, and step
/// completions.
pub struct HealthCheck {
    agent_id: String,
    last_heartbeat: String,
    heartbeat_count: usize,
    error_messages: Vec<String>,
    consecutive_error_count: usize,
    steps_completed: usize,
    total_checks: usize,
    /// Maximum number of seconds between heartbeats before the agent is
    /// considered unhealthy.
    _heartbeat_threshold_secs: u64,
}

impl HealthCheck {
    /// Creates a new health check monitor for the given agent.
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            last_heartbeat: Self::now(),
            heartbeat_count: 0,
            error_messages: Vec::new(),
            consecutive_error_count: 0,
            steps_completed: 0,
            total_checks: 0,
            _heartbeat_threshold_secs: 30,
        }
    }

    /// Records a successful heartbeat, resetting the consecutive error counter.
    pub fn record_heartbeat(&mut self) {
        self.last_heartbeat = Self::now();
        self.heartbeat_count += 1;
        self.consecutive_error_count = 0;
        self.total_checks += 1;
    }

    /// Records an error message, incrementing the consecutive error counter.
    pub fn record_error(&mut self, msg: String) {
        self.error_messages.push(msg);
        self.consecutive_error_count += 1;
        self.total_checks += 1;
    }

    /// Records a completed processing step.
    pub fn record_step_completion(&mut self) {
        self.steps_completed += 1;
        self.consecutive_error_count = 0;
        self.total_checks += 1;
    }

    /// Returns `true` if the agent appears healthy. An agent is healthy when
    /// there have been heartbeats and fewer than 3 consecutive errors.
    pub fn is_healthy(&self) -> bool {
        self.heartbeat_count > 0 && self.consecutive_error_count < 3
    }

    /// Returns the number of consecutive errors since the last success.
    pub fn consecutive_errors(&self) -> usize {
        self.consecutive_error_count
    }

    /// Returns the total number of steps completed.
    pub fn steps_completed(&self) -> usize {
        self.steps_completed
    }

    /// Returns the overall error rate as a fraction of total checks.
    pub fn error_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.error_messages.len() as f64 / self.total_checks as f64
    }

    /// Returns the timestamp of the most recent heartbeat.
    pub fn last_heartbeat(&self) -> &str {
        &self.last_heartbeat
    }

    /// Serializes the health check state to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "agent_id": self.agent_id,
            "is_healthy": self.is_healthy(),
            "heartbeat_count": self.heartbeat_count,
            "consecutive_errors": self.consecutive_error_count,
            "steps_completed": self.steps_completed,
            "error_rate": self.error_rate(),
            "last_heartbeat": self.last_heartbeat,
            "total_errors": self.error_messages.len(),
        })
    }

    fn now() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}.{:09}", d.as_secs(), d.subsec_nanos())
    }
}

// ---------------------------------------------------------------------------
// GracefulShutdown
// ---------------------------------------------------------------------------

/// Manages a graceful shutdown sequence with an optional timeout and ordered
/// cleanup callbacks.
pub struct GracefulShutdown {
    timeout_secs: u64,
    initiated: bool,
    initiated_at: Option<std::time::Instant>,
    cleanups: Vec<(String, Box<dyn Fn()>)>,
}

impl GracefulShutdown {
    /// Creates a new shutdown manager with the given timeout (in seconds).
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            initiated: false,
            initiated_at: None,
            cleanups: Vec::new(),
        }
    }

    /// Marks the shutdown as initiated and starts the timeout clock.
    pub fn initiate(&mut self) {
        self.initiated = true;
        self.initiated_at = Some(std::time::Instant::now());
    }

    /// Returns `true` if shutdown has been initiated.
    pub fn is_shutting_down(&self) -> bool {
        self.initiated
    }

    /// Returns `true` if the shutdown timeout has elapsed.
    pub fn is_timed_out(&self) -> bool {
        match self.initiated_at {
            Some(at) => at.elapsed().as_secs() >= self.timeout_secs,
            None => false,
        }
    }

    /// Registers a named cleanup callback to be run during shutdown.
    pub fn register_cleanup(&mut self, name: String, callback: Box<dyn Fn()>) {
        self.cleanups.push((name, callback));
    }

    /// Runs all registered cleanup callbacks in order, returning the names of
    /// those that completed successfully.
    pub fn run_cleanups(&self) -> Vec<String> {
        let mut completed = Vec::new();
        for (name, cb) in &self.cleanups {
            // In production we'd catch panics; here we just run them.
            cb();
            completed.push(name.clone());
        }
        completed
    }
}

// ---------------------------------------------------------------------------
// RestartPolicy
// ---------------------------------------------------------------------------

/// Controls whether and how an agent should be restarted after stopping or
/// failing.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    kind: RestartKind,
    max_restarts: usize,
    backoff_ms: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum RestartKind {
    Never,
    Always,
    OnFailure,
}

impl RestartPolicy {
    /// Creates a policy that never restarts the agent.
    pub fn never() -> Self {
        Self {
            kind: RestartKind::Never,
            max_restarts: 0,
            backoff_ms: 0,
        }
    }

    /// Creates a policy that always restarts, up to `max_restarts` times.
    pub fn always(max_restarts: usize) -> Self {
        Self {
            kind: RestartKind::Always,
            max_restarts,
            backoff_ms: 0,
        }
    }

    /// Creates a policy that restarts only on failure, with exponential backoff.
    pub fn on_failure(max_restarts: usize, backoff_ms: u64) -> Self {
        Self {
            kind: RestartKind::OnFailure,
            max_restarts,
            backoff_ms,
        }
    }

    /// Determines whether the agent should be restarted given the current
    /// failure count.
    pub fn should_restart(&self, failure_count: usize) -> bool {
        match self.kind {
            RestartKind::Never => false,
            RestartKind::Always | RestartKind::OnFailure => failure_count < self.max_restarts,
        }
    }

    /// Returns the delay in milliseconds before the next restart attempt,
    /// applying exponential backoff if configured.
    pub fn delay_ms(&self, failure_count: usize) -> u64 {
        if self.backoff_ms == 0 {
            return 0;
        }
        // Exponential backoff capped at 2^10 multiplier.
        let exp = failure_count.min(10) as u32;
        self.backoff_ms.saturating_mul(2u64.saturating_pow(exp))
    }

    /// Serializes this policy to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        let kind_str = match self.kind {
            RestartKind::Never => "never",
            RestartKind::Always => "always",
            RestartKind::OnFailure => "on_failure",
        };
        json!({
            "kind": kind_str,
            "max_restarts": self.max_restarts,
            "backoff_ms": self.backoff_ms,
        })
    }
}

// ---------------------------------------------------------------------------
// LifecycleMonitor
// ---------------------------------------------------------------------------

/// Monitors multiple agent lifecycles, providing aggregate views.
pub struct LifecycleMonitor {
    agents: Vec<AgentLifecycle>,
}

impl LifecycleMonitor {
    /// Creates a new empty monitor.
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Registers an agent lifecycle for monitoring.
    pub fn register(&mut self, lifecycle: AgentLifecycle) {
        self.agents.push(lifecycle);
    }

    /// Returns the lifecycle for the given agent id, if registered.
    pub fn get(&self, agent_id: &str) -> Option<&AgentLifecycle> {
        self.agents.iter().find(|a| a.agent_id() == agent_id)
    }

    /// Returns the ids of all agents in an active (non-terminal) state.
    pub fn active_agents(&self) -> Vec<&str> {
        self.agents
            .iter()
            .filter(|a| a.state().is_active())
            .map(|a| a.agent_id())
            .collect()
    }

    /// Returns the ids of all agents in a `Failed` state.
    pub fn failed_agents(&self) -> Vec<&str> {
        self.agents
            .iter()
            .filter(|a| matches!(a.state(), AgentState::Failed(_)))
            .map(|a| a.agent_id())
            .collect()
    }

    /// Returns the total number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Serializes the monitor state to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let agents: Vec<serde_json::Value> = self
            .agents
            .iter()
            .map(|a| {
                json!({
                    "agent_id": a.agent_id(),
                    "state": a.state().to_string(),
                    "steps": a.uptime_steps(),
                })
            })
            .collect();
        json!({
            "agent_count": self.agent_count(),
            "active": self.active_agents().len(),
            "failed": self.failed_agents().len(),
            "agents": agents,
        })
    }
}

impl Default for LifecycleMonitor {
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

    // -- AgentState tests ---------------------------------------------------

    #[test]
    fn test_agent_state_is_active_initializing() {
        assert!(AgentState::Initializing.is_active());
    }

    #[test]
    fn test_agent_state_is_active_ready() {
        assert!(AgentState::Ready.is_active());
    }

    #[test]
    fn test_agent_state_is_active_running() {
        assert!(AgentState::Running.is_active());
    }

    #[test]
    fn test_agent_state_is_active_paused() {
        assert!(AgentState::Paused.is_active());
    }

    #[test]
    fn test_agent_state_is_active_stopping() {
        assert!(AgentState::Stopping.is_active());
    }

    #[test]
    fn test_agent_state_is_not_active_stopped() {
        assert!(!AgentState::Stopped.is_active());
    }

    #[test]
    fn test_agent_state_is_not_active_failed() {
        assert!(!AgentState::Failed("err".into()).is_active());
    }

    #[test]
    fn test_agent_state_is_terminal_stopped() {
        assert!(AgentState::Stopped.is_terminal());
    }

    #[test]
    fn test_agent_state_is_terminal_failed() {
        assert!(AgentState::Failed("oops".into()).is_terminal());
    }

    #[test]
    fn test_agent_state_is_not_terminal_running() {
        assert!(!AgentState::Running.is_terminal());
    }

    #[test]
    fn test_agent_state_can_transition_init_to_ready() {
        assert!(AgentState::Initializing.can_transition_to(&AgentState::Ready));
    }

    #[test]
    fn test_agent_state_cannot_transition_init_to_running() {
        assert!(!AgentState::Initializing.can_transition_to(&AgentState::Running));
    }

    #[test]
    fn test_agent_state_can_transition_ready_to_running() {
        assert!(AgentState::Ready.can_transition_to(&AgentState::Running));
    }

    #[test]
    fn test_agent_state_can_transition_running_to_paused() {
        assert!(AgentState::Running.can_transition_to(&AgentState::Paused));
    }

    #[test]
    fn test_agent_state_can_transition_paused_to_running() {
        assert!(AgentState::Paused.can_transition_to(&AgentState::Running));
    }

    #[test]
    fn test_agent_state_cannot_transition_stopped() {
        assert!(!AgentState::Stopped.can_transition_to(&AgentState::Running));
    }

    #[test]
    fn test_agent_state_cannot_transition_failed() {
        assert!(!AgentState::Failed("x".into()).can_transition_to(&AgentState::Running));
    }

    #[test]
    fn test_agent_state_can_transition_running_to_failed() {
        assert!(AgentState::Running.can_transition_to(&AgentState::Failed("x".into())));
    }

    #[test]
    fn test_agent_state_display_running() {
        assert_eq!(AgentState::Running.to_string(), "Running");
    }

    #[test]
    fn test_agent_state_display_failed() {
        let s = AgentState::Failed("timeout".into()).to_string();
        assert_eq!(s, "Failed(timeout)");
    }

    // -- LifecycleEvent tests -----------------------------------------------

    #[test]
    fn test_lifecycle_event_name_created() {
        assert_eq!(LifecycleEvent::Created.name(), "Created");
    }

    #[test]
    fn test_lifecycle_event_name_step_completed() {
        let e = LifecycleEvent::StepCompleted { step: 5 };
        assert_eq!(e.name(), "StepCompleted");
    }

    #[test]
    fn test_lifecycle_event_to_json_created() {
        let j = LifecycleEvent::Created.to_json();
        assert_eq!(j["event"], "Created");
    }

    #[test]
    fn test_lifecycle_event_to_json_step_completed() {
        let j = LifecycleEvent::StepCompleted { step: 3 }.to_json();
        assert_eq!(j["step"], 3);
    }

    #[test]
    fn test_lifecycle_event_to_json_error() {
        let j = LifecycleEvent::Error {
            message: "oops".into(),
        }
        .to_json();
        assert_eq!(j["message"], "oops");
    }

    #[test]
    fn test_lifecycle_event_display_created() {
        assert_eq!(LifecycleEvent::Created.to_string(), "Created");
    }

    #[test]
    fn test_lifecycle_event_display_step() {
        let s = LifecycleEvent::StepCompleted { step: 7 }.to_string();
        assert!(s.contains("7"));
    }

    #[test]
    fn test_lifecycle_event_display_error() {
        let s = LifecycleEvent::Error {
            message: "bad".into(),
        }
        .to_string();
        assert!(s.contains("bad"));
    }

    // -- LifecycleError tests -----------------------------------------------

    #[test]
    fn test_lifecycle_error_display_invalid_transition() {
        let e = LifecycleError::InvalidTransition {
            from: "A".into(),
            to: "B".into(),
        };
        let s = e.to_string();
        assert!(s.contains("A"));
        assert!(s.contains("B"));
    }

    #[test]
    fn test_lifecycle_error_display_already_running() {
        assert!(LifecycleError::AlreadyRunning
            .to_string()
            .contains("already running"));
    }

    #[test]
    fn test_lifecycle_error_display_not_started() {
        assert!(LifecycleError::NotStarted
            .to_string()
            .contains("not been started"));
    }

    #[test]
    fn test_lifecycle_error_display_already_stopped() {
        assert!(LifecycleError::AlreadyStopped
            .to_string()
            .contains("already been stopped"));
    }

    #[test]
    fn test_lifecycle_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(LifecycleError::NotStarted);
        assert!(!e.to_string().is_empty());
    }

    // -- LifecycleTransition tests ------------------------------------------

    #[test]
    fn test_transition_new_has_timestamp() {
        let t = LifecycleTransition::new(AgentState::Ready, AgentState::Running);
        assert!(!t.timestamp.is_empty());
        assert!(t.reason.is_none());
    }

    #[test]
    fn test_transition_with_reason() {
        let t = LifecycleTransition::new(AgentState::Running, AgentState::Paused)
            .with_reason("user request");
        assert_eq!(t.reason.as_deref(), Some("user request"));
    }

    #[test]
    fn test_transition_to_json() {
        let t =
            LifecycleTransition::new(AgentState::Running, AgentState::Stopped).with_reason("done");
        let j = t.to_json();
        assert_eq!(j["from"], "Running");
        assert_eq!(j["to"], "Stopped");
        assert_eq!(j["reason"], "done");
    }

    // -- AgentLifecycle tests -----------------------------------------------

    #[test]
    fn test_lifecycle_new_is_initializing() {
        let lc = AgentLifecycle::new("a1".into());
        assert_eq!(*lc.state(), AgentState::Initializing);
    }

    #[test]
    fn test_lifecycle_start() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        assert_eq!(*lc.state(), AgentState::Running);
    }

    #[test]
    fn test_lifecycle_start_already_running() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        assert_eq!(lc.start(), Err(LifecycleError::AlreadyRunning));
    }

    #[test]
    fn test_lifecycle_pause_and_resume() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.pause().unwrap();
        assert_eq!(*lc.state(), AgentState::Paused);
        lc.resume().unwrap();
        assert_eq!(*lc.state(), AgentState::Running);
    }

    #[test]
    fn test_lifecycle_pause_not_running() {
        let mut lc = AgentLifecycle::new("a1".into());
        assert_eq!(lc.pause(), Err(LifecycleError::NotStarted));
    }

    #[test]
    fn test_lifecycle_resume_not_paused() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        assert_eq!(lc.resume(), Err(LifecycleError::AlreadyRunning));
    }

    #[test]
    fn test_lifecycle_stop() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.stop().unwrap();
        assert_eq!(*lc.state(), AgentState::Stopped);
    }

    #[test]
    fn test_lifecycle_stop_already_stopped() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.stop().unwrap();
        assert_eq!(lc.stop(), Err(LifecycleError::AlreadyStopped));
    }

    #[test]
    fn test_lifecycle_stop_not_started() {
        let mut lc = AgentLifecycle::new("a1".into());
        assert_eq!(lc.stop(), Err(LifecycleError::NotStarted));
    }

    #[test]
    fn test_lifecycle_fail() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.fail("something broke".into());
        assert_eq!(*lc.state(), AgentState::Failed("something broke".into()));
    }

    #[test]
    fn test_lifecycle_fail_is_terminal() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.fail("err".into());
        assert!(lc.state().is_terminal());
    }

    #[test]
    fn test_lifecycle_history_recorded() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.pause().unwrap();
        // creation + init->ready + ready->running + running->paused = 4
        assert!(lc.history().len() >= 4);
    }

    #[test]
    fn test_lifecycle_uptime_steps() {
        let mut lc = AgentLifecycle::new("a1".into());
        assert_eq!(lc.uptime_steps(), 0);
        lc.record_step();
        lc.record_step();
        assert_eq!(lc.uptime_steps(), 2);
    }

    #[test]
    fn test_lifecycle_agent_id() {
        let lc = AgentLifecycle::new("test-agent".into());
        assert_eq!(lc.agent_id(), "test-agent");
    }

    #[test]
    fn test_lifecycle_start_after_stopped_fails() {
        let mut lc = AgentLifecycle::new("a1".into());
        lc.start().unwrap();
        lc.stop().unwrap();
        assert_eq!(lc.start(), Err(LifecycleError::AlreadyStopped));
    }

    // -- HealthCheck tests --------------------------------------------------

    #[test]
    fn test_health_check_new_not_healthy() {
        let hc = HealthCheck::new("a1".into());
        // No heartbeats recorded yet.
        assert!(!hc.is_healthy());
    }

    #[test]
    fn test_health_check_after_heartbeat() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        assert!(hc.is_healthy());
    }

    #[test]
    fn test_health_check_consecutive_errors() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        hc.record_error("e1".into());
        hc.record_error("e2".into());
        assert_eq!(hc.consecutive_errors(), 2);
        assert!(hc.is_healthy()); // still < 3
    }

    #[test]
    fn test_health_check_unhealthy_after_three_errors() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        hc.record_error("e1".into());
        hc.record_error("e2".into());
        hc.record_error("e3".into());
        assert!(!hc.is_healthy());
    }

    #[test]
    fn test_health_check_heartbeat_resets_errors() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        hc.record_error("e1".into());
        hc.record_error("e2".into());
        hc.record_heartbeat();
        assert_eq!(hc.consecutive_errors(), 0);
    }

    #[test]
    fn test_health_check_steps_completed() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_step_completion();
        hc.record_step_completion();
        assert_eq!(hc.steps_completed(), 2);
    }

    #[test]
    fn test_health_check_error_rate() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat(); // 1 check
        hc.record_error("e".into()); // 2 checks, 1 error
                                     // error_rate = 1/2 = 0.5
        assert!((hc.error_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_health_check_error_rate_zero() {
        let hc = HealthCheck::new("a1".into());
        assert!((hc.error_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_health_check_last_heartbeat() {
        let hc = HealthCheck::new("a1".into());
        assert!(!hc.last_heartbeat().is_empty());
    }

    #[test]
    fn test_health_check_to_json() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        let j = hc.to_json();
        assert_eq!(j["agent_id"], "a1");
        assert_eq!(j["is_healthy"], true);
    }

    #[test]
    fn test_health_check_step_resets_consecutive_errors() {
        let mut hc = HealthCheck::new("a1".into());
        hc.record_heartbeat();
        hc.record_error("e1".into());
        hc.record_step_completion();
        assert_eq!(hc.consecutive_errors(), 0);
    }

    // -- GracefulShutdown tests ---------------------------------------------

    #[test]
    fn test_graceful_shutdown_not_initiated() {
        let gs = GracefulShutdown::new(30);
        assert!(!gs.is_shutting_down());
    }

    #[test]
    fn test_graceful_shutdown_initiate() {
        let mut gs = GracefulShutdown::new(30);
        gs.initiate();
        assert!(gs.is_shutting_down());
    }

    #[test]
    fn test_graceful_shutdown_not_timed_out_before_initiate() {
        let gs = GracefulShutdown::new(30);
        assert!(!gs.is_timed_out());
    }

    #[test]
    fn test_graceful_shutdown_not_timed_out_immediately() {
        let mut gs = GracefulShutdown::new(30);
        gs.initiate();
        assert!(!gs.is_timed_out());
    }

    #[test]
    fn test_graceful_shutdown_timed_out_zero_timeout() {
        let mut gs = GracefulShutdown::new(0);
        gs.initiate();
        // With 0 timeout, should immediately be timed out.
        assert!(gs.is_timed_out());
    }

    #[test]
    fn test_graceful_shutdown_register_and_run_cleanups() {
        let mut gs = GracefulShutdown::new(30);
        gs.register_cleanup("cleanup1".into(), Box::new(|| {}));
        gs.register_cleanup("cleanup2".into(), Box::new(|| {}));
        let completed = gs.run_cleanups();
        assert_eq!(completed, vec!["cleanup1", "cleanup2"]);
    }

    #[test]
    fn test_graceful_shutdown_no_cleanups() {
        let gs = GracefulShutdown::new(10);
        let completed = gs.run_cleanups();
        assert!(completed.is_empty());
    }

    // -- RestartPolicy tests ------------------------------------------------

    #[test]
    fn test_restart_policy_never() {
        let p = RestartPolicy::never();
        assert!(!p.should_restart(0));
        assert!(!p.should_restart(5));
    }

    #[test]
    fn test_restart_policy_always_within_limit() {
        let p = RestartPolicy::always(3);
        assert!(p.should_restart(0));
        assert!(p.should_restart(2));
        assert!(!p.should_restart(3));
    }

    #[test]
    fn test_restart_policy_on_failure_within_limit() {
        let p = RestartPolicy::on_failure(2, 100);
        assert!(p.should_restart(0));
        assert!(p.should_restart(1));
        assert!(!p.should_restart(2));
    }

    #[test]
    fn test_restart_policy_delay_no_backoff() {
        let p = RestartPolicy::always(5);
        assert_eq!(p.delay_ms(0), 0);
        assert_eq!(p.delay_ms(3), 0);
    }

    #[test]
    fn test_restart_policy_delay_with_backoff() {
        let p = RestartPolicy::on_failure(10, 100);
        assert_eq!(p.delay_ms(0), 100); // 100 * 2^0
        assert_eq!(p.delay_ms(1), 200); // 100 * 2^1
        assert_eq!(p.delay_ms(2), 400); // 100 * 2^2
    }

    #[test]
    fn test_restart_policy_delay_capped() {
        let p = RestartPolicy::on_failure(20, 100);
        // 2^10 = 1024, so max is 100 * 1024 = 102400
        assert_eq!(p.delay_ms(10), 102_400);
        assert_eq!(p.delay_ms(15), 102_400); // capped at 10
    }

    #[test]
    fn test_restart_policy_to_json_never() {
        let j = RestartPolicy::never().to_json();
        assert_eq!(j["kind"], "never");
    }

    #[test]
    fn test_restart_policy_to_json_always() {
        let j = RestartPolicy::always(5).to_json();
        assert_eq!(j["kind"], "always");
        assert_eq!(j["max_restarts"], 5);
    }

    #[test]
    fn test_restart_policy_to_json_on_failure() {
        let j = RestartPolicy::on_failure(3, 200).to_json();
        assert_eq!(j["kind"], "on_failure");
        assert_eq!(j["backoff_ms"], 200);
    }

    // -- LifecycleMonitor tests ---------------------------------------------

    #[test]
    fn test_monitor_new_empty() {
        let m = LifecycleMonitor::new();
        assert_eq!(m.agent_count(), 0);
    }

    #[test]
    fn test_monitor_register_and_count() {
        let mut m = LifecycleMonitor::new();
        m.register(AgentLifecycle::new("a1".into()));
        m.register(AgentLifecycle::new("a2".into()));
        assert_eq!(m.agent_count(), 2);
    }

    #[test]
    fn test_monitor_get_existing() {
        let mut m = LifecycleMonitor::new();
        m.register(AgentLifecycle::new("a1".into()));
        assert!(m.get("a1").is_some());
    }

    #[test]
    fn test_monitor_get_missing() {
        let m = LifecycleMonitor::new();
        assert!(m.get("nope").is_none());
    }

    #[test]
    fn test_monitor_active_agents() {
        let mut m = LifecycleMonitor::new();
        let mut lc1 = AgentLifecycle::new("a1".into());
        lc1.start().unwrap();
        let mut lc2 = AgentLifecycle::new("a2".into());
        lc2.start().unwrap();
        lc2.stop().unwrap();
        m.register(lc1);
        m.register(lc2);
        let active = m.active_agents();
        assert_eq!(active, vec!["a1"]);
    }

    #[test]
    fn test_monitor_failed_agents() {
        let mut m = LifecycleMonitor::new();
        let mut lc1 = AgentLifecycle::new("a1".into());
        lc1.fail("bad".into());
        let lc2 = AgentLifecycle::new("a2".into());
        m.register(lc1);
        m.register(lc2);
        let failed = m.failed_agents();
        assert_eq!(failed, vec!["a1"]);
    }

    #[test]
    fn test_monitor_to_json() {
        let mut m = LifecycleMonitor::new();
        m.register(AgentLifecycle::new("a1".into()));
        let j = m.to_json();
        assert_eq!(j["agent_count"], 1);
    }

    #[test]
    fn test_monitor_default() {
        let m = LifecycleMonitor::default();
        assert_eq!(m.agent_count(), 0);
    }

    // -- Integration / full workflow tests ----------------------------------

    #[test]
    fn test_full_lifecycle_workflow() {
        let mut lc = AgentLifecycle::new("agent-42".into());
        assert_eq!(*lc.state(), AgentState::Initializing);
        lc.start().unwrap();
        assert_eq!(*lc.state(), AgentState::Running);
        lc.record_step();
        lc.record_step();
        lc.pause().unwrap();
        assert_eq!(*lc.state(), AgentState::Paused);
        lc.resume().unwrap();
        assert_eq!(*lc.state(), AgentState::Running);
        lc.record_step();
        lc.stop().unwrap();
        assert_eq!(*lc.state(), AgentState::Stopped);
        assert_eq!(lc.uptime_steps(), 3);
        assert!(lc.state().is_terminal());
        assert!(!lc.state().is_active());
    }

    #[test]
    fn test_lifecycle_transition_to_json_no_reason() {
        let t = LifecycleTransition::new(AgentState::Ready, AgentState::Running);
        let j = t.to_json();
        assert!(j["reason"].is_null());
    }

    #[test]
    fn test_lifecycle_event_names_exhaustive() {
        let events = vec![
            LifecycleEvent::Created,
            LifecycleEvent::Started,
            LifecycleEvent::Paused,
            LifecycleEvent::Resumed,
            LifecycleEvent::StepCompleted { step: 0 },
            LifecycleEvent::Error {
                message: "x".into(),
            },
            LifecycleEvent::ShuttingDown,
            LifecycleEvent::Terminated,
        ];
        let names: Vec<&str> = events.iter().map(|e| e.name()).collect();
        assert_eq!(
            names,
            vec![
                "Created",
                "Started",
                "Paused",
                "Resumed",
                "StepCompleted",
                "Error",
                "ShuttingDown",
                "Terminated"
            ]
        );
    }

    #[test]
    fn test_agent_state_can_transition_stopping_to_stopped() {
        assert!(AgentState::Stopping.can_transition_to(&AgentState::Stopped));
    }

    #[test]
    fn test_agent_state_can_transition_stopping_to_failed() {
        assert!(AgentState::Stopping.can_transition_to(&AgentState::Failed("x".into())));
    }

    #[test]
    fn test_agent_state_cannot_transition_stopping_to_running() {
        assert!(!AgentState::Stopping.can_transition_to(&AgentState::Running));
    }
}
