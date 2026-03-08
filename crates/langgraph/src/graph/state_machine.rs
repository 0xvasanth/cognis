//! Finite state machine layer for LangGraph.
//!
//! This module provides a [`StateMachine`] abstraction for modeling agent workflows
//! as explicit state transitions with optional guards and actions. It complements
//! the graph-based [`StateGraph`](super::state::StateGraph) by offering a more
//! traditional FSM model where each state has a discrete identity and transitions
//! are driven by guard predicates evaluated against a shared JSON context.
//!
//! # Key types
//!
//! - [`StateId`] — unique identifier for a state in the machine.
//! - [`Transition`] — directed edge between two states, optionally guarded and
//!   carrying an action that mutates the context on traversal.
//! - [`StateMachine`] — the runtime FSM: holds states, transitions, and a cursor.
//! - [`StateHistory`] / [`StateTransitionRecord`] — auditable log of transitions.
//! - [`StateMachineBuilder`] — fluent builder for ergonomic construction.
//! - [`StateMachineValidator`] — static validation of FSM correctness.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// StateId
// ---------------------------------------------------------------------------

/// Unique identifier for a state within a [`StateMachine`].
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StateId(String);

impl StateId {
    /// Create a new `StateId` from any string-like value.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A directed transition between two states.
///
/// Transitions may optionally carry:
/// - A **guard** predicate that must return `true` for the transition to fire.
/// - An **action** closure that mutates the shared context when the transition fires.
pub struct Transition {
    /// Source state.
    pub from: StateId,
    /// Target state.
    pub to: StateId,
    /// Optional guard predicate evaluated against the current context.
    pub guard: Option<Box<dyn Fn(&Value) -> bool + Send + Sync>>,
    /// Optional action executed (mutating context) when the transition fires.
    pub action: Option<Box<dyn Fn(&mut Value) + Send + Sync>>,
    /// Human-readable label for this transition.
    pub label: String,
}

impl Transition {
    /// Create an unguarded, action-less transition.
    pub fn new(from: StateId, to: StateId, label: impl Into<String>) -> Self {
        Self {
            from,
            to,
            guard: None,
            action: None,
            label: label.into(),
        }
    }

    /// Attach a guard predicate. The transition will only fire when the guard
    /// returns `true` for the current context.
    pub fn with_guard(mut self, guard: impl Fn(&Value) -> bool + Send + Sync + 'static) -> Self {
        self.guard = Some(Box::new(guard));
        self
    }

    /// Attach an action that mutates the context when this transition fires.
    pub fn with_action(mut self, action: impl Fn(&mut Value) + Send + Sync + 'static) -> Self {
        self.action = Some(Box::new(action));
        self
    }

    /// Evaluate whether this transition can fire given the current context.
    ///
    /// Returns `true` when there is no guard or the guard returns `true`.
    pub fn can_transition(&self, state: &Value) -> bool {
        match &self.guard {
            Some(guard) => guard(state),
            None => true,
        }
    }
}

// Debug is useful but closures are opaque.
impl fmt::Debug for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transition")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("label", &self.label)
            .field("has_guard", &self.guard.is_some())
            .field("has_action", &self.action.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------

/// A finite state machine that drives agent workflows through explicit state
/// transitions evaluated against a shared JSON context.
pub struct StateMachine {
    /// The initial state (also used by [`reset`](Self::reset)).
    initial: StateId,
    /// Current state cursor.
    current: StateId,
    /// Set of registered states.
    states: HashSet<StateId>,
    /// Transitions grouped by source state for efficient lookup.
    transitions: HashMap<StateId, Vec<Transition>>,
    /// States that are considered terminal.
    final_states: HashSet<StateId>,
}

impl StateMachine {
    /// Create a new state machine with the given initial state.
    ///
    /// The initial state is automatically added to the set of known states.
    pub fn new(initial: StateId) -> Self {
        let mut states = HashSet::new();
        states.insert(initial.clone());
        Self {
            current: initial.clone(),
            initial,
            states,
            transitions: HashMap::new(),
            final_states: HashSet::new(),
        }
    }

    /// Register a state.
    pub fn add_state(&mut self, id: StateId) {
        self.states.insert(id);
    }

    /// Register a transition.
    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions
            .entry(transition.from.clone())
            .or_default()
            .push(transition);
    }

    /// Mark a state as final (terminal).
    pub fn add_final_state(&mut self, id: StateId) {
        self.states.insert(id.clone());
        self.final_states.insert(id);
    }

    /// Return a reference to the current state.
    pub fn current(&self) -> &StateId {
        &self.current
    }

    /// Evaluate transitions from the current state and advance to the first
    /// matching target. Returns `Ok(Some(new_state))` on success,
    /// `Ok(None)` when no transition matches, or `Err` on error.
    pub fn step(&mut self, context: &mut Value) -> Result<Option<StateId>, String> {
        if self.is_finished() {
            return Ok(None);
        }

        let transitions = match self.transitions.get(&self.current) {
            Some(t) => t,
            None => return Ok(None),
        };

        // Find the first transition whose guard passes.
        let idx = transitions.iter().position(|t| t.can_transition(context));

        match idx {
            Some(i) => {
                // We need to borrow the transition immutably to read `to`,
                // then potentially call the action. Since action takes &mut Value
                // (not &mut self), this is safe through indexing.
                let to = self.transitions.get(&self.current).unwrap()[i].to.clone();

                if !self.states.contains(&to) {
                    return Err(format!("Transition target '{}' is not a registered state", to));
                }

                // Execute the action if present.
                if let Some(action) = &self.transitions.get(&self.current).unwrap()[i].action {
                    action(context);
                }

                self.current = to.clone();
                Ok(Some(to))
            }
            None => Ok(None),
        }
    }

    /// Returns `true` if the machine is in a final state.
    pub fn is_finished(&self) -> bool {
        self.final_states.contains(&self.current)
    }

    /// Run the machine until it reaches a final state or no transition matches.
    ///
    /// Returns the sequence of states visited (excluding the initial state).
    pub fn run(&mut self, context: &mut Value) -> Result<Vec<StateId>, String> {
        let mut path = Vec::new();
        loop {
            if self.is_finished() {
                break;
            }
            match self.step(context)? {
                Some(next) => path.push(next),
                None => break,
            }
        }
        Ok(path)
    }

    /// Reset the machine to the initial state.
    pub fn reset(&mut self) {
        self.current = self.initial.clone();
    }

    // -- accessors used by the validator --

    /// Return the set of registered states (for validation).
    pub fn states(&self) -> &HashSet<StateId> {
        &self.states
    }

    /// Return the initial state id.
    pub fn initial(&self) -> &StateId {
        &self.initial
    }

    /// Return the final states set.
    pub fn final_states(&self) -> &HashSet<StateId> {
        &self.final_states
    }

    /// Return the transitions map.
    pub fn transitions(&self) -> &HashMap<StateId, Vec<Transition>> {
        &self.transitions
    }
}

impl fmt::Debug for StateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateMachine")
            .field("initial", &self.initial)
            .field("current", &self.current)
            .field("states", &self.states)
            .field("final_states", &self.final_states)
            .field("transition_count", &self.transitions.values().map(|v| v.len()).sum::<usize>())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StateTransitionRecord / StateHistory
// ---------------------------------------------------------------------------

/// A single recorded transition.
#[derive(Debug, Clone)]
pub struct StateTransitionRecord {
    /// Source state name.
    pub from: String,
    /// Target state name.
    pub to: String,
    /// Snapshot of the context at the time of transition.
    pub context: Value,
    /// ISO-8601 timestamp string.
    pub timestamp: String,
    /// Zero-based step index.
    pub step: usize,
}

impl StateTransitionRecord {
    /// Serialize this record to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "from": self.from,
            "to": self.to,
            "context": self.context,
            "timestamp": self.timestamp,
            "step": self.step,
        })
    }
}

/// Auditable history of state transitions.
#[derive(Debug, Clone, Default)]
pub struct StateHistory {
    entries: Vec<StateTransitionRecord>,
}

impl StateHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a transition.
    pub fn record(&mut self, from: &StateId, to: &StateId, context_snapshot: Value) {
        let step = self.entries.len();
        let timestamp = chrono::Utc::now().to_rfc3339();
        self.entries.push(StateTransitionRecord {
            from: from.to_string(),
            to: to.to_string(),
            context: context_snapshot,
            timestamp,
            step,
        });
    }

    /// Return all recorded entries.
    pub fn entries(&self) -> &[StateTransitionRecord] {
        &self.entries
    }

    /// Return a list of state names in visitation order (from → to → to → …).
    pub fn path(&self) -> Vec<String> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let mut path = vec![self.entries[0].from.clone()];
        for entry in &self.entries {
            path.push(entry.to.clone());
        }
        path
    }

    /// Number of recorded transitions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the full history to a JSON array.
    pub fn to_json(&self) -> Value {
        Value::Array(self.entries.iter().map(|e| e.to_json()).collect())
    }
}

// ---------------------------------------------------------------------------
// StateMachineBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`StateMachine`].
pub struct StateMachineBuilder {
    initial: StateId,
    states: Vec<StateId>,
    final_states: Vec<StateId>,
    transitions: Vec<Transition>,
}

impl StateMachineBuilder {
    /// Start building a state machine with the given initial state.
    pub fn new(initial: impl Into<String>) -> Self {
        Self {
            initial: StateId::new(initial),
            states: Vec::new(),
            final_states: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Add a non-final state.
    pub fn state(mut self, name: impl Into<String>) -> Self {
        self.states.push(StateId::new(name));
        self
    }

    /// Add a final (terminal) state.
    pub fn final_state(mut self, name: impl Into<String>) -> Self {
        self.final_states.push(StateId::new(name));
        self
    }

    /// Add an unguarded transition.
    pub fn transition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        self.transitions.push(Transition::new(
            StateId::new(from),
            StateId::new(to),
            label,
        ));
        self
    }

    /// Add a guarded transition (no action).
    pub fn guarded_transition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: impl Into<String>,
        guard: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.transitions.push(
            Transition::new(StateId::new(from), StateId::new(to), label).with_guard(guard),
        );
        self
    }

    /// Consume the builder and produce a [`StateMachine`].
    ///
    /// Returns `Err` if the initial state is empty.
    pub fn build(self) -> Result<StateMachine, String> {
        if self.initial.as_str().is_empty() {
            return Err("Initial state name must not be empty".into());
        }

        let mut machine = StateMachine::new(self.initial);
        for s in self.states {
            machine.add_state(s);
        }
        for s in self.final_states {
            machine.add_final_state(s);
        }
        for t in self.transitions {
            machine.add_transition(t);
        }
        Ok(machine)
    }
}

// ---------------------------------------------------------------------------
// StateMachineValidator
// ---------------------------------------------------------------------------

/// Static validator for [`StateMachine`] correctness.
pub struct StateMachineValidator;

impl StateMachineValidator {
    /// Validate the given state machine and return a list of issues (if any).
    ///
    /// Checks performed:
    /// 1. The initial state is in the set of registered states.
    /// 2. Every transition target is a registered state.
    /// 3. At least one final state exists.
    /// 4. Every non-initial state is reachable from the initial state.
    pub fn validate(machine: &StateMachine) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // 1. Initial state registered.
        if !machine.states().contains(machine.initial()) {
            errors.push(format!(
                "Initial state '{}' is not a registered state",
                machine.initial()
            ));
        }

        // 2. Transition targets exist.
        for transitions in machine.transitions().values() {
            for t in transitions {
                if !machine.states().contains(&t.to) {
                    errors.push(format!(
                        "Transition target '{}' (from '{}') is not a registered state",
                        t.to, t.from
                    ));
                }
                if !machine.states().contains(&t.from) {
                    errors.push(format!(
                        "Transition source '{}' is not a registered state",
                        t.from
                    ));
                }
            }
        }

        // 3. At least one final state.
        if machine.final_states().is_empty() {
            errors.push("No final states defined".into());
        }

        // 4. Reachability from initial state via BFS.
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(machine.initial().clone());
        reachable.insert(machine.initial().clone());

        while let Some(state) = queue.pop_front() {
            if let Some(transitions) = machine.transitions().get(&state) {
                for t in transitions {
                    if reachable.insert(t.to.clone()) {
                        queue.push_back(t.to.clone());
                    }
                }
            }
        }

        for s in machine.states() {
            if !reachable.contains(s) {
                errors.push(format!("State '{}' is not reachable from initial state", s));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- StateId tests ------------------------------------------------------

    #[test]
    fn state_id_creation() {
        let id = StateId::new("idle");
        assert_eq!(id.as_str(), "idle");
    }

    #[test]
    fn state_id_display() {
        let id = StateId::new("running");
        assert_eq!(format!("{}", id), "running");
    }

    #[test]
    fn state_id_equality() {
        let a = StateId::new("foo");
        let b = StateId::new("foo");
        let c = StateId::new("bar");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn state_id_clone() {
        let a = StateId::new("x");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn state_id_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StateId::new("a"));
        set.insert(StateId::new("a"));
        assert_eq!(set.len(), 1);
    }

    // -- Transition tests ---------------------------------------------------

    #[test]
    fn transition_new() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go");
        assert_eq!(t.from, StateId::new("a"));
        assert_eq!(t.to, StateId::new("b"));
        assert_eq!(t.label, "go");
        assert!(t.guard.is_none());
        assert!(t.action.is_none());
    }

    #[test]
    fn transition_can_transition_no_guard() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go");
        assert!(t.can_transition(&json!({})));
    }

    #[test]
    fn transition_with_guard_pass() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go")
            .with_guard(|v| v.get("ready").and_then(|r| r.as_bool()).unwrap_or(false));
        assert!(t.can_transition(&json!({"ready": true})));
    }

    #[test]
    fn transition_with_guard_fail() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go")
            .with_guard(|v| v.get("ready").and_then(|r| r.as_bool()).unwrap_or(false));
        assert!(!t.can_transition(&json!({"ready": false})));
        assert!(!t.can_transition(&json!({})));
    }

    #[test]
    fn transition_with_action() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go")
            .with_action(|v| {
                v.as_object_mut()
                    .unwrap()
                    .insert("visited".into(), json!(true));
            });
        assert!(t.action.is_some());
    }

    #[test]
    fn transition_debug_format() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "go");
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("Transition"));
        assert!(dbg.contains("has_guard"));
    }

    // -- StateMachine basic tests -------------------------------------------

    #[test]
    fn machine_new_sets_initial() {
        let m = StateMachine::new(StateId::new("init"));
        assert_eq!(m.current(), &StateId::new("init"));
    }

    #[test]
    fn machine_add_state() {
        let mut m = StateMachine::new(StateId::new("init"));
        m.add_state(StateId::new("processing"));
        assert!(m.states().contains(&StateId::new("processing")));
    }

    #[test]
    fn machine_add_final_state() {
        let mut m = StateMachine::new(StateId::new("init"));
        m.add_final_state(StateId::new("done"));
        assert!(m.final_states().contains(&StateId::new("done")));
        assert!(m.states().contains(&StateId::new("done")));
    }

    #[test]
    fn machine_is_finished_false() {
        let m = StateMachine::new(StateId::new("init"));
        assert!(!m.is_finished());
    }

    #[test]
    fn machine_is_finished_true() {
        let mut m = StateMachine::new(StateId::new("done"));
        m.add_final_state(StateId::new("done"));
        assert!(m.is_finished());
    }

    // -- StateMachine stepping ----------------------------------------------

    #[test]
    fn machine_step_simple() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "go"));

        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, Some(StateId::new("b")));
        assert_eq!(m.current(), &StateId::new("b"));
    }

    #[test]
    fn machine_step_no_transition() {
        let mut m = StateMachine::new(StateId::new("a"));
        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn machine_step_already_finished() {
        let mut m = StateMachine::new(StateId::new("done"));
        m.add_final_state(StateId::new("done"));
        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn machine_step_guard_blocks() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_transition(
            Transition::new(StateId::new("a"), StateId::new("b"), "go")
                .with_guard(|v| v.get("go").and_then(|g| g.as_bool()).unwrap_or(false)),
        );

        let result = m.step(&mut json!({"go": false})).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn machine_step_guard_passes() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_transition(
            Transition::new(StateId::new("a"), StateId::new("b"), "go")
                .with_guard(|v| v.get("go").and_then(|g| g.as_bool()).unwrap_or(false)),
        );

        let result = m.step(&mut json!({"go": true})).unwrap();
        assert_eq!(result, Some(StateId::new("b")));
    }

    #[test]
    fn machine_step_executes_action() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_transition(
            Transition::new(StateId::new("a"), StateId::new("b"), "go").with_action(|v| {
                v.as_object_mut()
                    .unwrap()
                    .insert("count".into(), json!(1));
            }),
        );

        let mut ctx = json!({});
        m.step(&mut ctx).unwrap();
        assert_eq!(ctx["count"], 1);
    }

    #[test]
    fn machine_step_first_matching_transition_wins() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_state(StateId::new("c"));

        // First transition always matches
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "first"));
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("c"), "second"));

        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, Some(StateId::new("b")));
    }

    #[test]
    fn machine_step_skips_first_guard_fails() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_state(StateId::new("c"));

        m.add_transition(
            Transition::new(StateId::new("a"), StateId::new("b"), "guarded")
                .with_guard(|_| false),
        );
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("c"), "fallback"));

        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, Some(StateId::new("c")));
    }

    #[test]
    fn machine_step_invalid_target() {
        let mut m = StateMachine::new(StateId::new("a"));
        // Add transition to unregistered state
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("z"), "bad"));

        let result = m.step(&mut json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a registered state"));
    }

    // -- StateMachine run ---------------------------------------------------

    #[test]
    fn machine_run_linear() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_final_state(StateId::new("c"));

        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "ab"));
        m.add_transition(Transition::new(StateId::new("b"), StateId::new("c"), "bc"));

        let path = m.run(&mut json!({})).unwrap();
        let names: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        assert_eq!(names, vec!["b", "c"]);
        assert!(m.is_finished());
    }

    #[test]
    fn machine_run_stops_at_dead_end() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));

        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "ab"));
        // No transition from b, and b is not final

        let path = m.run(&mut json!({})).unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], StateId::new("b"));
        assert!(!m.is_finished());
    }

    #[test]
    fn machine_run_guard_driven() {
        let mut m = StateMachine::new(StateId::new("check"));
        m.add_state(StateId::new("retry"));
        m.add_final_state(StateId::new("done"));

        // If count >= 3, go to done; otherwise go to retry and increment
        m.add_transition(
            Transition::new(StateId::new("check"), StateId::new("done"), "finish")
                .with_guard(|v| v.get("count").and_then(|c| c.as_i64()).unwrap_or(0) >= 3),
        );
        m.add_transition(
            Transition::new(StateId::new("check"), StateId::new("retry"), "retry")
                .with_action(|v| {
                    let count = v.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
                    v.as_object_mut()
                        .unwrap()
                        .insert("count".into(), json!(count + 1));
                }),
        );
        m.add_transition(Transition::new(
            StateId::new("retry"),
            StateId::new("check"),
            "recheck",
        ));

        let mut ctx = json!({"count": 0});
        let path = m.run(&mut ctx).unwrap();
        assert!(m.is_finished());
        assert_eq!(ctx["count"], 3);
        // Path: retry, check, retry, check, retry, check, done
        let names: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            names,
            vec!["retry", "check", "retry", "check", "retry", "check", "done"]
        );
    }

    #[test]
    fn machine_run_already_at_final() {
        let mut m = StateMachine::new(StateId::new("end"));
        m.add_final_state(StateId::new("end"));

        let path = m.run(&mut json!({})).unwrap();
        assert!(path.is_empty());
        assert!(m.is_finished());
    }

    // -- StateMachine reset -------------------------------------------------

    #[test]
    fn machine_reset() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "ab"));

        m.step(&mut json!({})).unwrap();
        assert_eq!(m.current(), &StateId::new("b"));

        m.reset();
        assert_eq!(m.current(), &StateId::new("a"));
    }

    // -- StateHistory tests -------------------------------------------------

    #[test]
    fn history_new_empty() {
        let h = StateHistory::new();
        assert_eq!(h.len(), 0);
        assert!(h.is_empty());
        assert!(h.entries().is_empty());
    }

    #[test]
    fn history_record_and_retrieve() {
        let mut h = StateHistory::new();
        h.record(&StateId::new("a"), &StateId::new("b"), json!({"x": 1}));
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());

        let entry = &h.entries()[0];
        assert_eq!(entry.from, "a");
        assert_eq!(entry.to, "b");
        assert_eq!(entry.context, json!({"x": 1}));
        assert_eq!(entry.step, 0);
    }

    #[test]
    fn history_path() {
        let mut h = StateHistory::new();
        h.record(&StateId::new("a"), &StateId::new("b"), json!({}));
        h.record(&StateId::new("b"), &StateId::new("c"), json!({}));

        assert_eq!(h.path(), vec!["a", "b", "c"]);
    }

    #[test]
    fn history_path_empty() {
        let h = StateHistory::new();
        assert!(h.path().is_empty());
    }

    #[test]
    fn history_step_increments() {
        let mut h = StateHistory::new();
        h.record(&StateId::new("a"), &StateId::new("b"), json!({}));
        h.record(&StateId::new("b"), &StateId::new("c"), json!({}));
        h.record(&StateId::new("c"), &StateId::new("d"), json!({}));

        assert_eq!(h.entries()[0].step, 0);
        assert_eq!(h.entries()[1].step, 1);
        assert_eq!(h.entries()[2].step, 2);
    }

    #[test]
    fn history_to_json() {
        let mut h = StateHistory::new();
        h.record(&StateId::new("a"), &StateId::new("b"), json!({"v": 1}));
        let j = h.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
        assert_eq!(j[0]["from"], "a");
        assert_eq!(j[0]["to"], "b");
    }

    #[test]
    fn transition_record_to_json() {
        let rec = StateTransitionRecord {
            from: "x".into(),
            to: "y".into(),
            context: json!({}),
            timestamp: "2026-01-01T00:00:00Z".into(),
            step: 5,
        };
        let j = rec.to_json();
        assert_eq!(j["from"], "x");
        assert_eq!(j["to"], "y");
        assert_eq!(j["step"], 5);
        assert_eq!(j["timestamp"], "2026-01-01T00:00:00Z");
    }

    // -- StateMachineBuilder tests ------------------------------------------

    #[test]
    fn builder_simple() {
        let m = StateMachineBuilder::new("start")
            .state("middle")
            .final_state("end")
            .transition("start", "middle", "go")
            .transition("middle", "end", "finish")
            .build()
            .unwrap();

        assert_eq!(m.current(), &StateId::new("start"));
        assert!(m.states().contains(&StateId::new("middle")));
        assert!(m.final_states().contains(&StateId::new("end")));
    }

    #[test]
    fn builder_guarded_transition() {
        let mut m = StateMachineBuilder::new("start")
            .state("a")
            .final_state("b")
            .guarded_transition("start", "a", "guard_fail", |_| false)
            .transition("start", "b", "fallback")
            .build()
            .unwrap();

        let result = m.step(&mut json!({})).unwrap();
        assert_eq!(result, Some(StateId::new("b")));
    }

    #[test]
    fn builder_empty_initial_error() {
        let result = StateMachineBuilder::new("").build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_run_through() {
        let mut m = StateMachineBuilder::new("idle")
            .state("working")
            .final_state("complete")
            .transition("idle", "working", "start")
            .transition("working", "complete", "done")
            .build()
            .unwrap();

        let path = m.run(&mut json!({})).unwrap();
        let names: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        assert_eq!(names, vec!["working", "complete"]);
        assert!(m.is_finished());
    }

    // -- StateMachineValidator tests ----------------------------------------

    #[test]
    fn validator_valid_machine() {
        let m = StateMachineBuilder::new("a")
            .state("b")
            .final_state("c")
            .transition("a", "b", "ab")
            .transition("b", "c", "bc")
            .build()
            .unwrap();

        assert!(StateMachineValidator::validate(&m).is_ok());
    }

    #[test]
    fn validator_no_final_state() {
        let m = StateMachineBuilder::new("a")
            .state("b")
            .transition("a", "b", "ab")
            .build()
            .unwrap();

        let err = StateMachineValidator::validate(&m).unwrap_err();
        assert!(err.iter().any(|e| e.contains("No final states")));
    }

    #[test]
    fn validator_unreachable_state() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_state(StateId::new("orphan"));
        m.add_final_state(StateId::new("b"));
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "ab"));

        let err = StateMachineValidator::validate(&m).unwrap_err();
        assert!(err.iter().any(|e| e.contains("orphan") && e.contains("not reachable")));
    }

    #[test]
    fn validator_transition_to_unknown_state() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_final_state(StateId::new("b"));
        // Transition targets "b" which is registered, but also one to "z" which is not.
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("b"), "ab"));
        m.add_transition(Transition::new(StateId::new("a"), StateId::new("z"), "az"));

        let err = StateMachineValidator::validate(&m).unwrap_err();
        assert!(err.iter().any(|e| e.contains("'z'")));
    }

    #[test]
    fn validator_all_reachable_complex() {
        let m = StateMachineBuilder::new("start")
            .state("a")
            .state("b")
            .final_state("end")
            .transition("start", "a", "sa")
            .transition("a", "b", "ab")
            .transition("b", "end", "be")
            .build()
            .unwrap();

        assert!(StateMachineValidator::validate(&m).is_ok());
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn machine_no_transitions_at_all() {
        let mut m = StateMachine::new(StateId::new("lonely"));
        let path = m.run(&mut json!({})).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn machine_multiple_actions_accumulate() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("b"));
        m.add_final_state(StateId::new("c"));

        m.add_transition(
            Transition::new(StateId::new("a"), StateId::new("b"), "ab").with_action(|v| {
                v.as_object_mut()
                    .unwrap()
                    .insert("step1".into(), json!(true));
            }),
        );
        m.add_transition(
            Transition::new(StateId::new("b"), StateId::new("c"), "bc").with_action(|v| {
                v.as_object_mut()
                    .unwrap()
                    .insert("step2".into(), json!(true));
            }),
        );

        let mut ctx = json!({});
        m.run(&mut ctx).unwrap();
        assert_eq!(ctx["step1"], true);
        assert_eq!(ctx["step2"], true);
    }

    #[test]
    fn machine_debug_format() {
        let m = StateMachine::new(StateId::new("init"));
        let dbg = format!("{:?}", m);
        assert!(dbg.contains("StateMachine"));
        assert!(dbg.contains("init"));
    }

    #[test]
    fn history_multiple_records() {
        let mut h = StateHistory::new();
        for i in 0..10 {
            h.record(
                &StateId::new(format!("s{}", i)),
                &StateId::new(format!("s{}", i + 1)),
                json!({"i": i}),
            );
        }
        assert_eq!(h.len(), 10);
        assert_eq!(h.path().len(), 11); // 10 transitions = 11 states
    }

    #[test]
    fn builder_chaining_preserves_order() {
        let m = StateMachineBuilder::new("a")
            .state("b")
            .state("c")
            .final_state("d")
            .transition("a", "b", "1")
            .transition("b", "c", "2")
            .transition("c", "d", "3")
            .build()
            .unwrap();

        assert!(m.states().contains(&StateId::new("b")));
        assert!(m.states().contains(&StateId::new("c")));
        assert!(m.states().contains(&StateId::new("d")));
    }

    #[test]
    fn machine_run_with_context_mutation() {
        let _m = StateMachineBuilder::new("init")
            .state("process")
            .final_state("done")
            .transition("init", "process", "start")
            .transition("process", "done", "complete")
            .build()
            .unwrap();

        // Add action manually
        let mut m2 = StateMachine::new(StateId::new("init"));
        m2.add_state(StateId::new("process"));
        m2.add_final_state(StateId::new("done"));
        m2.add_transition(
            Transition::new(StateId::new("init"), StateId::new("process"), "start")
                .with_action(|v| {
                    v.as_object_mut()
                        .unwrap()
                        .insert("started".into(), json!(true));
                }),
        );
        m2.add_transition(
            Transition::new(StateId::new("process"), StateId::new("done"), "complete")
                .with_action(|v| {
                    v.as_object_mut()
                        .unwrap()
                        .insert("completed".into(), json!(true));
                }),
        );

        let mut ctx = json!({});
        m2.run(&mut ctx).unwrap();
        assert_eq!(ctx["started"], true);
        assert_eq!(ctx["completed"], true);
    }

    #[test]
    fn validator_multiple_errors() {
        let mut m = StateMachine::new(StateId::new("a"));
        m.add_state(StateId::new("orphan"));
        // No final states, orphan is unreachable
        let err = StateMachineValidator::validate(&m).unwrap_err();
        assert!(err.len() >= 2); // at least "no final" + "unreachable"
    }

    #[test]
    fn machine_reset_after_run() {
        let mut m = StateMachineBuilder::new("a")
            .state("b")
            .final_state("c")
            .transition("a", "b", "ab")
            .transition("b", "c", "bc")
            .build()
            .unwrap();

        m.run(&mut json!({})).unwrap();
        assert!(m.is_finished());

        m.reset();
        assert_eq!(m.current(), &StateId::new("a"));
        assert!(!m.is_finished());

        // Can run again
        let path = m.run(&mut json!({})).unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn state_id_from_string() {
        let name = String::from("dynamic");
        let id = StateId::new(name);
        assert_eq!(id.as_str(), "dynamic");
    }

    #[test]
    fn transition_guard_and_action_combined() {
        let t = Transition::new(StateId::new("a"), StateId::new("b"), "combo")
            .with_guard(|v| v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false))
            .with_action(|v| {
                v.as_object_mut()
                    .unwrap()
                    .insert("acted".into(), json!(true));
            });

        assert!(!t.can_transition(&json!({"ok": false})));
        assert!(t.can_transition(&json!({"ok": true})));
        assert!(t.guard.is_some());
        assert!(t.action.is_some());
    }
}
