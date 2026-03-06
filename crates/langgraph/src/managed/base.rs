//! Base traits and types for managed values.
//!
//! Managed values are computed dynamically from the execution context (scratchpad)
//! rather than being stored directly in the graph state. They provide runtime
//! information such as whether the current step is the last step, or how many
//! steps remain.

use serde_json::Value;
use std::collections::HashMap;

/// Execution context for the Pregel engine, tracking step progress.
///
/// This is analogous to Python's `PregelScratchpad` and provides the runtime
/// information that managed values are computed from.
#[derive(Debug, Clone)]
pub struct PregelScratchpad {
    /// The current step number (0-indexed).
    pub step: usize,
    /// The maximum number of steps allowed (exclusive upper bound).
    pub stop: usize,
    /// Counter for tracking call operations within a step.
    pub call_counter: usize,
    /// Counter for tracking interrupt operations within a step.
    pub interrupt_counter: usize,
    /// Counter for tracking subgraph invocations.
    pub subgraph_counter: usize,
    /// Resume values for interrupt resumption.
    pub resume: Vec<serde_json::Value>,
}

impl PregelScratchpad {
    /// Create a new scratchpad with the given step and stop values.
    pub fn new(step: usize, stop: usize) -> Self {
        Self {
            step,
            stop,
            call_counter: 0,
            interrupt_counter: 0,
            subgraph_counter: 0,
            resume: Vec::new(),
        }
    }

    /// Increment and return the call counter.
    pub fn next_call_id(&mut self) -> usize {
        let id = self.call_counter;
        self.call_counter += 1;
        id
    }

    /// Increment and return the interrupt counter.
    pub fn next_interrupt_id(&mut self) -> usize {
        let id = self.interrupt_counter;
        self.interrupt_counter += 1;
        id
    }

    /// Increment and return the subgraph counter.
    pub fn next_subgraph_id(&mut self) -> usize {
        let id = self.subgraph_counter;
        self.subgraph_counter += 1;
        id
    }

    /// Get the resume value at the current interrupt counter position, if any.
    pub fn get_resume_value(&self) -> Option<&serde_json::Value> {
        self.resume.get(self.interrupt_counter)
    }
}

/// A managed value that is computed from the execution context.
///
/// Managed values are not stored in the graph state directly; instead,
/// they are derived from the `PregelScratchpad` at runtime. This trait
/// is the Rust equivalent of Python's `ManagedValue` abstract base class.
pub trait ManagedValue: Send + Sync {
    /// The output type produced by this managed value.
    type Value;

    /// Compute the managed value from the current scratchpad state.
    fn get(scratchpad: &PregelScratchpad) -> Self::Value;
}

/// A specification for a managed value, used to identify and create managed values.
///
/// This is the Rust equivalent of Python's `ManagedValueSpec` (which is a class reference).
/// In Rust, we use an enum to represent the different kinds of managed values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManagedValueSpec {
    /// Computes whether the current step is the last step.
    IsLastStep,
    /// Computes how many steps remain.
    RemainingSteps,
}

impl ManagedValueSpec {
    /// Resolve this spec to a concrete `serde_json::Value` given a scratchpad.
    pub fn resolve(&self, scratchpad: &PregelScratchpad) -> Value {
        match self {
            ManagedValueSpec::IsLastStep => {
                Value::Bool(super::is_last_step::IsLastStepManager::get(scratchpad))
            }
            ManagedValueSpec::RemainingSteps => {
                let remaining =
                    super::is_last_step::RemainingStepsManager::get(scratchpad) as i64;
                Value::Number(remaining.into())
            }
        }
    }
}

/// Check whether a string key corresponds to a known managed value spec.
pub fn is_managed_value(key: &str) -> Option<ManagedValueSpec> {
    match key {
        "is_last_step" => Some(ManagedValueSpec::IsLastStep),
        "remaining_steps" => Some(ManagedValueSpec::RemainingSteps),
        _ => None,
    }
}

/// A mapping from channel names to managed value specs.
pub type ManagedValueMapping = HashMap<String, ManagedValueSpec>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pregel_scratchpad_new() {
        let scratchpad = PregelScratchpad::new(3, 10);
        assert_eq!(scratchpad.step, 3);
        assert_eq!(scratchpad.stop, 10);
    }

    #[test]
    fn test_managed_value_spec_resolve_is_last_step() {
        let spec = ManagedValueSpec::IsLastStep;

        // Not last step
        let scratchpad = PregelScratchpad::new(3, 10);
        assert_eq!(spec.resolve(&scratchpad), Value::Bool(false));

        // Last step (step == stop - 1)
        let scratchpad = PregelScratchpad::new(9, 10);
        assert_eq!(spec.resolve(&scratchpad), Value::Bool(true));
    }

    #[test]
    fn test_managed_value_spec_resolve_remaining_steps() {
        let spec = ManagedValueSpec::RemainingSteps;

        let scratchpad = PregelScratchpad::new(3, 10);
        assert_eq!(spec.resolve(&scratchpad), Value::Number(7.into()));

        let scratchpad = PregelScratchpad::new(9, 10);
        assert_eq!(spec.resolve(&scratchpad), Value::Number(1.into()));

        let scratchpad = PregelScratchpad::new(0, 10);
        assert_eq!(spec.resolve(&scratchpad), Value::Number(10.into()));
    }

    #[test]
    fn test_is_managed_value() {
        assert_eq!(
            is_managed_value("is_last_step"),
            Some(ManagedValueSpec::IsLastStep)
        );
        assert_eq!(
            is_managed_value("remaining_steps"),
            Some(ManagedValueSpec::RemainingSteps)
        );
        assert_eq!(is_managed_value("unknown"), None);
    }

    #[test]
    fn test_managed_value_spec_eq_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ManagedValueSpec::IsLastStep);
        set.insert(ManagedValueSpec::RemainingSteps);
        set.insert(ManagedValueSpec::IsLastStep); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_scratchpad_counters() {
        let mut s = PregelScratchpad::new(0, 10);
        assert_eq!(s.next_call_id(), 0);
        assert_eq!(s.next_call_id(), 1);
        assert_eq!(s.next_interrupt_id(), 0);
        assert_eq!(s.next_subgraph_id(), 0);
        assert_eq!(s.next_subgraph_id(), 1);
    }

    #[test]
    fn test_scratchpad_resume() {
        let mut s = PregelScratchpad::new(0, 10);
        s.resume = vec![serde_json::json!("value1"), serde_json::json!("value2")];
        assert_eq!(s.get_resume_value(), Some(&serde_json::json!("value1")));
        s.next_interrupt_id();
        assert_eq!(s.get_resume_value(), Some(&serde_json::json!("value2")));
        s.next_interrupt_id();
        assert_eq!(s.get_resume_value(), None);
    }

    #[test]
    fn test_managed_value_mapping() {
        let mut mapping = ManagedValueMapping::new();
        mapping.insert("is_last_step".to_string(), ManagedValueSpec::IsLastStep);
        mapping.insert(
            "remaining_steps".to_string(),
            ManagedValueSpec::RemainingSteps,
        );

        let scratchpad = PregelScratchpad::new(4, 5);
        for (key, spec) in &mapping {
            let val = spec.resolve(&scratchpad);
            match key.as_str() {
                "is_last_step" => assert_eq!(val, Value::Bool(true)),
                "remaining_steps" => assert_eq!(val, Value::Number(1.into())),
                _ => panic!("unexpected key"),
            }
        }
    }
}
