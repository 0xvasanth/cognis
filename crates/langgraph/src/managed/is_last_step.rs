//! Managed values for step tracking during graph execution.
//!
//! Provides `IsLastStepManager` and `RemainingStepsManager`, which compute
//! whether the current step is the final one and how many steps remain,
//! respectively.

use super::base::{ManagedValue, PregelScratchpad};

/// Managed value that returns `true` when the current step is the last step.
///
/// This is the Rust equivalent of Python's `IsLastStepManager`. It computes
/// `step == stop - 1`, meaning the execution is on its final allowed step.
pub struct IsLastStepManager;

impl ManagedValue for IsLastStepManager {
    type Value = bool;

    fn get(scratchpad: &PregelScratchpad) -> bool {
        scratchpad.stop > 0 && scratchpad.step == scratchpad.stop - 1
    }
}

/// Managed value that returns the number of remaining steps.
///
/// This is the Rust equivalent of Python's `RemainingStepsManager`. It computes
/// `stop - step`, representing how many steps can still be executed.
pub struct RemainingStepsManager;

impl ManagedValue for RemainingStepsManager {
    type Value = usize;

    fn get(scratchpad: &PregelScratchpad) -> usize {
        scratchpad.stop.saturating_sub(scratchpad.step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_last_step_true() {
        let scratchpad = PregelScratchpad::new(9, 10);
        assert!(IsLastStepManager::get(&scratchpad));
    }

    #[test]
    fn test_is_last_step_false() {
        let scratchpad = PregelScratchpad::new(0, 10);
        assert!(!IsLastStepManager::get(&scratchpad));

        let scratchpad = PregelScratchpad::new(5, 10);
        assert!(!IsLastStepManager::get(&scratchpad));
    }

    #[test]
    fn test_is_last_step_single_step() {
        // Only one step: step=0, stop=1 => last step
        let scratchpad = PregelScratchpad::new(0, 1);
        assert!(IsLastStepManager::get(&scratchpad));
    }

    #[test]
    fn test_is_last_step_zero_stop() {
        // Edge case: stop=0 means no steps allowed
        let scratchpad = PregelScratchpad::new(0, 0);
        assert!(!IsLastStepManager::get(&scratchpad));
    }

    #[test]
    fn test_remaining_steps_basic() {
        let scratchpad = PregelScratchpad::new(3, 10);
        assert_eq!(RemainingStepsManager::get(&scratchpad), 7);
    }

    #[test]
    fn test_remaining_steps_at_start() {
        let scratchpad = PregelScratchpad::new(0, 10);
        assert_eq!(RemainingStepsManager::get(&scratchpad), 10);
    }

    #[test]
    fn test_remaining_steps_at_last() {
        let scratchpad = PregelScratchpad::new(9, 10);
        assert_eq!(RemainingStepsManager::get(&scratchpad), 1);
    }

    #[test]
    fn test_remaining_steps_at_stop() {
        let scratchpad = PregelScratchpad::new(10, 10);
        assert_eq!(RemainingStepsManager::get(&scratchpad), 0);
    }

    #[test]
    fn test_remaining_steps_saturating() {
        // step > stop should not underflow
        let scratchpad = PregelScratchpad::new(15, 10);
        assert_eq!(RemainingStepsManager::get(&scratchpad), 0);
    }

    #[test]
    fn test_is_last_step_and_remaining_consistency() {
        // When is_last_step is true, remaining should be 1
        for stop in 1..20 {
            let step = stop - 1;
            let scratchpad = PregelScratchpad::new(step, stop);
            assert!(IsLastStepManager::get(&scratchpad));
            assert_eq!(RemainingStepsManager::get(&scratchpad), 1);
        }
    }
}
