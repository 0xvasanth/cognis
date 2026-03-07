//! Managed values for LangGraph.
//!
//! Managed values are dynamically computed from the Pregel execution context
//! rather than being stored in the graph state. They provide runtime information
//! like whether the current step is the last step or how many steps remain.

pub mod base;
pub mod is_last_step;
pub mod shared_value;

pub use base::{
    is_managed_value, ManagedValue, ManagedValueMapping, ManagedValueSpec, PregelScratchpad,
};
pub use is_last_step::{IsLastStepManager, RemainingStepsManager};
pub use shared_value::{
    SharedAccumulator, SharedCounter, SharedMap, SharedValue, SharedValueConfig, SharedValueError,
    ValueHistory,
};
