//! Stub — full impl in Task 6.
use cognis2_core::Result;
use crate::builder::Graph;
use crate::state::GraphState;

pub(crate) fn validate<S: GraphState>(_g: &Graph<S>) -> Result<()> {
    Ok(())
}
