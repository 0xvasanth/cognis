//! Standard agent state. Use this for the default ReAct flow; for
//! custom agents, define your own state struct and use the raw graph.

#![allow(missing_docs, clippy::assign_op_pattern)]

use cognis2_core::{Extensions, Message};
use cognis2_graph::GraphState;

/// The default agent state: append-accumulated messages, an iteration
/// counter, and a typed `extras` map for plugin/middleware data.
#[derive(GraphState, Clone, Default, Debug)]
pub struct AgentState {
    /// Conversation messages, accumulated across supersteps.
    #[reducer(append)]
    pub messages: Vec<Message>,

    /// Number of `think` calls executed (incremented by ThinkNode).
    #[reducer(add)]
    pub iterations: u32,

    /// Plugin/middleware payloads. Auto-skipped by the GraphState derive.
    pub extras: Extensions,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis2_graph::GraphState as GraphStateTrait;

    #[test]
    fn append_messages() {
        let mut s = AgentState::default();
        s.apply(AgentStateUpdate {
            messages: vec![Message::human("hi"), Message::ai("hello")],
            iterations: 0,
        });
        assert_eq!(s.messages.len(), 2);
        assert_eq!(s.iterations, 0);
    }

    #[test]
    fn iterations_add() {
        let mut s = AgentState::default();
        s.apply(AgentStateUpdate {
            messages: vec![],
            iterations: 1,
        });
        s.apply(AgentStateUpdate {
            messages: vec![],
            iterations: 2,
        });
        assert_eq!(s.iterations, 3);
    }
}
