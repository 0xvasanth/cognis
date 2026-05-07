//! Codegen-shape test for #[derive(GraphStateV2)]. We stub the
//! cognis_graph::GraphState trait locally and a __merge_json helper so the
//! macro's output compiles in isolation. Plan #2 wires the real types and
//! adds end-to-end semantic tests.

use cognis_macros::GraphStateV2;
use serde::{Deserialize, Serialize};

mod cognis_graph {
    pub trait GraphState {
        type Update;
        fn apply(&mut self, update: Self::Update);
    }

    pub fn __merge_json(target: &mut serde_json::Value, source: serde_json::Value) {
        // Trivial stub: overwrite. The real impl in cognis-graph deep-merges.
        *target = source;
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize, GraphStateV2)]
#[graph_state(crate_path = "crate::cognis_graph")]
pub struct AgentState {
    #[reducer(append)]
    pub messages: Vec<String>,

    #[reducer(add)]
    pub iterations: u32,

    #[reducer(last)]
    pub last_intent: Option<String>,

    #[reducer(merge)]
    pub metadata: serde_json::Value,

    pub extras: std::collections::HashMap<String, serde_json::Value>,
}

#[test]
fn update_struct_exists_and_defaults() {
    let upd: AgentStateUpdate = AgentStateUpdate::default();
    assert!(upd.messages.is_empty());
    assert_eq!(upd.iterations, 0);
    assert!(upd.last_intent.is_none());
    assert!(upd.metadata.is_none());
    // No `extras` field — auto-skipped.
}

#[test]
fn apply_append_concatenates() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        messages: vec!["a".into()],
        ..Default::default()
    };
    s.apply(AgentStateUpdate {
        messages: vec!["b".into(), "c".into()],
        ..Default::default()
    });
    assert_eq!(s.messages, vec!["a", "b", "c"]);
}

#[test]
fn apply_add_increments() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        iterations: 5,
        ..Default::default()
    };
    s.apply(AgentStateUpdate {
        iterations: 3,
        ..Default::default()
    });
    assert_eq!(s.iterations, 8);
}

#[test]
fn apply_last_overwrites_when_some() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        last_intent: Some("old".into()),
        ..Default::default()
    };
    s.apply(AgentStateUpdate {
        last_intent: Some("new".into()),
        ..Default::default()
    });
    assert_eq!(s.last_intent.as_deref(), Some("new"));
}

#[test]
fn apply_last_keeps_existing_when_none() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        last_intent: Some("keep".into()),
        ..Default::default()
    };
    s.apply(AgentStateUpdate {
        last_intent: None,
        ..Default::default()
    });
    assert_eq!(s.last_intent.as_deref(), Some("keep"));
}

#[test]
fn apply_last_cannot_unset_option_field() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        last_intent: Some("keep".into()),
        ..Default::default()
    };
    // Documented limitation: passing None as the update doesn't clear
    // the field. Users who need clear-to-None semantics use Reducer::Custom.
    s.apply(AgentStateUpdate {
        last_intent: None,
        ..Default::default()
    });
    assert_eq!(s.last_intent.as_deref(), Some("keep"));
}

#[test]
fn apply_merge_replaces_via_helper() {
    use cognis_graph::GraphState;
    let mut s = AgentState {
        metadata: serde_json::json!({"a": 1}),
        ..Default::default()
    };
    s.apply(AgentStateUpdate {
        metadata: Some(serde_json::json!({"b": 2})),
        ..Default::default()
    });
    // Stub helper overwrites — real cognis_graph deep-merges in Plan #2.
    assert_eq!(s.metadata, serde_json::json!({"b": 2}));
}
