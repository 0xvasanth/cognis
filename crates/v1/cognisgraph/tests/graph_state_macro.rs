use cognis_macros::GraphState;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, GraphState)]
struct SimpleState {
    /// Messages in the conversation
    #[reducer(append)]
    messages: Vec<Value>,

    /// Current status
    #[reducer(last_value)]
    status: String,

    /// Accumulated count
    #[reducer(add)]
    count: i64,
}

#[test]
fn test_graph_state_generates_schema() {
    let schema = SimpleState::graph_state();
    assert_eq!(schema.fields.len(), 3);
    assert!(schema.fields.contains_key("messages"));
    assert!(schema.fields.contains_key("status"));
    assert!(schema.fields.contains_key("count"));
}

#[test]
fn test_graph_state_field_descriptions() {
    let schema = SimpleState::graph_state();
    let field = schema.fields.get("messages").unwrap();
    assert_eq!(
        field.description.as_deref(),
        Some("Messages in the conversation")
    );
}

#[test]
fn test_graph_state_append_reducer() {
    let schema = SimpleState::graph_state();
    let field = schema.fields.get("messages").unwrap();
    let result = (field.reducer)(&json!(["hello"]), &json!(["world"]));
    assert_eq!(result, json!(["hello", "world"]));
}

#[test]
fn test_graph_state_last_value_reducer() {
    let schema = SimpleState::graph_state();
    let field = schema.fields.get("status").unwrap();
    let result = (field.reducer)(&json!("old"), &json!("new"));
    assert_eq!(result, json!("new"));
}

#[test]
fn test_graph_state_add_reducer() {
    let schema = SimpleState::graph_state();
    let field = schema.fields.get("count").unwrap();
    let result = (field.reducer)(&json!(10), &json!(5));
    assert_eq!(result, json!(15));
}

#[derive(Debug, Clone, Serialize, Deserialize, GraphState)]
struct DefaultReducerState {
    name: String,
}

#[test]
fn test_default_reducer_is_last_value() {
    let schema = DefaultReducerState::graph_state();
    let field = schema.fields.get("name").unwrap();
    let result = (field.reducer)(&json!("old"), &json!("new"));
    assert_eq!(result, json!("new"));
}

#[derive(Debug, Clone, Serialize, Deserialize, GraphState)]
struct MergeState {
    /// Config object
    #[reducer(merge)]
    config: Value,
}

#[test]
fn test_merge_reducer() {
    let schema = MergeState::graph_state();
    let field = schema.fields.get("config").unwrap();
    let current = json!({"a": 1, "b": 2});
    let update = json!({"b": 99, "c": 3});
    let result = (field.reducer)(&current, &update);
    assert_eq!(result, json!({"a": 1, "b": 99, "c": 3}));
}
