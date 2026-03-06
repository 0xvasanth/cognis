//! Message graph — a convenience wrapper for message-based state graphs.
//!
//! Provides a factory function that creates a [`StateGraph`](super::state::StateGraph)
//! pre-configured for message passing workflows where state is a list of messages
//! that accumulates over time.
//!
//! Also provides the [`add_messages`] reducer function which merges two message
//! lists with ID-based deduplication and `RemoveMessage` support.

use serde_json::Value;
use std::collections::{HashMap, HashSet};

use super::state::StateGraph;

/// Merge two message lists with ID-based deduplication.
///
/// Messages are matched by their `"id"` field. If a message in `right` has the same
/// id as one in `left`, it replaces the original. If a message has `type: "remove"`
/// (or `type: "RemoveMessage"`), the message with the matching id is removed from the
/// result.
///
/// Messages without ids are always appended.
///
/// # Arguments
///
/// * `left` — The existing message list (a JSON array, or a single message value).
/// * `right` — The new messages to merge in (a JSON array, or a single message value).
///
/// # Returns
///
/// A `Value::Array` containing the merged messages.
///
/// # Examples
///
/// ```
/// use langgraph::graph::message::add_messages;
/// use serde_json::json;
///
/// let left = json!([
///     {"type": "human", "content": "hi", "id": "1"}
/// ]);
/// let right = json!([
///     {"type": "ai", "content": "hello!", "id": "2"}
/// ]);
/// let result = add_messages(&left, &right);
/// assert_eq!(result.as_array().unwrap().len(), 2);
/// ```
pub fn add_messages(left: &Value, right: &Value) -> Value {
    let left_msgs = match left {
        Value::Array(arr) => arr.clone(),
        _ => vec![left.clone()],
    };
    let right_msgs = match right {
        Value::Array(arr) => arr.clone(),
        _ => vec![right.clone()],
    };

    // Collect IDs to remove
    let mut ids_to_remove: HashSet<String> = HashSet::new();
    for msg in &right_msgs {
        if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
            if msg_type == "remove" || msg_type == "RemoveMessage" {
                if let Some(id) = msg.get("id").and_then(|id| id.as_str()) {
                    ids_to_remove.insert(id.to_string());
                }
            }
        }
    }

    // Build result: start with left, filtering out removed messages
    let mut result: Vec<Value> = Vec::new();
    let mut left_ids: HashMap<String, usize> = HashMap::new();

    for msg in left_msgs {
        let id = msg
            .get("id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string());
        if let Some(ref id_str) = id {
            if ids_to_remove.contains(id_str) {
                continue; // Skip removed messages
            }
            left_ids.insert(id_str.clone(), result.len());
        }
        result.push(msg);
    }

    // Add right messages, replacing by ID if one already exists in left
    for msg in right_msgs {
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if msg_type == "remove" || msg_type == "RemoveMessage" {
            continue; // Skip remove markers — they have already been applied
        }

        let id = msg
            .get("id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string());
        if let Some(ref id_str) = id {
            if let Some(&idx) = left_ids.get(id_str) {
                result[idx] = msg; // Replace existing
                continue;
            }
        }
        result.push(msg);
    }

    Value::Array(result)
}

/// Create a state graph pre-configured for message passing.
///
/// The resulting graph uses [`add_messages`] as the reducer for the `"messages"`
/// key in the state. Node outputs that include a `"messages"` key will have
/// their values merged using ID-based deduplication.
///
/// # Example
///
/// ```rust
/// use langgraph::graph::message::message_graph;
///
/// let graph = message_graph();
/// // Add nodes and edges, then compile.
/// ```
pub fn message_graph() -> StateGraph {
    StateGraph::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::state::AsyncNodeAction;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn test_add_messages_append() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"}
        ]);
        let right = json!([
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_replace_by_id() {
        let left = json!([
            {"type": "human", "content": "old", "id": "1"},
            {"type": "ai", "content": "response", "id": "2"}
        ]);
        let right = json!([
            {"type": "human", "content": "updated", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"], "updated");
        assert_eq!(arr[1]["content"], "response");
    }

    #[test]
    fn test_add_messages_remove() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"},
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let right = json!([
            {"type": "remove", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "2");
    }

    #[test]
    fn test_add_messages_no_id_always_appends() {
        let left = json!([
            {"type": "human", "content": "hi"}
        ]);
        let right = json!([
            {"type": "ai", "content": "hello!"}
        ]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_empty_left() {
        let left = json!([]);
        let right = json!([{"type": "human", "content": "hi", "id": "1"}]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_add_messages_empty_right() {
        let left = json!([{"type": "human", "content": "hi", "id": "1"}]);
        let right = json!([]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_add_messages_mixed_operations() {
        let left = json!([
            {"type": "human", "content": "msg1", "id": "1"},
            {"type": "ai", "content": "msg2", "id": "2"},
            {"type": "human", "content": "msg3", "id": "3"}
        ]);
        let right = json!([
            {"type": "remove", "id": "2"},
            {"type": "human", "content": "msg1-updated", "id": "1"},
            {"type": "ai", "content": "new-msg", "id": "4"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3); // msg1 (updated) + msg3 + new-msg (msg2 removed)
        assert_eq!(arr[0]["content"], "msg1-updated");
        assert_eq!(arr[1]["content"], "msg3");
        assert_eq!(arr[2]["content"], "new-msg");
    }

    #[test]
    fn test_add_messages_single_value_not_array() {
        let left = json!({"type": "human", "content": "hi", "id": "1"});
        let right = json!({"type": "ai", "content": "hello!", "id": "2"});
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_remove_message_variant() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"},
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let right = json!([
            {"type": "RemoveMessage", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "2");
    }

    #[tokio::test]
    async fn test_message_graph_basic() {
        let graph = message_graph()
            .add_node(
                "greeter",
                Arc::new(|state: Value| -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, crate::errors::LangGraphError>> + Send>> {
                    Box::pin(async move {
                        let name = state
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("world");
                        Ok(json!({"greeting": format!("Hello, {name}!")}))
                    })
                }) as AsyncNodeAction,
            )
            .set_entry_point("greeter")
            .set_finish_point("greeter")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"name": "Alice"})).await.unwrap();
        assert_eq!(result["greeting"], json!("Hello, Alice!"));
    }
}
