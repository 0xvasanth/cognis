//! UI message types and reducer for rendering UI components in LangGraph.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// A UI message for rendering UI components in LangGraph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UIMessage {
    /// Literal type indicator: always "ui".
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Unique identifier for the UI message.
    pub id: String,
    /// Name of the UI component to render.
    pub name: String,
    /// Properties to pass to the UI component.
    pub props: HashMap<String, Value>,
    /// Additional metadata about the UI message.
    pub metadata: HashMap<String, Value>,
}

impl UIMessage {
    /// Create a new UIMessage with the given id, name, and props.
    /// Sets type to "ui" and metadata to empty.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        props: HashMap<String, Value>,
    ) -> Self {
        Self {
            msg_type: "ui".to_string(),
            id: id.into(),
            name: name.into(),
            props,
            metadata: HashMap::new(),
        }
    }
}

/// A message to remove a UI component.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoveUIMessage {
    /// Literal type indicator: always "remove-ui".
    #[serde(rename = "type")]
    pub msg_type: String,
    /// ID of the UI message to remove.
    pub id: String,
}

impl RemoveUIMessage {
    /// Create a new RemoveUIMessage with the given id.
    /// Sets type to "remove-ui".
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            msg_type: "remove-ui".to_string(),
            id: id.into(),
        }
    }
}

/// An enum representing any UI message type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AnyUIMessage {
    /// A UI message for rendering a component.
    /// Note: UI variant comes first because it has more required fields,
    /// ensuring correct untagged deserialization.
    UI(UIMessage),
    /// A message to remove a UI component.
    Remove(RemoveUIMessage),
}

impl AnyUIMessage {
    /// Returns the ID of the UI message.
    pub fn id(&self) -> &str {
        match self {
            AnyUIMessage::UI(msg) => &msg.id,
            AnyUIMessage::Remove(msg) => &msg.id,
        }
    }

    /// Returns true if this is a remove-ui message.
    pub fn is_remove(&self) -> bool {
        matches!(self, AnyUIMessage::Remove(_))
    }
}

/// Merge two lists of UI messages, supporting removing UI messages.
///
/// When a `remove-ui` message is encountered, it removes any
/// UI message with the matching ID from the current state.
/// If `merge` metadata is true on a UI message, its props merge with existing.
pub fn ui_message_reducer(
    left: &[AnyUIMessage],
    right: &[AnyUIMessage],
) -> Result<Vec<AnyUIMessage>, crate::errors::LangGraphError> {
    let mut merged: Vec<AnyUIMessage> = left.to_vec();
    let mut merged_by_id: HashMap<String, usize> = HashMap::new();

    for (idx, msg) in merged.iter().enumerate() {
        merged_by_id.insert(msg.id().to_string(), idx);
    }

    let mut ids_to_remove: HashSet<String> = HashSet::new();

    for msg in right {
        let id = msg.id().to_string();

        if let Some(&existing_idx) = merged_by_id.get(&id) {
            if msg.is_remove() {
                ids_to_remove.insert(id);
            } else {
                ids_to_remove.remove(&id);

                if let AnyUIMessage::UI(new_ui) = msg {
                    // Check if metadata.merge is true
                    let should_merge = new_ui
                        .metadata
                        .get("merge")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if should_merge {
                        if let AnyUIMessage::UI(existing_ui) = &merged[existing_idx] {
                            let mut merged_props = existing_ui.props.clone();
                            for (k, v) in &new_ui.props {
                                merged_props.insert(k.clone(), v.clone());
                            }
                            let mut merged_msg = new_ui.clone();
                            merged_msg.props = merged_props;
                            merged[existing_idx] = AnyUIMessage::UI(merged_msg);
                        } else {
                            merged[existing_idx] = msg.clone();
                        }
                    } else {
                        merged[existing_idx] = msg.clone();
                    }
                }
            }
        } else if msg.is_remove() {
            return Err(crate::errors::LangGraphError::InvalidUpdateError(
                "Attempting to delete a UI message with an ID that doesn't exist".to_string(),
            ));
        } else {
            merged_by_id.insert(id, merged.len());
            merged.push(msg.clone());
        }
    }

    merged.retain(|msg| !ids_to_remove.contains(msg.id()));

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ui_message_new() {
        let props = HashMap::from([("key".to_string(), json!("value"))]);
        let msg = UIMessage::new("id1", "button", props.clone());
        assert_eq!(msg.msg_type, "ui");
        assert_eq!(msg.id, "id1");
        assert_eq!(msg.name, "button");
        assert_eq!(msg.props, props);
        assert!(msg.metadata.is_empty());
    }

    #[test]
    fn test_remove_ui_message_new() {
        let msg = RemoveUIMessage::new("id1");
        assert_eq!(msg.msg_type, "remove-ui");
        assert_eq!(msg.id, "id1");
    }

    #[test]
    fn test_reducer_append() {
        let left = vec![AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::new(),
        ))];
        let right = vec![AnyUIMessage::UI(UIMessage::new(
            "id2",
            "input",
            HashMap::new(),
        ))];
        let result = ui_message_reducer(&left, &right).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id(), "id1");
        assert_eq!(result[1].id(), "id2");
    }

    #[test]
    fn test_reducer_replace_by_id() {
        let left = vec![AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::from([("label".to_string(), json!("old"))]),
        ))];
        let right = vec![AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::from([("label".to_string(), json!("new"))]),
        ))];
        let result = ui_message_reducer(&left, &right).unwrap();
        assert_eq!(result.len(), 1);
        if let AnyUIMessage::UI(ui) = &result[0] {
            assert_eq!(ui.props.get("label").unwrap(), &json!("new"));
        } else {
            panic!("Expected UI message");
        }
    }

    #[test]
    fn test_reducer_remove_by_id() {
        let left = vec![
            AnyUIMessage::UI(UIMessage::new("id1", "button", HashMap::new())),
            AnyUIMessage::UI(UIMessage::new("id2", "input", HashMap::new())),
        ];
        let right = vec![AnyUIMessage::Remove(RemoveUIMessage::new("id1"))];
        let result = ui_message_reducer(&left, &right).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id(), "id2");
    }

    #[test]
    fn test_reducer_remove_nonexistent_errors() {
        let left = vec![AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::new(),
        ))];
        let right = vec![AnyUIMessage::Remove(RemoveUIMessage::new("id999"))];
        let result = ui_message_reducer(&left, &right);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Attempting to delete a UI message with an ID that doesn't exist"));
    }

    #[test]
    fn test_reducer_merge_props() {
        let left = vec![AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::from([
                ("label".to_string(), json!("click")),
                ("color".to_string(), json!("red")),
            ]),
        ))];

        let mut merge_msg = UIMessage::new(
            "id1",
            "button",
            HashMap::from([("color".to_string(), json!("blue"))]),
        );
        merge_msg.metadata.insert("merge".to_string(), json!(true));

        let right = vec![AnyUIMessage::UI(merge_msg)];
        let result = ui_message_reducer(&left, &right).unwrap();
        assert_eq!(result.len(), 1);
        if let AnyUIMessage::UI(ui) = &result[0] {
            assert_eq!(ui.props.get("label").unwrap(), &json!("click"));
            assert_eq!(ui.props.get("color").unwrap(), &json!("blue"));
        } else {
            panic!("Expected UI message");
        }
    }

    #[test]
    fn test_any_ui_message_serde() {
        let ui_msg = AnyUIMessage::UI(UIMessage::new(
            "id1",
            "button",
            HashMap::from([("label".to_string(), json!("click"))]),
        ));
        let serialized = serde_json::to_string(&ui_msg).unwrap();
        let deserialized: AnyUIMessage = serde_json::from_str(&serialized).unwrap();
        assert_eq!(ui_msg, deserialized);

        let remove_msg = AnyUIMessage::Remove(RemoveUIMessage::new("id1"));
        let serialized = serde_json::to_string(&remove_msg).unwrap();
        let deserialized: AnyUIMessage = serde_json::from_str(&serialized).unwrap();
        assert_eq!(remove_msg, deserialized);
    }
}
