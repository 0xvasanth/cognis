//! Comprehensive state reducer utilities with schema validation.
//!
//! This module extends the basic [`Reducer`](super::reducers::Reducer) system
//! with a richer set of built-in reducers, per-field validation, and a
//! schema-driven approach to state management.
//!
//! # Built-in Reducers
//!
//! - [`OverwriteReducer`] — Always takes the new value.
//! - [`MergeObjectReducer`] — Deep merges JSON objects.
//! - [`AppendListReducer`] — Appends update items to a list.
//! - [`UniqueListReducer`] — Appends only unique items.
//! - [`AddNumberReducer`] — Adds numeric values.
//! - [`MaxNumberReducer`] / [`MinNumberReducer`] — Keeps max/min.
//! - [`ConcatStringReducer`] — Concatenates strings with optional separator.
//! - [`MessageListReducer`] — Specialized for message arrays with dedup by ID.
//! - [`CustomReducer`] — Wraps an arbitrary closure.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// StateReducer trait
// ---------------------------------------------------------------------------

/// Defines how a single state field is updated, with error handling.
///
/// Unlike the simpler [`Reducer`](super::reducers::Reducer) trait, this trait
/// returns a [`Result`] so that reducers can signal type mismatches or other
/// validation failures.
pub trait StateReducer: Send + Sync {
    /// Combine `current` with `update` and return the result, or an error if
    /// the values are incompatible.
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value>;

    /// A human-readable name for this reducer, useful for diagnostics.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// OverwriteReducer
// ---------------------------------------------------------------------------

/// A reducer that always replaces the current value with the update.
#[derive(Debug, Clone, Copy, Default)]
pub struct OverwriteReducer;

impl StateReducer for OverwriteReducer {
    fn reduce(&self, _current: &Value, update: &Value) -> Result<Value> {
        Ok(update.clone())
    }

    fn name(&self) -> &str {
        "overwrite"
    }
}

// ---------------------------------------------------------------------------
// MergeObjectReducer
// ---------------------------------------------------------------------------

/// A reducer that deep-merges JSON objects.
///
/// - Keys from the update overwrite keys in the current value.
/// - Keys present only in the current value are preserved.
/// - Nested objects are merged recursively.
/// - Non-object values are overwritten.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeObjectReducer;

impl MergeObjectReducer {
    fn deep_merge(current: &Value, update: &Value) -> Value {
        match (current, update) {
            (Value::Object(cur), Value::Object(upd)) => {
                let mut result = cur.clone();
                for (key, upd_val) in upd {
                    let merged = if let Some(cur_val) = result.get(key) {
                        Self::deep_merge(cur_val, upd_val)
                    } else {
                        upd_val.clone()
                    };
                    result.insert(key.clone(), merged);
                }
                Value::Object(result)
            }
            (_, update) => update.clone(),
        }
    }
}

impl StateReducer for MergeObjectReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        Ok(Self::deep_merge(current, update))
    }

    fn name(&self) -> &str {
        "merge_object"
    }
}

// ---------------------------------------------------------------------------
// AppendListReducer
// ---------------------------------------------------------------------------

/// A reducer that appends update items to a JSON array.
///
/// - If `current` is an array, items are appended.
/// - If `current` is `Null`, a new array is started.
/// - If `update` is an array, its elements are individually appended.
/// - Otherwise the update is pushed as a single element.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppendListReducer;

impl StateReducer for AppendListReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let mut arr = match current {
            Value::Array(a) => a.clone(),
            Value::Null => Vec::new(),
            other => vec![other.clone()],
        };

        match update {
            Value::Array(items) => arr.extend(items.iter().cloned()),
            other => arr.push(other.clone()),
        }

        Ok(Value::Array(arr))
    }

    fn name(&self) -> &str {
        "append_list"
    }
}

// ---------------------------------------------------------------------------
// UniqueListReducer
// ---------------------------------------------------------------------------

/// A reducer that appends only unique items to a JSON array.
///
/// Uniqueness is determined by JSON value equality.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniqueListReducer;

impl StateReducer for UniqueListReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let mut arr = match current {
            Value::Array(a) => a.clone(),
            Value::Null => Vec::new(),
            other => vec![other.clone()],
        };

        let items_to_add: Vec<&Value> = match update {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };

        for item in items_to_add {
            if !arr.contains(item) {
                arr.push(item.clone());
            }
        }

        Ok(Value::Array(arr))
    }

    fn name(&self) -> &str {
        "unique_list"
    }
}

// ---------------------------------------------------------------------------
// AddNumberReducer
// ---------------------------------------------------------------------------

/// A reducer that adds numeric values together.
///
/// Both values must be numbers. If either is not a number, returns an error.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddNumberReducer;

impl StateReducer for AddNumberReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        // Handle null current as zero
        let current = if current.is_null() {
            &Value::from(0)
        } else {
            current
        };

        match (current.as_f64(), update.as_f64()) {
            (Some(a), Some(b)) => {
                // Preserve integer type if both are integers
                if current.is_i64() && update.is_i64() {
                    Ok(Value::from(
                        current.as_i64().unwrap() + update.as_i64().unwrap(),
                    ))
                } else {
                    Ok(Value::from(a + b))
                }
            }
            _ => Err(LangGraphError::InvalidUpdateError(format!(
                "AddNumberReducer: both values must be numbers, got current={current}, update={update}"
            ))),
        }
    }

    fn name(&self) -> &str {
        "add_number"
    }
}

// ---------------------------------------------------------------------------
// MaxNumberReducer
// ---------------------------------------------------------------------------

/// A reducer that keeps the maximum of two numeric values.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxNumberReducer;

impl StateReducer for MaxNumberReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let current = if current.is_null() { update } else { current };

        match (current.as_f64(), update.as_f64()) {
            (Some(a), Some(b)) => {
                if a >= b {
                    Ok(current.clone())
                } else {
                    Ok(update.clone())
                }
            }
            _ => Err(LangGraphError::InvalidUpdateError(format!(
                "MaxNumberReducer: both values must be numbers, got current={current}, update={update}"
            ))),
        }
    }

    fn name(&self) -> &str {
        "max_number"
    }
}

// ---------------------------------------------------------------------------
// MinNumberReducer
// ---------------------------------------------------------------------------

/// A reducer that keeps the minimum of two numeric values.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinNumberReducer;

impl StateReducer for MinNumberReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let current = if current.is_null() { update } else { current };

        match (current.as_f64(), update.as_f64()) {
            (Some(a), Some(b)) => {
                if a <= b {
                    Ok(current.clone())
                } else {
                    Ok(update.clone())
                }
            }
            _ => Err(LangGraphError::InvalidUpdateError(format!(
                "MinNumberReducer: both values must be numbers, got current={current}, update={update}"
            ))),
        }
    }

    fn name(&self) -> &str {
        "min_number"
    }
}

// ---------------------------------------------------------------------------
// ConcatStringReducer
// ---------------------------------------------------------------------------

/// A reducer that concatenates string values with an optional separator.
#[derive(Debug, Clone)]
pub struct ConcatStringReducer {
    separator: String,
}

impl ConcatStringReducer {
    /// Create a new `ConcatStringReducer` with the given separator.
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
        }
    }

    /// Create a `ConcatStringReducer` with no separator (direct concatenation).
    pub fn no_separator() -> Self {
        Self {
            separator: String::new(),
        }
    }
}

impl Default for ConcatStringReducer {
    fn default() -> Self {
        Self::no_separator()
    }
}

impl StateReducer for ConcatStringReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let cur_str = match current {
            Value::String(s) => s.as_str(),
            Value::Null => "",
            _ => {
                return Err(LangGraphError::InvalidUpdateError(format!(
                    "ConcatStringReducer: current value must be a string or null, got {current}"
                )));
            }
        };

        let upd_str = match update {
            Value::String(s) => s.as_str(),
            _ => {
                return Err(LangGraphError::InvalidUpdateError(format!(
                    "ConcatStringReducer: update value must be a string, got {update}"
                )));
            }
        };

        if cur_str.is_empty() {
            Ok(Value::String(upd_str.to_string()))
        } else {
            Ok(Value::String(format!(
                "{}{}{}",
                cur_str, self.separator, upd_str
            )))
        }
    }

    fn name(&self) -> &str {
        "concat_string"
    }
}

// ---------------------------------------------------------------------------
// MessageListReducer
// ---------------------------------------------------------------------------

/// A reducer specialized for message arrays, appending with dedup by message ID.
///
/// Messages are JSON objects expected to have an `"id"` field. When appending,
/// if a message with the same ID already exists, the existing message is
/// replaced in place. Messages without an `"id"` field are always appended.
#[derive(Debug, Clone, Copy, Default)]
pub struct MessageListReducer;

impl StateReducer for MessageListReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        let mut messages = match current {
            Value::Array(a) => a.clone(),
            Value::Null => Vec::new(),
            _ => {
                return Err(LangGraphError::InvalidUpdateError(
                    "MessageListReducer: current value must be an array or null".to_string(),
                ));
            }
        };

        let new_messages: Vec<&Value> = match update {
            Value::Array(items) => items.iter().collect(),
            Value::Object(_) => vec![update],
            _ => {
                return Err(LangGraphError::InvalidUpdateError(
                    "MessageListReducer: update must be a message object or array of messages"
                        .to_string(),
                ));
            }
        };

        for msg in new_messages {
            let msg_id = msg.get("id").and_then(Value::as_str);

            if let Some(id) = msg_id {
                // Check if a message with this ID already exists
                if let Some(pos) = messages
                    .iter()
                    .position(|m| m.get("id").and_then(Value::as_str) == Some(id))
                {
                    // Replace in place
                    messages[pos] = msg.clone();
                    continue;
                }
            }
            // Append new message
            messages.push(msg.clone());
        }

        Ok(Value::Array(messages))
    }

    fn name(&self) -> &str {
        "message_list"
    }
}

// ---------------------------------------------------------------------------
// CustomReducer
// ---------------------------------------------------------------------------

/// A reducer backed by a user-supplied closure, with error handling.
pub struct CustomReducer {
    reducer_name: String,
    #[allow(clippy::type_complexity)]
    op: Arc<dyn Fn(&Value, &Value) -> Result<Value> + Send + Sync>,
}

impl CustomReducer {
    /// Create a new `CustomReducer` with a name and closure.
    pub fn new<F>(name: impl Into<String>, op: F) -> Self
    where
        F: Fn(&Value, &Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            reducer_name: name.into(),
            op: Arc::new(op),
        }
    }
}

impl std::fmt::Debug for CustomReducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomReducer")
            .field("name", &self.reducer_name)
            .field("op", &"<closure>")
            .finish()
    }
}

impl StateReducer for CustomReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Result<Value> {
        (self.op)(current, update)
    }

    fn name(&self) -> &str {
        &self.reducer_name
    }
}

// ---------------------------------------------------------------------------
// FieldSpec
// ---------------------------------------------------------------------------

/// Specification for a single field in a state schema.
pub struct FieldSpec {
    /// The reducer to apply when this field is updated.
    pub reducer: Box<dyn StateReducer>,
    /// An optional default value for this field.
    pub default_value: Option<Value>,
    /// Whether this field is required in the state.
    pub required: bool,
}

impl std::fmt::Debug for FieldSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldSpec")
            .field("reducer", &self.reducer.name())
            .field("default_value", &self.default_value)
            .field("required", &self.required)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StateSchema
// ---------------------------------------------------------------------------

/// Schema that defines per-field reducers, defaults, and validation for a
/// graph's state.
#[derive(Debug)]
pub struct StateSchema {
    /// Per-field specifications, keyed by field name.
    pub fields: HashMap<String, FieldSpec>,
}

impl StateSchema {
    /// Create a new empty `StateSchema`.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Start building a `StateSchema` using the builder pattern.
    pub fn builder() -> StateSchemaBuilder {
        StateSchemaBuilder {
            fields: HashMap::new(),
        }
    }

    /// Add a field to the schema.
    pub fn add_field(&mut self, name: &str, reducer: Box<dyn StateReducer>) {
        self.fields.insert(
            name.to_string(),
            FieldSpec {
                reducer,
                default_value: None,
                required: false,
            },
        );
    }

    /// Apply per-field reducers to merge an update into the current state.
    ///
    /// Fields present in both `current` and `update` are reduced using the
    /// configured reducer. Fields only in `current` are preserved. Fields only
    /// in `update` use the field's default as the base (or `Null`). Missing
    /// fields with defaults are populated.
    pub fn reduce_state(&self, current: &Value, updates: &Value) -> Result<Value> {
        let (cur_obj, upd_obj) = match (current.as_object(), updates.as_object()) {
            (Some(c), Some(u)) => (c, u),
            _ => return Ok(updates.clone()),
        };

        let mut result = cur_obj.clone();

        for (key, upd_val) in upd_obj {
            let cur_val = result
                .get(key)
                .cloned()
                .or_else(|| self.fields.get(key).and_then(|s| s.default_value.clone()))
                .unwrap_or(Value::Null);

            if let Some(spec) = self.fields.get(key) {
                result.insert(key.clone(), spec.reducer.reduce(&cur_val, upd_val)?);
            } else {
                // No reducer configured — default to overwrite
                result.insert(key.clone(), upd_val.clone());
            }
        }

        // Populate defaults for fields absent from the result
        for (key, spec) in &self.fields {
            if !result.contains_key(key) {
                if let Some(default) = &spec.default_value {
                    result.insert(key.clone(), default.clone());
                }
            }
        }

        Ok(Value::Object(result))
    }

    /// Validate that the given state satisfies the schema constraints.
    ///
    /// Currently checks:
    /// - All required fields are present and not null.
    pub fn validate_state(&self, state: &Value) -> Result<()> {
        let obj = state.as_object().ok_or_else(|| {
            LangGraphError::InvalidUpdateError("State must be a JSON object".to_string())
        })?;

        for (field_name, spec) in &self.fields {
            if spec.required {
                match obj.get(field_name) {
                    None => {
                        return Err(LangGraphError::InvalidUpdateError(format!(
                            "Required field '{field_name}' is missing from state"
                        )));
                    }
                    Some(Value::Null) => {
                        return Err(LangGraphError::InvalidUpdateError(format!(
                            "Required field '{field_name}' is null"
                        )));
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

impl Default for StateSchema {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StateSchemaBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`StateSchema`] incrementally.
pub struct StateSchemaBuilder {
    fields: HashMap<String, FieldSpec>,
}

impl StateSchemaBuilder {
    /// Add a field with a reducer.
    pub fn field(mut self, name: impl Into<String>, reducer: impl StateReducer + 'static) -> Self {
        let name = name.into();
        self.fields.insert(
            name,
            FieldSpec {
                reducer: Box::new(reducer),
                default_value: None,
                required: false,
            },
        );
        self
    }

    /// Add a field with a reducer and default value.
    pub fn field_with_default(
        mut self,
        name: impl Into<String>,
        reducer: impl StateReducer + 'static,
        default: Value,
    ) -> Self {
        let name = name.into();
        self.fields.insert(
            name,
            FieldSpec {
                reducer: Box::new(reducer),
                default_value: Some(default),
                required: false,
            },
        );
        self
    }

    /// Add a required field with a reducer.
    pub fn required_field(
        mut self,
        name: impl Into<String>,
        reducer: impl StateReducer + 'static,
    ) -> Self {
        let name = name.into();
        self.fields.insert(
            name,
            FieldSpec {
                reducer: Box::new(reducer),
                default_value: None,
                required: true,
            },
        );
        self
    }

    /// Add a required field with a reducer and default value.
    pub fn required_field_with_default(
        mut self,
        name: impl Into<String>,
        reducer: impl StateReducer + 'static,
        default: Value,
    ) -> Self {
        let name = name.into();
        self.fields.insert(
            name,
            FieldSpec {
                reducer: Box::new(reducer),
                default_value: Some(default),
                required: true,
            },
        );
        self
    }

    /// Consume the builder and produce the finished [`StateSchema`].
    pub fn build(self) -> StateSchema {
        StateSchema {
            fields: self.fields,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // 1. OverwriteReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_overwrite_reducer_replaces_value() {
        let r = OverwriteReducer;
        assert_eq!(r.reduce(&json!(1), &json!(2)).unwrap(), json!(2));
    }

    #[test]
    fn test_overwrite_reducer_replaces_object() {
        let r = OverwriteReducer;
        assert_eq!(
            r.reduce(&json!({"a": 1}), &json!({"b": 2})).unwrap(),
            json!({"b": 2})
        );
    }

    #[test]
    fn test_overwrite_reducer_name() {
        assert_eq!(OverwriteReducer.name(), "overwrite");
    }

    // -----------------------------------------------------------------------
    // 2. MergeObjectReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_object_deep_merges() {
        let r = MergeObjectReducer;
        let current = json!({"a": 1, "nested": {"x": 10, "y": 20}});
        let update = json!({"b": 2, "nested": {"y": 99, "z": 30}});
        let result = r.reduce(&current, &update).unwrap();
        assert_eq!(
            result,
            json!({"a": 1, "b": 2, "nested": {"x": 10, "y": 99, "z": 30}})
        );
    }

    #[test]
    fn test_merge_object_scalar_overwrite() {
        let r = MergeObjectReducer;
        assert_eq!(
            r.reduce(&json!("old"), &json!("new")).unwrap(),
            json!("new")
        );
    }

    // -----------------------------------------------------------------------
    // 3. AppendListReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_append_list_single_item() {
        let r = AppendListReducer;
        assert_eq!(
            r.reduce(&json!([1, 2]), &json!(3)).unwrap(),
            json!([1, 2, 3])
        );
    }

    #[test]
    fn test_append_list_array_update() {
        let r = AppendListReducer;
        assert_eq!(
            r.reduce(&json!([1]), &json!([2, 3])).unwrap(),
            json!([1, 2, 3])
        );
    }

    #[test]
    fn test_append_list_null_start() {
        let r = AppendListReducer;
        assert_eq!(
            r.reduce(&Value::Null, &json!("hello")).unwrap(),
            json!(["hello"])
        );
    }

    // -----------------------------------------------------------------------
    // 4. UniqueListReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_unique_list_deduplicates() {
        let r = UniqueListReducer;
        let current = json!([1, 2, 3]);
        let update = json!([2, 3, 4, 5]);
        assert_eq!(r.reduce(&current, &update).unwrap(), json!([1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_unique_list_single_duplicate() {
        let r = UniqueListReducer;
        assert_eq!(r.reduce(&json!([1, 2]), &json!(2)).unwrap(), json!([1, 2]));
    }

    #[test]
    fn test_unique_list_null_start() {
        let r = UniqueListReducer;
        assert_eq!(
            r.reduce(&Value::Null, &json!([1, 1, 2])).unwrap(),
            json!([1, 2])
        );
    }

    // -----------------------------------------------------------------------
    // 5. AddNumberReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_number_integers() {
        let r = AddNumberReducer;
        assert_eq!(r.reduce(&json!(3), &json!(7)).unwrap(), json!(10));
    }

    #[test]
    fn test_add_number_floats() {
        let r = AddNumberReducer;
        let result = r.reduce(&json!(1.5), &json!(2.5)).unwrap();
        assert_eq!(result, json!(4.0));
    }

    #[test]
    fn test_add_number_null_current() {
        let r = AddNumberReducer;
        assert_eq!(r.reduce(&Value::Null, &json!(5)).unwrap(), json!(5));
    }

    #[test]
    fn test_add_number_type_error() {
        let r = AddNumberReducer;
        assert!(r.reduce(&json!("not a number"), &json!(5)).is_err());
    }

    // -----------------------------------------------------------------------
    // 6. MaxNumberReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_number_keeps_larger() {
        let r = MaxNumberReducer;
        assert_eq!(r.reduce(&json!(5), &json!(10)).unwrap(), json!(10));
        assert_eq!(r.reduce(&json!(10), &json!(5)).unwrap(), json!(10));
    }

    #[test]
    fn test_max_number_null_current() {
        let r = MaxNumberReducer;
        assert_eq!(r.reduce(&Value::Null, &json!(42)).unwrap(), json!(42));
    }

    // -----------------------------------------------------------------------
    // 7. MinNumberReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_min_number_keeps_smaller() {
        let r = MinNumberReducer;
        assert_eq!(r.reduce(&json!(5), &json!(10)).unwrap(), json!(5));
        assert_eq!(r.reduce(&json!(10), &json!(5)).unwrap(), json!(5));
    }

    #[test]
    fn test_min_number_error_on_non_number() {
        let r = MinNumberReducer;
        assert!(r.reduce(&json!("a"), &json!(1)).is_err());
    }

    // -----------------------------------------------------------------------
    // 8. ConcatStringReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_concat_string_no_separator() {
        let r = ConcatStringReducer::no_separator();
        assert_eq!(
            r.reduce(&json!("hello"), &json!("world")).unwrap(),
            json!("helloworld")
        );
    }

    #[test]
    fn test_concat_string_with_separator() {
        let r = ConcatStringReducer::new(", ");
        assert_eq!(
            r.reduce(&json!("hello"), &json!("world")).unwrap(),
            json!("hello, world")
        );
    }

    #[test]
    fn test_concat_string_null_current() {
        let r = ConcatStringReducer::new(" ");
        assert_eq!(
            r.reduce(&Value::Null, &json!("first")).unwrap(),
            json!("first")
        );
    }

    #[test]
    fn test_concat_string_error_on_non_string_update() {
        let r = ConcatStringReducer::default();
        assert!(r.reduce(&json!("hello"), &json!(42)).is_err());
    }

    // -----------------------------------------------------------------------
    // 9. MessageListReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_message_list_appends() {
        let r = MessageListReducer;
        let current = json!([{"id": "1", "content": "hello"}]);
        let update = json!({"id": "2", "content": "world"});
        let result = r.reduce(&current, &update).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_message_list_dedup_by_id() {
        let r = MessageListReducer;
        let current = json!([
            {"id": "1", "content": "old"},
            {"id": "2", "content": "keep"}
        ]);
        let update = json!([{"id": "1", "content": "updated"}]);
        let result = r.reduce(&current, &update).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
        assert_eq!(result[0]["content"], json!("updated"));
        assert_eq!(result[1]["content"], json!("keep"));
    }

    #[test]
    fn test_message_list_no_id_always_appends() {
        let r = MessageListReducer;
        let current = json!([{"content": "a"}]);
        let update = json!([{"content": "a"}]);
        let result = r.reduce(&current, &update).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_message_list_null_start() {
        let r = MessageListReducer;
        let result = r.reduce(&Value::Null, &json!([{"id": "1"}])).unwrap();
        assert_eq!(result.as_array().unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // 10. CustomReducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_custom_reducer() {
        let r = CustomReducer::new("sum", |a, b| {
            let sum = a.as_i64().unwrap_or(0) + b.as_i64().unwrap_or(0);
            Ok(json!(sum))
        });
        assert_eq!(r.reduce(&json!(3), &json!(7)).unwrap(), json!(10));
        assert_eq!(r.name(), "sum");
    }

    #[test]
    fn test_custom_reducer_error() {
        let r = CustomReducer::new("fail", |_a, _b| {
            Err(LangGraphError::InvalidUpdateError(
                "intentional error".to_string(),
            ))
        });
        assert!(r.reduce(&json!(1), &json!(2)).is_err());
    }

    // -----------------------------------------------------------------------
    // 11. StateSchema reduce_state
    // -----------------------------------------------------------------------
    #[test]
    fn test_schema_reduce_state_mixed_reducers() {
        let schema = StateSchema::builder()
            .field("messages", AppendListReducer)
            .field("config", MergeObjectReducer)
            .field("counter", AddNumberReducer)
            .build();

        let current = json!({
            "messages": ["hello"],
            "config": {"debug": true},
            "counter": 1
        });
        let update = json!({
            "messages": "world",
            "config": {"verbose": true},
            "counter": 5
        });
        let result = schema.reduce_state(&current, &update).unwrap();
        assert_eq!(result["messages"], json!(["hello", "world"]));
        assert_eq!(result["config"], json!({"debug": true, "verbose": true}));
        assert_eq!(result["counter"], json!(6));
    }

    #[test]
    fn test_schema_reduce_state_preserves_untouched() {
        let schema = StateSchema::builder().field("a", OverwriteReducer).build();
        let current = json!({"a": 1, "b": 2});
        let update = json!({"a": 10});
        let result = schema.reduce_state(&current, &update).unwrap();
        assert_eq!(result["a"], json!(10));
        assert_eq!(result["b"], json!(2));
    }

    #[test]
    fn test_schema_reduce_state_defaults() {
        let schema = StateSchema::builder()
            .field_with_default("items", AppendListReducer, json!([]))
            .field_with_default("name", OverwriteReducer, json!("unnamed"))
            .build();

        let result = schema.reduce_state(&json!({}), &json!({})).unwrap();
        assert_eq!(result["items"], json!([]));
        assert_eq!(result["name"], json!("unnamed"));
    }

    #[test]
    fn test_schema_reduce_state_default_as_base() {
        let schema = StateSchema::builder()
            .field_with_default("counter", AddNumberReducer, json!(0))
            .build();
        let result = schema
            .reduce_state(&json!({}), &json!({"counter": 5}))
            .unwrap();
        assert_eq!(result["counter"], json!(5));
    }

    // -----------------------------------------------------------------------
    // 12. StateSchema validate_state
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_state_required_present() {
        let schema = StateSchema::builder()
            .required_field("name", OverwriteReducer)
            .build();
        assert!(schema.validate_state(&json!({"name": "test"})).is_ok());
    }

    #[test]
    fn test_validate_state_required_missing() {
        let schema = StateSchema::builder()
            .required_field("name", OverwriteReducer)
            .build();
        assert!(schema.validate_state(&json!({})).is_err());
    }

    #[test]
    fn test_validate_state_required_null() {
        let schema = StateSchema::builder()
            .required_field("name", OverwriteReducer)
            .build();
        assert!(schema.validate_state(&json!({"name": null})).is_err());
    }

    #[test]
    fn test_validate_state_non_object() {
        let schema = StateSchema::new();
        assert!(schema.validate_state(&json!(42)).is_err());
    }

    #[test]
    fn test_validate_state_optional_missing_ok() {
        let schema = StateSchema::builder()
            .field("optional", OverwriteReducer)
            .build();
        assert!(schema.validate_state(&json!({})).is_ok());
    }

    // -----------------------------------------------------------------------
    // 13. StateSchemaBuilder
    // -----------------------------------------------------------------------
    #[test]
    fn test_builder_produces_correct_schema() {
        let schema = StateSchema::builder()
            .field("a", OverwriteReducer)
            .field_with_default("b", AppendListReducer, json!([]))
            .required_field("c", AddNumberReducer)
            .required_field_with_default("d", MergeObjectReducer, json!({}))
            .build();

        assert_eq!(schema.fields.len(), 4);
        assert!(!schema.fields["a"].required);
        assert!(!schema.fields["b"].required);
        assert!(schema.fields["c"].required);
        assert!(schema.fields["d"].required);
        assert_eq!(schema.fields["b"].default_value, Some(json!([])));
        assert_eq!(schema.fields["d"].default_value, Some(json!({})));
    }

    // -----------------------------------------------------------------------
    // 14. add_field method
    // -----------------------------------------------------------------------
    #[test]
    fn test_add_field_to_schema() {
        let mut schema = StateSchema::new();
        schema.add_field("counter", Box::new(AddNumberReducer));
        assert!(schema.fields.contains_key("counter"));
        assert_eq!(schema.fields["counter"].reducer.name(), "add_number");
    }

    // -----------------------------------------------------------------------
    // 15. Edge case: non-object state falls back
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduce_state_non_object_fallback() {
        let schema = StateSchema::new();
        let result = schema.reduce_state(&json!(42), &json!(99)).unwrap();
        assert_eq!(result, json!(99));
    }

    // -----------------------------------------------------------------------
    // 16. Reducer propagates errors through schema
    // -----------------------------------------------------------------------
    #[test]
    fn test_schema_propagates_reducer_error() {
        let schema = StateSchema::builder()
            .field("counter", AddNumberReducer)
            .build();
        let current = json!({"counter": "not a number"});
        let update = json!({"counter": 5});
        assert!(schema.reduce_state(&current, &update).is_err());
    }

    // -----------------------------------------------------------------------
    // 17. Fields not in schema use overwrite
    // -----------------------------------------------------------------------
    #[test]
    fn test_unknown_fields_use_overwrite() {
        let schema = StateSchema::new();
        let current = json!({"x": 1});
        let update = json!({"x": 99});
        let result = schema.reduce_state(&current, &update).unwrap();
        assert_eq!(result["x"], json!(99));
    }
}
