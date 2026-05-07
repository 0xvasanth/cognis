//! Typed state channel reducers for controlled state merging.
//!
//! Reducers define how individual fields in a graph's state are updated when a
//! node returns a new value for that field. By default, fields are overwritten
//! (last-write-wins), but custom reducers allow accumulation, deep merging, or
//! arbitrary transformation logic.
//!
//! # Overview
//!
//! - [`Reducer`] — Trait that all reducers implement.
//! - [`LastValueReducer`] — Overwrites the field with the new value (default).
//! - [`AppendReducer`] — Accumulates values into a JSON array.
//! - [`MergeReducer`] — Deep-merges JSON objects, concatenates arrays, overwrites scalars.
//! - [`BinaryOpReducer`] — User-supplied closure for maximum flexibility.
//! - [`StateSchema`] — Per-field reducer configuration for a graph's state.
//! - [`reduce_state`] — Applies per-field reducers according to a schema.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Reducer trait
// ---------------------------------------------------------------------------

/// Defines how a single state field is updated when a node produces a new value.
pub trait Reducer: Send + Sync {
    /// Combine `current` (the existing field value) with `update` (the new
    /// value produced by a node) and return the result.
    fn reduce(&self, current: &Value, update: &Value) -> Value;
}

// ---------------------------------------------------------------------------
// LastValueReducer
// ---------------------------------------------------------------------------

/// A reducer that simply overwrites the current value with the update.
///
/// This is the default behaviour when no reducer is specified for a field.
#[derive(Debug, Clone, Copy, Default)]
pub struct LastValueReducer;

impl Reducer for LastValueReducer {
    fn reduce(&self, _current: &Value, update: &Value) -> Value {
        update.clone()
    }
}

// ---------------------------------------------------------------------------
// AppendReducer
// ---------------------------------------------------------------------------

/// A reducer that accumulates values into a JSON array.
///
/// - If `current` is an array, the update value(s) are pushed onto it.
/// - If `current` is `Null`, a new array is started.
/// - If the update is itself an array, its elements are individually appended.
/// - Otherwise the update is pushed as a single element.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppendReducer;

impl Reducer for AppendReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Value {
        let mut arr = match current {
            Value::Array(a) => a.clone(),
            Value::Null => Vec::new(),
            other => vec![other.clone()],
        };

        match update {
            Value::Array(items) => arr.extend(items.iter().cloned()),
            other => arr.push(other.clone()),
        }

        Value::Array(arr)
    }
}

// ---------------------------------------------------------------------------
// MergeReducer
// ---------------------------------------------------------------------------

/// A reducer that deep-merges JSON values.
///
/// - **Objects** are recursively merged: keys from the update are applied on
///   top of the current value, with nested objects merged recursively.
/// - **Arrays** are concatenated (current elements followed by update elements).
/// - **Scalars** (strings, numbers, booleans, null) are overwritten by the
///   update value.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeReducer;

impl MergeReducer {
    /// Recursively deep-merge two JSON values.
    fn deep_merge(current: &Value, update: &Value) -> Value {
        match (current, update) {
            (Value::Object(cur_map), Value::Object(upd_map)) => {
                let mut result = cur_map.clone();
                for (key, upd_val) in upd_map {
                    let merged = if let Some(cur_val) = result.get(key) {
                        Self::deep_merge(cur_val, upd_val)
                    } else {
                        upd_val.clone()
                    };
                    result.insert(key.clone(), merged);
                }
                Value::Object(result)
            }
            (Value::Array(cur_arr), Value::Array(upd_arr)) => {
                let mut combined = cur_arr.clone();
                combined.extend(upd_arr.iter().cloned());
                Value::Array(combined)
            }
            (_current, update) => update.clone(),
        }
    }
}

impl Reducer for MergeReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Value {
        Self::deep_merge(current, update)
    }
}

// ---------------------------------------------------------------------------
// BinaryOpReducer
// ---------------------------------------------------------------------------

/// A reducer backed by a user-supplied closure.
///
/// This provides maximum flexibility: any `Fn(&Value, &Value) -> Value`
/// closure can be used to define custom reduction logic.
pub struct BinaryOpReducer {
    #[allow(clippy::type_complexity)]
    op: Arc<dyn Fn(&Value, &Value) -> Value + Send + Sync>,
}

impl BinaryOpReducer {
    /// Create a new `BinaryOpReducer` from a closure.
    pub fn new<F>(op: F) -> Self
    where
        F: Fn(&Value, &Value) -> Value + Send + Sync + 'static,
    {
        Self { op: Arc::new(op) }
    }
}

impl std::fmt::Debug for BinaryOpReducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryOpReducer")
            .field("op", &"<closure>")
            .finish()
    }
}

impl Reducer for BinaryOpReducer {
    fn reduce(&self, current: &Value, update: &Value) -> Value {
        (self.op)(current, update)
    }
}

// ---------------------------------------------------------------------------
// FieldSpec & StateSchema
// ---------------------------------------------------------------------------

/// Specification for a single field in a graph state schema.
pub struct FieldSpec {
    /// The name of the field.
    pub field_name: String,
    /// The reducer to apply when this field is updated.
    /// If `None`, [`LastValueReducer`] semantics are used.
    pub reducer: Option<Box<dyn Reducer>>,
    /// An optional default value for this field.
    pub default_value: Option<Value>,
}

impl std::fmt::Debug for FieldSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldSpec")
            .field("field_name", &self.field_name)
            .field("reducer", &self.reducer.is_some())
            .field("default_value", &self.default_value)
            .finish()
    }
}

/// Schema that defines per-field reducers and defaults for a graph's state.
///
/// Use the builder API ([`StateSchema::builder`]) for ergonomic construction.
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
}

impl Default for StateSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing a [`StateSchema`] incrementally.
pub struct StateSchemaBuilder {
    fields: HashMap<String, FieldSpec>,
}

impl StateSchemaBuilder {
    /// Add a field with a custom reducer.
    pub fn field(mut self, name: impl Into<String>, reducer: impl Reducer + 'static) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSpec {
                field_name: name,
                reducer: Some(Box::new(reducer)),
                default_value: None,
            },
        );
        self
    }

    /// Add a field with a custom reducer and a default value.
    pub fn field_with_default(
        mut self,
        name: impl Into<String>,
        reducer: impl Reducer + 'static,
        default: Value,
    ) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSpec {
                field_name: name,
                reducer: Some(Box::new(reducer)),
                default_value: Some(default),
            },
        );
        self
    }

    /// Add a field that uses the default [`LastValueReducer`] (overwrite) with
    /// an optional default value.
    pub fn last_value_field(mut self, name: impl Into<String>, default: Option<Value>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSpec {
                field_name: name,
                reducer: None,
                default_value: default,
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

// ---------------------------------------------------------------------------
// reduce_state
// ---------------------------------------------------------------------------

/// The fallback reducer used when no per-field reducer is configured.
static DEFAULT_REDUCER: LastValueReducer = LastValueReducer;

/// Apply per-field reducers to merge an update into the current state.
///
/// # Behaviour
///
/// 1. Fields present in **both** `current` and `update` are reduced using the
///    reducer from `schema` (or [`LastValueReducer`] if none is configured).
/// 2. Fields present in `current` but **not** in `update` are preserved as-is.
/// 3. Fields present in `update` but **not** in `current` are added. If the
///    schema defines a default for that field, the reducer is applied with the
///    default as the current value.
/// 4. Fields defined in the schema with a default value that are absent from
///    both `current` and `update` are populated with their defaults.
///
/// Both `current` and `update` are expected to be JSON objects. If either is
/// not an object, the update value is returned as-is (replacing current).
pub fn reduce_state(schema: &StateSchema, current: &Value, update: &Value) -> Value {
    let (cur_obj, upd_obj) = match (current.as_object(), update.as_object()) {
        (Some(c), Some(u)) => (c, u),
        // Non-object state: fall back to overwrite.
        _ => return update.clone(),
    };

    let mut result = cur_obj.clone();

    // Apply updates.
    for (key, upd_val) in upd_obj {
        let reducer: &dyn Reducer = match schema.fields.get(key) {
            Some(spec) => match &spec.reducer {
                Some(r) => r.as_ref(),
                None => &DEFAULT_REDUCER,
            },
            None => &DEFAULT_REDUCER,
        };

        let cur_val = result
            .get(key)
            .cloned()
            .or_else(|| schema.fields.get(key).and_then(|s| s.default_value.clone()))
            .unwrap_or(Value::Null);

        result.insert(key.clone(), reducer.reduce(&cur_val, upd_val));
    }

    // Populate defaults for fields that exist in the schema but are absent
    // from the result so far.
    for (key, spec) in &schema.fields {
        if !result.contains_key(key) {
            if let Some(default) = &spec.default_value {
                result.insert(key.clone(), default.clone());
            }
        }
    }

    Value::Object(result)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // 1. LastValueReducer overwrites
    // -----------------------------------------------------------------------
    #[test]
    fn test_last_value_overwrites() {
        let r = LastValueReducer;
        let current = json!(42);
        let update = json!(99);
        assert_eq!(r.reduce(&current, &update), json!(99));
    }

    #[test]
    fn test_last_value_overwrites_object() {
        let r = LastValueReducer;
        let current = json!({"a": 1});
        let update = json!({"b": 2});
        assert_eq!(r.reduce(&current, &update), json!({"b": 2}));
    }

    // -----------------------------------------------------------------------
    // 2. AppendReducer accumulates
    // -----------------------------------------------------------------------
    #[test]
    fn test_append_reducer_accumulates() {
        let r = AppendReducer;
        let current = json!([1, 2]);
        let update = json!(3);
        assert_eq!(r.reduce(&current, &update), json!([1, 2, 3]));
    }

    #[test]
    fn test_append_reducer_accumulates_array_update() {
        let r = AppendReducer;
        let current = json!([1]);
        let update = json!([2, 3]);
        assert_eq!(r.reduce(&current, &update), json!([1, 2, 3]));
    }

    // -----------------------------------------------------------------------
    // 3. AppendReducer with null start
    // -----------------------------------------------------------------------
    #[test]
    fn test_append_reducer_null_start() {
        let r = AppendReducer;
        let current = Value::Null;
        let update = json!("hello");
        assert_eq!(r.reduce(&current, &update), json!(["hello"]));
    }

    #[test]
    fn test_append_reducer_null_start_array_update() {
        let r = AppendReducer;
        let current = Value::Null;
        let update = json!([1, 2]);
        assert_eq!(r.reduce(&current, &update), json!([1, 2]));
    }

    // -----------------------------------------------------------------------
    // 4. MergeReducer deep merges objects
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_reducer_deep_merges_objects() {
        let r = MergeReducer;
        let current = json!({
            "a": 1,
            "nested": {"x": 10, "y": 20}
        });
        let update = json!({
            "b": 2,
            "nested": {"y": 99, "z": 30}
        });
        let result = r.reduce(&current, &update);
        assert_eq!(
            result,
            json!({
                "a": 1,
                "b": 2,
                "nested": {"x": 10, "y": 99, "z": 30}
            })
        );
    }

    // -----------------------------------------------------------------------
    // 5. MergeReducer concatenates arrays
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_reducer_concatenates_arrays() {
        let r = MergeReducer;
        let current = json!([1, 2]);
        let update = json!([3, 4]);
        assert_eq!(r.reduce(&current, &update), json!([1, 2, 3, 4]));
    }

    #[test]
    fn test_merge_reducer_scalar_overwrite() {
        let r = MergeReducer;
        assert_eq!(r.reduce(&json!("old"), &json!("new")), json!("new"));
        assert_eq!(r.reduce(&json!(1), &json!(2)), json!(2));
    }

    // -----------------------------------------------------------------------
    // 6. BinaryOpReducer custom logic
    // -----------------------------------------------------------------------
    #[test]
    fn test_binary_op_reducer_max() {
        let r = BinaryOpReducer::new(|a, b| {
            let a_num = a.as_i64().unwrap_or(0);
            let b_num = b.as_i64().unwrap_or(0);
            json!(a_num.max(b_num))
        });
        assert_eq!(r.reduce(&json!(5), &json!(3)), json!(5));
        assert_eq!(r.reduce(&json!(5), &json!(10)), json!(10));
    }

    #[test]
    fn test_binary_op_reducer_sum() {
        let r = BinaryOpReducer::new(|a, b| {
            let a_num = a.as_i64().unwrap_or(0);
            let b_num = b.as_i64().unwrap_or(0);
            json!(a_num + b_num)
        });
        assert_eq!(r.reduce(&json!(3), &json!(7)), json!(10));
    }

    // -----------------------------------------------------------------------
    // 7. StateSchema with mixed reducers
    // -----------------------------------------------------------------------
    #[test]
    fn test_schema_with_mixed_reducers() {
        let schema = StateSchema::builder()
            .field("messages", AppendReducer)
            .field("config", MergeReducer)
            .last_value_field("status", None)
            .build();

        let current = json!({
            "messages": ["hello"],
            "config": {"debug": true},
            "status": "idle"
        });
        let update = json!({
            "messages": "world",
            "config": {"verbose": true},
            "status": "running"
        });
        let result = reduce_state(&schema, &current, &update);

        assert_eq!(result["messages"], json!(["hello", "world"]));
        assert_eq!(result["config"], json!({"debug": true, "verbose": true}));
        assert_eq!(result["status"], json!("running"));
    }

    // -----------------------------------------------------------------------
    // 8. reduce_state preserves untouched fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduce_state_preserves_untouched_fields() {
        let schema = StateSchema::new();
        let current = json!({
            "a": 1,
            "b": 2,
            "c": 3
        });
        let update = json!({
            "b": 20
        });
        let result = reduce_state(&schema, &current, &update);
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(20));
        assert_eq!(result["c"], json!(3));
    }

    // -----------------------------------------------------------------------
    // 9. reduce_state with defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduce_state_with_defaults() {
        let schema = StateSchema::builder()
            .field_with_default("counter", AppendReducer, json!([]))
            .field_with_default("name", LastValueReducer, json!("unnamed"))
            .build();

        // Start with empty state.
        let current = json!({});
        let update = json!({});
        let result = reduce_state(&schema, &current, &update);

        // Defaults should be populated.
        assert_eq!(result["counter"], json!([]));
        assert_eq!(result["name"], json!("unnamed"));
    }

    #[test]
    fn test_reduce_state_default_used_as_base_for_reducer() {
        let schema = StateSchema::builder()
            .field_with_default("items", AppendReducer, json!([]))
            .build();

        let current = json!({});
        let update = json!({"items": "first"});
        let result = reduce_state(&schema, &current, &update);
        // The default [] is used as base, then "first" is appended.
        assert_eq!(result["items"], json!(["first"]));
    }

    // -----------------------------------------------------------------------
    // 10. Schema builder pattern
    // -----------------------------------------------------------------------
    #[test]
    fn test_schema_builder_pattern() {
        let schema = StateSchema::builder()
            .field("log", AppendReducer)
            .field("metadata", MergeReducer)
            .last_value_field("version", Some(json!("1.0")))
            .field_with_default(
                "count",
                BinaryOpReducer::new(|a, b| {
                    let a_num = a.as_i64().unwrap_or(0);
                    let b_num = b.as_i64().unwrap_or(0);
                    json!(a_num + b_num)
                }),
                json!(0),
            )
            .build();

        assert_eq!(schema.fields.len(), 4);
        assert!(schema.fields.contains_key("log"));
        assert!(schema.fields.contains_key("metadata"));
        assert!(schema.fields.contains_key("version"));
        assert!(schema.fields.contains_key("count"));

        // Verify defaults.
        assert_eq!(schema.fields["version"].default_value, Some(json!("1.0")));
        assert_eq!(schema.fields["count"].default_value, Some(json!(0)));
    }

    // -----------------------------------------------------------------------
    // 11. Reducer with complex nested state
    // -----------------------------------------------------------------------
    #[test]
    fn test_reducer_complex_nested_state() {
        let schema = StateSchema::builder()
            .field("agent_state", MergeReducer)
            .field("messages", AppendReducer)
            .field("tool_calls", AppendReducer)
            .last_value_field("current_node", None)
            .build();

        let current = json!({
            "agent_state": {
                "memory": {
                    "short_term": ["fact1"],
                    "long_term": {"key1": "val1"}
                },
                "iteration": 1
            },
            "messages": [
                {"role": "user", "content": "hello"}
            ],
            "tool_calls": [],
            "current_node": "agent"
        });

        let update = json!({
            "agent_state": {
                "memory": {
                    "short_term": ["fact2"],
                    "long_term": {"key2": "val2"}
                },
                "iteration": 2
            },
            "messages": {"role": "assistant", "content": "hi there"},
            "tool_calls": [{"name": "search", "args": {}}],
            "current_node": "tools"
        });

        let result = reduce_state(&schema, &current, &update);

        // MergeReducer deep-merges agent_state.
        assert_eq!(result["agent_state"]["iteration"], json!(2));
        assert_eq!(
            result["agent_state"]["memory"]["short_term"],
            json!(["fact1", "fact2"])
        );
        assert_eq!(
            result["agent_state"]["memory"]["long_term"],
            json!({"key1": "val1", "key2": "val2"})
        );

        // AppendReducer accumulates messages.
        assert_eq!(result["messages"].as_array().unwrap().len(), 2);
        assert_eq!(result["messages"][0]["role"], json!("user"));
        assert_eq!(result["messages"][1]["role"], json!("assistant"));

        // AppendReducer accumulates tool_calls.
        assert_eq!(result["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(result["tool_calls"][0]["name"], json!("search"));

        // LastValue overwrites current_node.
        assert_eq!(result["current_node"], json!("tools"));
    }

    // -----------------------------------------------------------------------
    // Extra: non-object state falls back to overwrite
    // -----------------------------------------------------------------------
    #[test]
    fn test_reduce_state_non_object_fallback() {
        let schema = StateSchema::new();
        let current = json!(42);
        let update = json!(99);
        assert_eq!(reduce_state(&schema, &current, &update), json!(99));
    }

    // -----------------------------------------------------------------------
    // Extra: append from non-array current
    // -----------------------------------------------------------------------
    #[test]
    fn test_append_reducer_from_non_array_current() {
        let r = AppendReducer;
        let current = json!("existing");
        let update = json!("new");
        assert_eq!(r.reduce(&current, &update), json!(["existing", "new"]));
    }
}
