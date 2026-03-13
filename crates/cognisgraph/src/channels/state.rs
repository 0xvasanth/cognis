//! State channels with reducers for managing graph node communication.
//!
//! This module provides a higher-level state channel abstraction that combines
//! named channels with configurable reducers, a channel manager for coordinating
//! multiple channels, and specialized channel types for common patterns.
//!
//! # Types
//!
//! - [`ChannelValue`] — Typed wrapper around common JSON value types.
//! - [`ChannelReducer`] — Enum defining how channel values are combined on update.
//! - [`StateChannel`] — A named channel with a reducer and version tracking.
//! - [`ChannelManager`] — Manages a collection of [`StateChannel`]s.
//! - [`BinaryChannel`] — A channel that accepts exactly one write per superstep.
//! - [`LastValueChannel`] — A channel that always keeps only the most recent value.
//! - [`ChannelProtocol`] — Schema definition and validation for a set of channels.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::errors::LangGraphError;

// ---------------------------------------------------------------------------
// ChannelValue
// ---------------------------------------------------------------------------

/// Typed wrapper around common value kinds found in graph state.
///
/// `ChannelValue` provides convenient conversions to and from [`serde_json::Value`]
/// along with type-inspection helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelValue {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// A UTF-8 string.
    Text(String),
    /// An ordered list of JSON values.
    List(Vec<Value>),
    /// A string-keyed map of JSON values.
    Map(HashMap<String, Value>),
    /// An arbitrary JSON value.
    Json(Value),
}

impl ChannelValue {
    /// Create a `ChannelValue` from a [`serde_json::Value`].
    ///
    /// Booleans, integers, floats, strings, arrays, and objects are mapped to
    /// their corresponding variants. `null` maps to [`ChannelValue::Null`].
    pub fn from_value(value: Value) -> Self {
        match value {
            Value::Null => ChannelValue::Null,
            Value::Bool(b) => ChannelValue::Bool(b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ChannelValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    ChannelValue::Float(f)
                } else {
                    ChannelValue::Json(Value::Number(n))
                }
            }
            Value::String(s) => ChannelValue::Text(s),
            Value::Array(arr) => ChannelValue::List(arr),
            Value::Object(map) => ChannelValue::Map(map.into_iter().collect()),
        }
    }

    /// Convert this `ChannelValue` back to a [`serde_json::Value`].
    pub fn to_value(&self) -> Value {
        match self {
            ChannelValue::Null => Value::Null,
            ChannelValue::Bool(b) => Value::Bool(*b),
            ChannelValue::Int(i) => json!(*i),
            ChannelValue::Float(f) => json!(*f),
            ChannelValue::Text(s) => Value::String(s.clone()),
            ChannelValue::List(arr) => Value::Array(arr.clone()),
            ChannelValue::Map(map) => {
                let obj: serde_json::Map<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Value::Object(obj)
            }
            ChannelValue::Json(v) => v.clone(),
        }
    }

    /// Returns `true` if this value is [`ChannelValue::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, ChannelValue::Null)
    }

    /// Returns a human-readable name for this value's type.
    pub fn type_name(&self) -> &'static str {
        match self {
            ChannelValue::Null => "null",
            ChannelValue::Bool(_) => "bool",
            ChannelValue::Int(_) => "int",
            ChannelValue::Float(_) => "float",
            ChannelValue::Text(_) => "text",
            ChannelValue::List(_) => "list",
            ChannelValue::Map(_) => "map",
            ChannelValue::Json(_) => "json",
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelReducer
// ---------------------------------------------------------------------------

/// Defines how channel values are combined when a channel receives an update.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelReducer {
    /// Last write wins — the incoming value replaces the current value.
    Replace,
    /// Append incoming items to a list. Both current and incoming are coerced
    /// to arrays before appending.
    Append,
    /// Deep-merge JSON objects. Non-object values fall back to replacement.
    Merge,
    /// Add numeric values. Non-numeric values fall back to replacement.
    Sum,
    /// Keep the maximum of two numeric values. Non-numeric values fall back to
    /// replacement.
    Max,
    /// Keep the minimum of two numeric values. Non-numeric values fall back to
    /// replacement.
    Min,
    /// A named custom reducer (identifier only — actual logic is external).
    Custom(String),
}

impl ChannelReducer {
    /// Apply this reducer to combine `current` with `incoming` and return the
    /// resulting value.
    pub fn apply(&self, current: &Value, incoming: &Value) -> Value {
        match self {
            ChannelReducer::Replace => incoming.clone(),

            ChannelReducer::Append => {
                let mut list = match current {
                    Value::Array(arr) => arr.clone(),
                    Value::Null => Vec::new(),
                    other => vec![other.clone()],
                };
                match incoming {
                    Value::Array(arr) => list.extend(arr.clone()),
                    other => list.push(other.clone()),
                }
                Value::Array(list)
            }

            ChannelReducer::Merge => {
                match (current, incoming) {
                    (Value::Object(base), Value::Object(update)) => {
                        let mut merged = base.clone();
                        for (k, v) in update {
                            merged.insert(k.clone(), v.clone());
                        }
                        Value::Object(merged)
                    }
                    // Fall back to replace for non-objects.
                    _ => incoming.clone(),
                }
            }

            ChannelReducer::Sum => {
                let a = current.as_f64().unwrap_or(0.0);
                let b = incoming.as_f64().unwrap_or(0.0);
                let sum = a + b;
                // Preserve integer representation when possible.
                if sum.fract() == 0.0 && sum >= i64::MIN as f64 && sum <= i64::MAX as f64 {
                    json!(sum as i64)
                } else {
                    json!(sum)
                }
            }

            ChannelReducer::Max => {
                let a = current.as_f64();
                let b = incoming.as_f64();
                match (a, b) {
                    (Some(va), Some(vb)) => {
                        let m = va.max(vb);
                        if m.fract() == 0.0 && m >= i64::MIN as f64 && m <= i64::MAX as f64 {
                            json!(m as i64)
                        } else {
                            json!(m)
                        }
                    }
                    _ => incoming.clone(),
                }
            }

            ChannelReducer::Min => {
                let a = current.as_f64();
                let b = incoming.as_f64();
                match (a, b) {
                    (Some(va), Some(vb)) => {
                        let m = va.min(vb);
                        if m.fract() == 0.0 && m >= i64::MIN as f64 && m <= i64::MAX as f64 {
                            json!(m as i64)
                        } else {
                            json!(m)
                        }
                    }
                    _ => incoming.clone(),
                }
            }

            ChannelReducer::Custom(_) => {
                // Custom reducers are resolved externally; default to replace.
                incoming.clone()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StateChannel
// ---------------------------------------------------------------------------

/// A named channel that applies a [`ChannelReducer`] on each update and tracks
/// a monotonically increasing version number.
#[derive(Debug, Clone)]
pub struct StateChannel {
    /// The channel name.
    name: String,
    /// The current value stored in the channel.
    value: Value,
    /// The reducer applied when the channel is updated.
    reducer: ChannelReducer,
    /// Monotonically increasing version; incremented on every successful update.
    version: u64,
}

impl StateChannel {
    /// Create a new `StateChannel` with the given name and reducer.
    ///
    /// The initial value is `null`.
    pub fn new(name: impl Into<String>, reducer: ChannelReducer) -> Self {
        Self {
            name: name.into(),
            value: Value::Null,
            reducer,
            version: 0,
        }
    }

    /// Set an initial default value (builder pattern).
    pub fn with_default(mut self, value: Value) -> Self {
        self.value = value;
        self
    }

    /// Update the channel by applying the reducer to the current value and
    /// `incoming`. Increments the version counter.
    pub fn update(&mut self, incoming: Value) {
        self.value = self.reducer.apply(&self.value, &incoming);
        self.version += 1;
    }

    /// Return a reference to the current value.
    pub fn get(&self) -> &Value {
        &self.value
    }

    /// Return the current version of the channel.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Return the channel name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset the channel to `null` with version 0.
    pub fn reset(&mut self) {
        self.value = Value::Null;
        self.version = 0;
    }
}

// ---------------------------------------------------------------------------
// ChannelManager
// ---------------------------------------------------------------------------

/// Manages a collection of named [`StateChannel`]s.
///
/// Provides CRUD operations, snapshot/restore, and introspection.
#[derive(Debug, Clone)]
pub struct ChannelManager {
    channels: HashMap<String, StateChannel>,
}

impl ChannelManager {
    /// Create an empty `ChannelManager`.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Add a new channel with the given name and reducer. If a channel with the
    /// same name already exists, it is replaced.
    pub fn add_channel(&mut self, name: impl Into<String>, reducer: ChannelReducer) {
        let name = name.into();
        self.channels
            .insert(name.clone(), StateChannel::new(name, reducer));
    }

    /// Update the named channel with `value`. Returns `Ok(())` if the channel
    /// exists, or an error otherwise.
    pub fn update(&mut self, name: &str, value: Value) -> Result<(), LangGraphError> {
        let ch = self
            .channels
            .get_mut(name)
            .ok_or_else(|| LangGraphError::Other(format!("channel not found: {name}")))?;
        ch.update(value);
        Ok(())
    }

    /// Get the current value of the named channel.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.channels.get(name).map(|ch| ch.get())
    }

    /// Take a snapshot of all channel values.
    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.channels
            .iter()
            .map(|(k, ch)| (k.clone(), ch.get().clone()))
            .collect()
    }

    /// Restore channel values from a snapshot. Only channels that exist in both
    /// the manager and the snapshot are updated; extra keys in the snapshot are
    /// ignored.
    pub fn restore(&mut self, snapshot: HashMap<String, Value>) {
        for (name, value) in snapshot {
            if let Some(ch) = self.channels.get_mut(&name) {
                ch.value = value;
            }
        }
    }

    /// Return the names of all channels (in arbitrary order).
    pub fn channel_names(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }

    /// Return the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Returns `true` if a channel with the given name exists.
    pub fn has_channel(&self, name: &str) -> bool {
        self.channels.contains_key(name)
    }
}

impl Default for ChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BinaryChannel
// ---------------------------------------------------------------------------

/// A channel that holds a single value and only allows one write per superstep.
///
/// Calling [`set`](BinaryChannel::set) a second time before
/// [`advance_step`](BinaryChannel::advance_step) returns an error. This is
/// useful for channels that represent a computation result that should be
/// produced exactly once per step.
#[derive(Debug, Clone)]
pub struct BinaryChannel {
    /// The channel name.
    name: String,
    /// The stored value (if any).
    value: Option<Value>,
    /// Whether a value has been set in the current step.
    set_in_step: bool,
}

impl BinaryChannel {
    /// Create a new empty `BinaryChannel`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
            set_in_step: false,
        }
    }

    /// Set the channel value.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel has already been set in the current step.
    pub fn set(&mut self, value: Value) -> Result<(), LangGraphError> {
        if self.set_in_step {
            return Err(LangGraphError::InvalidUpdateError(format!(
                "BinaryChannel '{}' already set in this step",
                self.name
            )));
        }
        self.value = Some(value);
        self.set_in_step = true;
        Ok(())
    }

    /// Return a reference to the current value, if any.
    pub fn get(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Advance to the next superstep, allowing a new write.
    pub fn advance_step(&mut self) {
        self.set_in_step = false;
    }

    /// Returns `true` if the channel has a value (regardless of step).
    pub fn is_set(&self) -> bool {
        self.value.is_some()
    }

    /// Return the channel name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// LastValueChannel
// ---------------------------------------------------------------------------

/// A lightweight channel that always keeps only the most recent value.
///
/// Unlike the [`BaseChannel`](super::base::BaseChannel)-based [`LastValue`](super::last_value::LastValue),
/// this struct does not enforce the one-write-per-step rule and does not
/// participate in the Pregel checkpoint protocol. It is intended for simple
/// state management scenarios.
#[derive(Debug, Clone)]
pub struct LastValueChannel {
    /// The channel name.
    name: String,
    /// The stored value.
    value: Option<Value>,
}

impl LastValueChannel {
    /// Create a new empty `LastValueChannel`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    /// Update the channel with a new value, discarding any previous value.
    pub fn update(&mut self, value: Value) {
        self.value = Some(value);
    }

    /// Return a reference to the stored value, if any.
    pub fn get(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Returns `true` if the channel has a value.
    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }

    /// Return the channel name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// ChannelProtocol
// ---------------------------------------------------------------------------

/// A single channel definition within a [`ChannelProtocol`].
#[derive(Debug, Clone)]
struct ChannelDefinition {
    name: String,
    reducer: ChannelReducer,
    default: Value,
}

/// Schema that defines the expected set of channels, their reducers, and
/// default values.
///
/// A `ChannelProtocol` can be validated against a [`ChannelManager`] to ensure
/// all required channels are present.
#[derive(Debug, Clone)]
pub struct ChannelProtocol {
    definitions: Vec<ChannelDefinition>,
}

impl ChannelProtocol {
    /// Create an empty protocol.
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    /// Define a channel with the given name, reducer, and default value.
    pub fn define(&mut self, name: impl Into<String>, reducer: ChannelReducer, default: Value) {
        self.definitions.push(ChannelDefinition {
            name: name.into(),
            reducer,
            default,
        });
    }

    /// Validate that all channels defined in this protocol exist in `manager`.
    ///
    /// # Errors
    ///
    /// Returns an error listing all missing channel names.
    pub fn validate(&self, manager: &ChannelManager) -> Result<(), LangGraphError> {
        let missing: Vec<&str> = self
            .definitions
            .iter()
            .filter(|d| !manager.has_channel(&d.name))
            .map(|d| d.name.as_str())
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(LangGraphError::Other(format!(
                "missing channels: {}",
                missing.join(", ")
            )))
        }
    }

    /// Serialize the protocol to a JSON value.
    pub fn to_json(&self) -> Value {
        let defs: Vec<Value> = self
            .definitions
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "reducer": format!("{:?}", d.reducer),
                    "default": d.default,
                })
            })
            .collect();
        json!({ "channels": defs })
    }

    /// Return the number of channel definitions.
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns `true` if no channels are defined.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

impl Default for ChannelProtocol {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ChannelValue -------------------------------------------------------

    #[test]
    fn test_channel_value_from_null() {
        let cv = ChannelValue::from_value(Value::Null);
        assert_eq!(cv, ChannelValue::Null);
        assert!(cv.is_null());
        assert_eq!(cv.type_name(), "null");
    }

    #[test]
    fn test_channel_value_from_bool() {
        let cv = ChannelValue::from_value(json!(true));
        assert_eq!(cv, ChannelValue::Bool(true));
        assert!(!cv.is_null());
        assert_eq!(cv.type_name(), "bool");
    }

    #[test]
    fn test_channel_value_from_int() {
        let cv = ChannelValue::from_value(json!(42));
        assert_eq!(cv, ChannelValue::Int(42));
        assert_eq!(cv.type_name(), "int");
    }

    #[test]
    fn test_channel_value_from_float() {
        let cv = ChannelValue::from_value(json!(3.14));
        assert_eq!(cv, ChannelValue::Float(3.14));
        assert_eq!(cv.type_name(), "float");
    }

    #[test]
    fn test_channel_value_from_text() {
        let cv = ChannelValue::from_value(json!("hello"));
        assert_eq!(cv, ChannelValue::Text("hello".to_string()));
        assert_eq!(cv.type_name(), "text");
    }

    #[test]
    fn test_channel_value_from_list() {
        let cv = ChannelValue::from_value(json!([1, 2, 3]));
        assert_eq!(cv, ChannelValue::List(vec![json!(1), json!(2), json!(3)]));
        assert_eq!(cv.type_name(), "list");
    }

    #[test]
    fn test_channel_value_from_map() {
        let cv = ChannelValue::from_value(json!({"a": 1}));
        let mut expected = HashMap::new();
        expected.insert("a".to_string(), json!(1));
        assert_eq!(cv, ChannelValue::Map(expected));
        assert_eq!(cv.type_name(), "map");
    }

    #[test]
    fn test_channel_value_roundtrip_null() {
        let original = Value::Null;
        let cv = ChannelValue::from_value(original.clone());
        assert_eq!(cv.to_value(), original);
    }

    #[test]
    fn test_channel_value_roundtrip_int() {
        let cv = ChannelValue::Int(99);
        assert_eq!(cv.to_value(), json!(99));
    }

    #[test]
    fn test_channel_value_roundtrip_text() {
        let cv = ChannelValue::Text("world".to_string());
        assert_eq!(cv.to_value(), json!("world"));
    }

    #[test]
    fn test_channel_value_json_variant() {
        let cv = ChannelValue::Json(json!({"nested": [1, 2]}));
        assert_eq!(cv.type_name(), "json");
        assert_eq!(cv.to_value(), json!({"nested": [1, 2]}));
    }

    // -- ChannelReducer -----------------------------------------------------

    #[test]
    fn test_reducer_replace() {
        let r = ChannelReducer::Replace;
        let result = r.apply(&json!(1), &json!(2));
        assert_eq!(result, json!(2));
    }

    #[test]
    fn test_reducer_append_both_arrays() {
        let r = ChannelReducer::Append;
        let result = r.apply(&json!([1, 2]), &json!([3, 4]));
        assert_eq!(result, json!([1, 2, 3, 4]));
    }

    #[test]
    fn test_reducer_append_scalar_to_array() {
        let r = ChannelReducer::Append;
        let result = r.apply(&json!([1]), &json!(2));
        assert_eq!(result, json!([1, 2]));
    }

    #[test]
    fn test_reducer_append_to_null() {
        let r = ChannelReducer::Append;
        let result = r.apply(&Value::Null, &json!([1, 2]));
        assert_eq!(result, json!([1, 2]));
    }

    #[test]
    fn test_reducer_append_scalar_to_scalar() {
        let r = ChannelReducer::Append;
        let result = r.apply(&json!("a"), &json!("b"));
        assert_eq!(result, json!(["a", "b"]));
    }

    #[test]
    fn test_reducer_merge_objects() {
        let r = ChannelReducer::Merge;
        let result = r.apply(&json!({"a": 1}), &json!({"b": 2}));
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_reducer_merge_overwrites_keys() {
        let r = ChannelReducer::Merge;
        let result = r.apply(&json!({"a": 1, "b": 2}), &json!({"b": 99}));
        assert_eq!(result, json!({"a": 1, "b": 99}));
    }

    #[test]
    fn test_reducer_merge_non_objects_fallback() {
        let r = ChannelReducer::Merge;
        let result = r.apply(&json!(1), &json!(2));
        assert_eq!(result, json!(2));
    }

    #[test]
    fn test_reducer_sum_integers() {
        let r = ChannelReducer::Sum;
        let result = r.apply(&json!(10), &json!(20));
        assert_eq!(result, json!(30));
    }

    #[test]
    fn test_reducer_sum_floats() {
        let r = ChannelReducer::Sum;
        let result = r.apply(&json!(1.5), &json!(2.5));
        assert_eq!(result, json!(4));
    }

    #[test]
    fn test_reducer_sum_with_null() {
        let r = ChannelReducer::Sum;
        let result = r.apply(&Value::Null, &json!(5));
        assert_eq!(result, json!(5));
    }

    #[test]
    fn test_reducer_max() {
        let r = ChannelReducer::Max;
        assert_eq!(r.apply(&json!(3), &json!(7)), json!(7));
        assert_eq!(r.apply(&json!(10), &json!(2)), json!(10));
    }

    #[test]
    fn test_reducer_min() {
        let r = ChannelReducer::Min;
        assert_eq!(r.apply(&json!(3), &json!(7)), json!(3));
        assert_eq!(r.apply(&json!(10), &json!(2)), json!(2));
    }

    #[test]
    fn test_reducer_max_non_numeric_fallback() {
        let r = ChannelReducer::Max;
        let result = r.apply(&json!("a"), &json!("b"));
        assert_eq!(result, json!("b"));
    }

    #[test]
    fn test_reducer_min_non_numeric_fallback() {
        let r = ChannelReducer::Min;
        let result = r.apply(&json!("x"), &json!("y"));
        assert_eq!(result, json!("y"));
    }

    #[test]
    fn test_reducer_custom_defaults_to_replace() {
        let r = ChannelReducer::Custom("my_reducer".to_string());
        let result = r.apply(&json!(1), &json!(2));
        assert_eq!(result, json!(2));
    }

    // -- StateChannel -------------------------------------------------------

    #[test]
    fn test_state_channel_new() {
        let ch = StateChannel::new("counter", ChannelReducer::Sum);
        assert_eq!(ch.name(), "counter");
        assert_eq!(*ch.get(), Value::Null);
        assert_eq!(ch.version(), 0);
    }

    #[test]
    fn test_state_channel_with_default() {
        let ch = StateChannel::new("items", ChannelReducer::Append).with_default(json!([]));
        assert_eq!(*ch.get(), json!([]));
    }

    #[test]
    fn test_state_channel_update_increments_version() {
        let mut ch = StateChannel::new("v", ChannelReducer::Replace);
        assert_eq!(ch.version(), 0);
        ch.update(json!(1));
        assert_eq!(ch.version(), 1);
        ch.update(json!(2));
        assert_eq!(ch.version(), 2);
    }

    #[test]
    fn test_state_channel_update_applies_reducer() {
        let mut ch = StateChannel::new("sum", ChannelReducer::Sum).with_default(json!(0));
        ch.update(json!(5));
        assert_eq!(*ch.get(), json!(5));
        ch.update(json!(3));
        assert_eq!(*ch.get(), json!(8));
    }

    #[test]
    fn test_state_channel_reset() {
        let mut ch = StateChannel::new("x", ChannelReducer::Replace);
        ch.update(json!(42));
        assert_eq!(ch.version(), 1);
        ch.reset();
        assert_eq!(*ch.get(), Value::Null);
        assert_eq!(ch.version(), 0);
    }

    // -- ChannelManager -----------------------------------------------------

    #[test]
    fn test_manager_new_is_empty() {
        let mgr = ChannelManager::new();
        assert_eq!(mgr.channel_count(), 0);
        assert!(mgr.channel_names().is_empty());
    }

    #[test]
    fn test_manager_add_and_has() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("score", ChannelReducer::Sum);
        assert!(mgr.has_channel("score"));
        assert!(!mgr.has_channel("missing"));
        assert_eq!(mgr.channel_count(), 1);
    }

    #[test]
    fn test_manager_update_and_get() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("name", ChannelReducer::Replace);
        mgr.update("name", json!("Alice")).unwrap();
        assert_eq!(mgr.get("name"), Some(&json!("Alice")));
    }

    #[test]
    fn test_manager_update_missing_channel() {
        let mut mgr = ChannelManager::new();
        let result = mgr.update("ghost", json!(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_get_missing() {
        let mgr = ChannelManager::new();
        assert_eq!(mgr.get("nothing"), None);
    }

    #[test]
    fn test_manager_snapshot() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("a", ChannelReducer::Replace);
        mgr.add_channel("b", ChannelReducer::Replace);
        mgr.update("a", json!(1)).unwrap();
        mgr.update("b", json!(2)).unwrap();

        let snap = mgr.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap["a"], json!(1));
        assert_eq!(snap["b"], json!(2));
    }

    #[test]
    fn test_manager_restore() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("x", ChannelReducer::Replace);
        mgr.add_channel("y", ChannelReducer::Replace);

        let mut snap = HashMap::new();
        snap.insert("x".to_string(), json!(100));
        snap.insert("y".to_string(), json!(200));
        snap.insert("z".to_string(), json!(300)); // extra key, ignored

        mgr.restore(snap);
        assert_eq!(mgr.get("x"), Some(&json!(100)));
        assert_eq!(mgr.get("y"), Some(&json!(200)));
        assert!(!mgr.has_channel("z"));
    }

    #[test]
    fn test_manager_channel_names() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("b", ChannelReducer::Replace);
        mgr.add_channel("a", ChannelReducer::Replace);

        let mut names = mgr.channel_names();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_manager_replace_existing_channel() {
        let mut mgr = ChannelManager::new();
        mgr.add_channel("x", ChannelReducer::Replace);
        mgr.update("x", json!(42)).unwrap();
        // Adding again replaces the channel.
        mgr.add_channel("x", ChannelReducer::Sum);
        assert_eq!(mgr.get("x"), Some(&Value::Null));
    }

    // -- BinaryChannel ------------------------------------------------------

    #[test]
    fn test_binary_channel_new() {
        let ch = BinaryChannel::new("result");
        assert_eq!(ch.name(), "result");
        assert!(!ch.is_set());
        assert!(ch.get().is_none());
    }

    #[test]
    fn test_binary_channel_set_once() {
        let mut ch = BinaryChannel::new("result");
        ch.set(json!(42)).unwrap();
        assert!(ch.is_set());
        assert_eq!(ch.get(), Some(&json!(42)));
    }

    #[test]
    fn test_binary_channel_double_set_errors() {
        let mut ch = BinaryChannel::new("result");
        ch.set(json!(1)).unwrap();
        let result = ch.set(json!(2));
        assert!(result.is_err());
        // Value should remain the first one.
        assert_eq!(ch.get(), Some(&json!(1)));
    }

    #[test]
    fn test_binary_channel_advance_step_allows_new_set() {
        let mut ch = BinaryChannel::new("result");
        ch.set(json!("step1")).unwrap();
        ch.advance_step();
        ch.set(json!("step2")).unwrap();
        assert_eq!(ch.get(), Some(&json!("step2")));
    }

    #[test]
    fn test_binary_channel_advance_without_set() {
        let mut ch = BinaryChannel::new("result");
        ch.advance_step();
        assert!(!ch.is_set());
    }

    // -- LastValueChannel ---------------------------------------------------

    #[test]
    fn test_last_value_channel_new() {
        let ch = LastValueChannel::new("msg");
        assert_eq!(ch.name(), "msg");
        assert!(!ch.has_value());
        assert!(ch.get().is_none());
    }

    #[test]
    fn test_last_value_channel_update() {
        let mut ch = LastValueChannel::new("msg");
        ch.update(json!("hello"));
        assert!(ch.has_value());
        assert_eq!(ch.get(), Some(&json!("hello")));
    }

    #[test]
    fn test_last_value_channel_overwrites() {
        let mut ch = LastValueChannel::new("msg");
        ch.update(json!("first"));
        ch.update(json!("second"));
        assert_eq!(ch.get(), Some(&json!("second")));
    }

    // -- ChannelProtocol ----------------------------------------------------

    #[test]
    fn test_protocol_new_empty() {
        let proto = ChannelProtocol::new();
        assert!(proto.is_empty());
        assert_eq!(proto.len(), 0);
    }

    #[test]
    fn test_protocol_define() {
        let mut proto = ChannelProtocol::new();
        proto.define("score", ChannelReducer::Sum, json!(0));
        proto.define("items", ChannelReducer::Append, json!([]));
        assert_eq!(proto.len(), 2);
        assert!(!proto.is_empty());
    }

    #[test]
    fn test_protocol_validate_success() {
        let mut proto = ChannelProtocol::new();
        proto.define("a", ChannelReducer::Replace, Value::Null);
        proto.define("b", ChannelReducer::Sum, json!(0));

        let mut mgr = ChannelManager::new();
        mgr.add_channel("a", ChannelReducer::Replace);
        mgr.add_channel("b", ChannelReducer::Sum);

        assert!(proto.validate(&mgr).is_ok());
    }

    #[test]
    fn test_protocol_validate_missing_channels() {
        let mut proto = ChannelProtocol::new();
        proto.define("a", ChannelReducer::Replace, Value::Null);
        proto.define("b", ChannelReducer::Replace, Value::Null);

        let mut mgr = ChannelManager::new();
        mgr.add_channel("a", ChannelReducer::Replace);

        let result = proto.validate(&mgr);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("b"));
    }

    #[test]
    fn test_protocol_validate_empty_protocol() {
        let proto = ChannelProtocol::new();
        let mgr = ChannelManager::new();
        assert!(proto.validate(&mgr).is_ok());
    }

    #[test]
    fn test_protocol_to_json() {
        let mut proto = ChannelProtocol::new();
        proto.define("score", ChannelReducer::Sum, json!(0));

        let j = proto.to_json();
        assert!(j["channels"].is_array());
        assert_eq!(j["channels"].as_array().unwrap().len(), 1);
        assert_eq!(j["channels"][0]["name"], "score");
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_reducer_sum_non_numeric_treated_as_zero() {
        let r = ChannelReducer::Sum;
        let result = r.apply(&json!("not a number"), &json!(5));
        assert_eq!(result, json!(5));
    }

    #[test]
    fn test_reducer_append_null_incoming() {
        let r = ChannelReducer::Append;
        let result = r.apply(&json!([1]), &Value::Null);
        assert_eq!(result, json!([1, null]));
    }

    #[test]
    fn test_state_channel_multiple_appends() {
        let mut ch = StateChannel::new("log", ChannelReducer::Append).with_default(json!([]));
        ch.update(json!("entry1"));
        ch.update(json!("entry2"));
        ch.update(json!(["entry3", "entry4"]));
        assert_eq!(*ch.get(), json!(["entry1", "entry2", "entry3", "entry4"]));
    }

    #[test]
    fn test_manager_default_trait() {
        let mgr = ChannelManager::default();
        assert_eq!(mgr.channel_count(), 0);
    }

    #[test]
    fn test_protocol_default_trait() {
        let proto = ChannelProtocol::default();
        assert!(proto.is_empty());
    }

    #[test]
    fn test_channel_value_float_roundtrip() {
        let cv = ChannelValue::Float(2.718);
        let v = cv.to_value();
        let cv2 = ChannelValue::from_value(v);
        assert_eq!(cv2, ChannelValue::Float(2.718));
    }

    #[test]
    fn test_manager_snapshot_empty() {
        let mgr = ChannelManager::new();
        let snap = mgr.snapshot();
        assert!(snap.is_empty());
    }
}
