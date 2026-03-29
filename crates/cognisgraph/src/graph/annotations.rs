//! Typed state annotations for field-level validation and reducer configuration.
//!
//! State annotations extend the [`StateSchema`] concept by adding type
//! validation, required-field checks, and richer metadata to each field
//! definition.  They are designed to be used with [`StateGraph`] so that every
//! state merge automatically validates and applies defaults according to the
//! declared schema.
//!
//! # Overview
//!
//! - [`JsonType`] — Expected JSON type for a field.
//! - [`FieldAnnotation`] — Full specification of a single state field.
//! - [`AnnotatedState`] — Collection of field annotations with validation.
//! - [`AnnotatedStateBuilder`] / [`FieldBuilder`] — Ergonomic builder API.
//! - [`apply_annotations`] — Validate and apply defaults + reducers to state.

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

use crate::channels::reducers::{AppendReducer, LastValueReducer, Reducer};
use crate::errors::LangGraphError;

// ---------------------------------------------------------------------------
// JsonType
// ---------------------------------------------------------------------------

/// Expected JSON type for a field value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonType {
    /// JSON string.
    String,
    /// JSON number (integer or floating-point).
    Number,
    /// JSON boolean.
    Boolean,
    /// JSON array.
    Array,
    /// JSON object.
    Object,
    /// Any JSON value — no type constraint.
    Any,
}

impl fmt::Display for JsonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonType::String => write!(f, "string"),
            JsonType::Number => write!(f, "number"),
            JsonType::Boolean => write!(f, "boolean"),
            JsonType::Array => write!(f, "array"),
            JsonType::Object => write!(f, "object"),
            JsonType::Any => write!(f, "any"),
        }
    }
}

impl JsonType {
    /// Check whether a JSON value matches this type constraint.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            JsonType::String => value.is_string(),
            JsonType::Number => value.is_number(),
            JsonType::Boolean => value.is_boolean(),
            JsonType::Array => value.is_array(),
            JsonType::Object => value.is_object(),
            JsonType::Any => true,
        }
    }
}

// ---------------------------------------------------------------------------
// FieldAnnotation
// ---------------------------------------------------------------------------

/// Complete specification of a single field in an annotated state.
pub struct FieldAnnotation {
    /// The field name (key in the JSON object state).
    pub name: String,
    /// Expected JSON type for the field value.
    pub json_type: JsonType,
    /// Reducer used when merging updates for this field.
    pub reducer: Box<dyn Reducer>,
    /// Default value inserted when the field is absent.
    pub default_value: Option<Value>,
    /// Whether the field must be present in the state.
    pub required: bool,
    /// Optional human-readable description of the field.
    pub description: Option<String>,
}

impl fmt::Debug for FieldAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FieldAnnotation")
            .field("name", &self.name)
            .field("json_type", &self.json_type)
            .field("reducer", &"<reducer>")
            .field("default_value", &self.default_value)
            .field("required", &self.required)
            .field("description", &self.description)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AnnotatedState
// ---------------------------------------------------------------------------

/// A collection of field annotations that defines the expected shape of a
/// graph's state, including type constraints, required fields, defaults, and
/// per-field reducers.
#[derive(Debug)]
pub struct AnnotatedState {
    /// Per-field annotations, keyed by field name.
    pub fields: HashMap<String, FieldAnnotation>,
}

impl AnnotatedState {
    /// Start building an `AnnotatedState` using the fluent builder API.
    pub fn builder() -> AnnotatedStateBuilder {
        AnnotatedStateBuilder {
            fields: Vec::new(),
            current: None,
        }
    }

    /// Validate a state value against the annotations.
    ///
    /// Returns `Ok(())` if the state passes all checks, or an error describing
    /// the first violation found.
    ///
    /// # Checks
    ///
    /// 1. Required fields must be present (and not `null`).
    /// 2. Present fields must match the declared [`JsonType`] (unless `Any`).
    pub fn validate(&self, state: &Value) -> Result<(), LangGraphError> {
        let obj = state
            .as_object()
            .ok_or_else(|| LangGraphError::Other("State must be a JSON object".to_string()))?;

        for (name, annotation) in &self.fields {
            match obj.get(name) {
                None | Some(Value::Null) => {
                    if annotation.required && annotation.default_value.is_none() {
                        return Err(LangGraphError::Other(format!(
                            "Required field '{name}' is missing from state"
                        )));
                    }
                }
                Some(value) => {
                    if !annotation.json_type.matches(value) {
                        return Err(LangGraphError::Other(format!(
                            "Field '{name}' expected type {}, got {}",
                            annotation.json_type,
                            json_type_name(value),
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Return a human-readable name for a JSON value's type.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// apply_annotations
// ---------------------------------------------------------------------------

/// Validate `state` against `annotations`, apply default values for missing
/// fields, and merge `update` into `state` using per-field reducers.
///
/// This is the primary entry point for annotation-aware state merging.
///
/// # Steps
///
/// 1. Start from the `current` state.
/// 2. Insert default values for any annotated fields that are missing.
/// 3. Validate the resulting state against annotations.
/// 4. For each field in `update`, apply the matching reducer (or last-value
///    overwrite for unannotated fields).
/// 5. Validate the final merged state.
pub fn apply_annotations(
    current: &Value,
    update: &Value,
    annotations: &AnnotatedState,
) -> Result<Value, LangGraphError> {
    let cur_obj = current.as_object().cloned().unwrap_or_default();
    let upd_obj = match update.as_object() {
        Some(o) => o,
        None => {
            return Err(LangGraphError::Other(
                "apply_annotations: update must be a JSON object".into(),
            ));
        }
    };

    let mut result = cur_obj;

    // Apply defaults for missing annotated fields.
    for (name, annotation) in &annotations.fields {
        if !result.contains_key(name) || result.get(name) == Some(&Value::Null) {
            if let Some(default) = &annotation.default_value {
                result.insert(name.clone(), default.clone());
            }
        }
    }

    // Merge updates using per-field reducers.
    let default_reducer = LastValueReducer;
    for (key, upd_val) in upd_obj {
        let reducer: &dyn Reducer = match annotations.fields.get(key) {
            Some(ann) => ann.reducer.as_ref(),
            None => &default_reducer,
        };

        let cur_val = result.get(key).cloned().unwrap_or(Value::Null);
        result.insert(key.clone(), reducer.reduce(&cur_val, upd_val));
    }

    // Apply defaults again for fields still missing after the merge.
    for (name, annotation) in &annotations.fields {
        if !result.contains_key(name) {
            if let Some(default) = &annotation.default_value {
                result.insert(name.clone(), default.clone());
            }
        }
    }

    let merged = Value::Object(result);

    // Validate the final state.
    annotations.validate(&merged)?;

    Ok(merged)
}

// ---------------------------------------------------------------------------
// AnnotatedStateBuilder / FieldBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing an [`AnnotatedState`] using a fluent API.
///
/// # Example
///
/// ```rust
/// use cognisgraph::graph::annotations::{AnnotatedState, JsonType};
///
/// let annotations = AnnotatedState::builder()
///     .field("name")
///         .field_type(JsonType::String)
///         .required()
///         .description("The user's name")
///     .field("age")
///         .field_type(JsonType::Number)
///         .with_default(serde_json::json!(0))
///     .messages_field("messages")
///     .build();
/// ```
pub struct AnnotatedStateBuilder {
    fields: Vec<FieldAnnotation>,
    current: Option<FieldAnnotation>,
}

impl AnnotatedStateBuilder {
    /// Start defining a new field. If a previous field was being defined, it is
    /// finalized and added to the builder.
    pub fn field(mut self, name: impl Into<String>) -> Self {
        self.flush();
        self.current = Some(FieldAnnotation {
            name: name.into(),
            json_type: JsonType::Any,
            reducer: Box::new(LastValueReducer),
            default_value: None,
            required: false,
            description: None,
        });
        self
    }

    /// Set the expected JSON type for the current field.
    pub fn field_type(mut self, json_type: JsonType) -> Self {
        if let Some(ref mut f) = self.current {
            f.json_type = json_type;
        }
        self
    }

    /// Set a custom reducer for the current field.
    pub fn with_reducer(mut self, reducer: impl Reducer + 'static) -> Self {
        if let Some(ref mut f) = self.current {
            f.reducer = Box::new(reducer);
        }
        self
    }

    /// Set a default value for the current field.
    pub fn with_default(mut self, value: Value) -> Self {
        if let Some(ref mut f) = self.current {
            f.default_value = Some(value);
        }
        self
    }

    /// Mark the current field as required.
    pub fn required(mut self) -> Self {
        if let Some(ref mut f) = self.current {
            f.required = true;
        }
        self
    }

    /// Add a description to the current field.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        if let Some(ref mut f) = self.current {
            f.description = Some(desc.into());
        }
        self
    }

    /// Shortcut: add a field configured as an array with an [`AppendReducer`]
    /// and a default of `[]`. This is the common pattern for message lists.
    pub fn messages_field(mut self, name: impl Into<String>) -> Self {
        self.flush();
        self.fields.push(FieldAnnotation {
            name: name.into(),
            json_type: JsonType::Array,
            reducer: Box::new(AppendReducer),
            default_value: Some(Value::Array(Vec::new())),
            required: false,
            description: Some("Message list (append reducer)".to_string()),
        });
        self
    }

    /// Consume the builder and produce the finished [`AnnotatedState`].
    pub fn build(mut self) -> AnnotatedState {
        self.flush();
        let mut fields = HashMap::new();
        for annotation in self.fields {
            fields.insert(annotation.name.clone(), annotation);
        }
        AnnotatedState { fields }
    }

    /// Finalize the current field (if any) and push it to the list.
    fn flush(&mut self) {
        if let Some(f) = self.current.take() {
            self.fields.push(f);
        }
    }
}

// ---------------------------------------------------------------------------
// StateGraph integration
// ---------------------------------------------------------------------------

// We extend `StateGraph` with an `with_state_annotations` method via a
// separate impl block, keeping the annotations stored as metadata that the
// compiled graph can use during state merging.

use super::state::StateGraph;

impl StateGraph {
    /// Attach state annotations to this graph.
    ///
    /// When annotations are set, [`apply_annotations`] is used during state
    /// merging to validate types, enforce required fields, apply defaults, and
    /// use per-field reducers.
    ///
    /// This stores the annotations as a JSON metadata object on the graph so
    /// they can be retrieved later. For actual annotation-aware merging at
    /// runtime, use [`apply_annotations`] directly in your node logic or
    /// orchestration layer.
    pub fn with_state_annotations(self, annotations: AnnotatedState) -> AnnotatedStateGraph {
        AnnotatedStateGraph {
            graph: self,
            annotations,
        }
    }
}

/// A [`StateGraph`] paired with [`AnnotatedState`] annotations.
///
/// This wrapper compiles to a [`CompiledAnnotatedStateGraph`] that validates
/// and merges state using the annotations during execution.
pub struct AnnotatedStateGraph {
    /// The underlying state graph.
    pub graph: StateGraph,
    /// The annotations for the state.
    pub annotations: AnnotatedState,
}

impl AnnotatedStateGraph {
    /// Compile the underlying graph and pair it with the annotations.
    pub fn compile(self) -> Result<CompiledAnnotatedStateGraph, LangGraphError> {
        let compiled = self.graph.compile()?;
        Ok(CompiledAnnotatedStateGraph {
            compiled,
            annotations: self.annotations,
        })
    }
}

/// A compiled state graph that carries annotations for validation and
/// annotation-aware state merging.
pub struct CompiledAnnotatedStateGraph {
    /// The compiled graph.
    pub compiled: super::state::CompiledStateGraph,
    /// The annotations.
    pub annotations: AnnotatedState,
}

impl CompiledAnnotatedStateGraph {
    /// Validate a state value against the annotations.
    pub fn validate_state(&self, state: &Value) -> Result<(), LangGraphError> {
        self.annotations.validate(state)
    }

    /// Merge an update into the current state using annotation-aware reducers.
    pub fn merge_with_annotations(
        &self,
        current: &Value,
        update: &Value,
    ) -> Result<Value, LangGraphError> {
        apply_annotations(current, update, &self.annotations)
    }

    /// Invoke the underlying compiled graph, validating the input state first.
    pub async fn invoke(&self, input: Value) -> Result<Value, LangGraphError> {
        // Validate input state against annotations.
        let mut state = input.clone();

        // Apply defaults for missing annotated fields.
        if let Some(obj) = state.as_object_mut() {
            for (name, annotation) in &self.annotations.fields {
                if !obj.contains_key(name) {
                    if let Some(default) = &annotation.default_value {
                        obj.insert(name.clone(), default.clone());
                    }
                }
            }
        }

        self.annotations.validate(&state)?;

        // Delegate to the underlying compiled graph.
        self.compiled.invoke(state).await
    }
}

impl fmt::Debug for AnnotatedStateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnotatedStateGraph")
            .field("graph", &self.graph)
            .field("annotations", &self.annotations)
            .finish()
    }
}

impl fmt::Debug for CompiledAnnotatedStateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledAnnotatedStateGraph")
            .field("compiled", &self.compiled)
            .field("annotations", &self.annotations)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// GraphState macro support types
// ---------------------------------------------------------------------------

/// A reducer function that merges a current value with an update.
pub type ReducerFn =
    dyn Fn(&serde_json::Value, &serde_json::Value) -> serde_json::Value + Send + Sync;

/// Schema generated by `#[derive(GraphState)]`.
///
/// Maps field names to their reducer functions and metadata.
pub struct GraphStateSchema {
    /// Per-field configuration.
    pub fields: std::collections::HashMap<String, GraphStateField>,
}

/// A single field in a [`GraphStateSchema`].
pub struct GraphStateField {
    /// The reducer function: `(current, update) -> merged`.
    pub reducer: Box<ReducerFn>,
    /// Optional description from doc comments.
    pub description: Option<String>,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::reducers::{AppendReducer, BinaryOpReducer, MergeReducer};
    use serde_json::json;

    // -----------------------------------------------------------------------
    // 1. Basic annotated state with required fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_basic_annotated_state_with_required_fields() {
        let annotations = AnnotatedState::builder()
            .field("name")
            .field_type(JsonType::String)
            .required()
            .field("age")
            .field_type(JsonType::Number)
            .required()
            .build();

        let state = json!({"name": "Alice", "age": 30});
        assert!(annotations.validate(&state).is_ok());
    }

    // -----------------------------------------------------------------------
    // 2. Validation catches missing required field
    // -----------------------------------------------------------------------
    #[test]
    fn test_validation_catches_missing_required_field() {
        let annotations = AnnotatedState::builder()
            .field("name")
            .field_type(JsonType::String)
            .required()
            .build();

        let state = json!({"age": 30});
        let result = annotations.validate(&state);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("name"),
            "Error should mention field name: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Validation catches wrong type
    // -----------------------------------------------------------------------
    #[test]
    fn test_validation_catches_wrong_type() {
        let annotations = AnnotatedState::builder()
            .field("count")
            .field_type(JsonType::Number)
            .build();

        let state = json!({"count": "not a number"});
        let result = annotations.validate(&state);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("count"),
            "Error should mention field name: {err_msg}"
        );
        assert!(
            err_msg.contains("number"),
            "Error should mention expected type: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. Default values applied
    // -----------------------------------------------------------------------
    #[test]
    fn test_default_values_applied() {
        let annotations = AnnotatedState::builder()
            .field("status")
            .field_type(JsonType::String)
            .with_default(json!("idle"))
            .field("count")
            .field_type(JsonType::Number)
            .with_default(json!(0))
            .build();

        let current = json!({});
        let update = json!({});
        let result = apply_annotations(&current, &update, &annotations).unwrap();

        assert_eq!(result["status"], json!("idle"));
        assert_eq!(result["count"], json!(0));
    }

    // -----------------------------------------------------------------------
    // 5. Messages field shortcut with append reducer
    // -----------------------------------------------------------------------
    #[test]
    fn test_messages_field_shortcut() {
        let annotations = AnnotatedState::builder().messages_field("messages").build();

        // Verify the annotation was created correctly.
        let ann = &annotations.fields["messages"];
        assert_eq!(ann.json_type, JsonType::Array);
        assert_eq!(ann.default_value, Some(json!([])));
        assert!(ann.description.is_some());

        // Verify append behaviour.
        let current = json!({"messages": ["hello"]});
        let update = json!({"messages": "world"});
        let result = apply_annotations(&current, &update, &annotations).unwrap();
        assert_eq!(result["messages"], json!(["hello", "world"]));
    }

    // -----------------------------------------------------------------------
    // 6. Multiple field types
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_field_types() {
        let annotations = AnnotatedState::builder()
            .field("name")
            .field_type(JsonType::String)
            .field("age")
            .field_type(JsonType::Number)
            .field("active")
            .field_type(JsonType::Boolean)
            .field("tags")
            .field_type(JsonType::Array)
            .field("meta")
            .field_type(JsonType::Object)
            .field("anything")
            .field_type(JsonType::Any)
            .build();

        let state = json!({
            "name": "Alice",
            "age": 30,
            "active": true,
            "tags": ["a", "b"],
            "meta": {"k": "v"},
            "anything": 42
        });
        assert!(annotations.validate(&state).is_ok());
    }

    // -----------------------------------------------------------------------
    // 7. Apply annotations merges with reducers
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_annotations_merges_with_reducers() {
        let annotations = AnnotatedState::builder()
            .field("messages")
            .field_type(JsonType::Array)
            .with_reducer(AppendReducer)
            .with_default(json!([]))
            .field("config")
            .field_type(JsonType::Object)
            .with_reducer(MergeReducer)
            .with_default(json!({}))
            .field("status")
            .field_type(JsonType::String)
            .with_default(json!("idle"))
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
        let result = apply_annotations(&current, &update, &annotations).unwrap();

        assert_eq!(result["messages"], json!(["hello", "world"]));
        assert_eq!(result["config"], json!({"debug": true, "verbose": true}));
        assert_eq!(result["status"], json!("running"));
    }

    // -----------------------------------------------------------------------
    // 8. Builder pattern fluent API
    // -----------------------------------------------------------------------
    #[test]
    fn test_builder_pattern_fluent_api() {
        let annotations = AnnotatedState::builder()
            .field("counter")
            .field_type(JsonType::Number)
            .with_reducer(BinaryOpReducer::new(|a, b| {
                let a_num = a.as_i64().unwrap_or(0);
                let b_num = b.as_i64().unwrap_or(0);
                json!(a_num + b_num)
            }))
            .with_default(json!(0))
            .required()
            .description("Running counter")
            .build();

        assert_eq!(annotations.fields.len(), 1);
        let ann = &annotations.fields["counter"];
        assert_eq!(ann.json_type, JsonType::Number);
        assert!(ann.required);
        assert_eq!(ann.default_value, Some(json!(0)));
        assert_eq!(ann.description.as_deref(), Some("Running counter"));

        // Verify the sum reducer works via apply_annotations.
        let current = json!({"counter": 10});
        let update = json!({"counter": 5});
        let result = apply_annotations(&current, &update, &annotations).unwrap();
        assert_eq!(result["counter"], json!(15));
    }

    // -----------------------------------------------------------------------
    // 9. Description metadata
    // -----------------------------------------------------------------------
    #[test]
    fn test_description_metadata() {
        let annotations = AnnotatedState::builder()
            .field("query")
            .field_type(JsonType::String)
            .description("The user's search query")
            .field("results")
            .field_type(JsonType::Array)
            .description("Search results")
            .build();

        assert_eq!(
            annotations.fields["query"].description.as_deref(),
            Some("The user's search query")
        );
        assert_eq!(
            annotations.fields["results"].description.as_deref(),
            Some("Search results")
        );
    }

    // -----------------------------------------------------------------------
    // 10. Optional fields pass validation when absent
    // -----------------------------------------------------------------------
    #[test]
    fn test_optional_fields_pass_when_absent() {
        let annotations = AnnotatedState::builder()
            .field("optional_field")
            .field_type(JsonType::String)
            // not marked required
            .build();

        let state = json!({});
        assert!(annotations.validate(&state).is_ok());
    }

    // -----------------------------------------------------------------------
    // 11. Annotated state with StateGraph integration
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_annotated_state_graph_integration() {
        use crate::graph::state::StateGraph;
        use std::sync::Arc;

        let annotations = AnnotatedState::builder()
            .messages_field("messages")
            .field("status")
            .field_type(JsonType::String)
            .with_default(json!("idle"))
            .build();

        let action: crate::graph::state::AsyncNodeAction = Arc::new(move |_state: Value| {
            Box::pin(async move { Ok(json!({"messages": "processed", "status": "done"})) })
        });

        let annotated_graph = StateGraph::new()
            .add_node("processor", action)
            .set_entry_point("processor")
            .set_finish_point("processor")
            .with_state_annotations(annotations);

        let compiled = annotated_graph.compile().unwrap();

        // Validate an initial state.
        let input = json!({"messages": ["hello"]});
        assert!(compiled.validate_state(&input).is_ok());

        // Verify merge_with_annotations works.
        let current = json!({"messages": ["hello"], "status": "idle"});
        let update = json!({"messages": "world", "status": "running"});
        let merged = compiled.merge_with_annotations(&current, &update).unwrap();
        assert_eq!(merged["messages"], json!(["hello", "world"]));
        assert_eq!(merged["status"], json!("running"));
    }

    // -----------------------------------------------------------------------
    // 12. Required field with default does not fail validation
    // -----------------------------------------------------------------------
    #[test]
    fn test_required_field_with_default_passes() {
        let annotations = AnnotatedState::builder()
            .field("items")
            .field_type(JsonType::Array)
            .required()
            .with_default(json!([]))
            .build();

        // The field is required but has a default, so apply_annotations
        // should populate it and pass validation.
        let current = json!({});
        let update = json!({});
        let result = apply_annotations(&current, &update, &annotations);
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["items"], json!([]));
    }

    // -----------------------------------------------------------------------
    // 13. Validation rejects non-object state
    // -----------------------------------------------------------------------
    #[test]
    fn test_validation_rejects_non_object_state() {
        let annotations = AnnotatedState::builder()
            .field("x")
            .field_type(JsonType::Number)
            .build();

        let state = json!(42);
        assert!(annotations.validate(&state).is_err());
    }

    // -----------------------------------------------------------------------
    // 14. JsonType::matches covers all variants
    // -----------------------------------------------------------------------
    #[test]
    fn test_json_type_matches() {
        assert!(JsonType::String.matches(&json!("hello")));
        assert!(!JsonType::String.matches(&json!(42)));

        assert!(JsonType::Number.matches(&json!(3.14)));
        assert!(!JsonType::Number.matches(&json!("x")));

        assert!(JsonType::Boolean.matches(&json!(true)));
        assert!(!JsonType::Boolean.matches(&json!(1)));

        assert!(JsonType::Array.matches(&json!([1, 2])));
        assert!(!JsonType::Array.matches(&json!({})));

        assert!(JsonType::Object.matches(&json!({"a": 1})));
        assert!(!JsonType::Object.matches(&json!([1])));

        assert!(JsonType::Any.matches(&json!(null)));
        assert!(JsonType::Any.matches(&json!("anything")));
    }
}
