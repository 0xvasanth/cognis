//! Type coercion and conversion utilities for the LLM framework.
//!
//! Safely converts between different value representations used across the pipeline,
//! with configurable strictness levels, validation schemas, and dot-notation path access.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// CoercionError
// ---------------------------------------------------------------------------

/// Errors that can occur during type coercion.
#[derive(Debug, Clone, PartialEq)]
pub enum CoercionError {
    /// The source type cannot be converted to the target type.
    IncompatibleTypes {
        from: String,
        to: String,
    },
    /// The conversion would lose information and strict mode is enabled.
    LossyConversion {
        from: String,
        to: String,
    },
    /// The value is outside the representable range of the target type.
    OutOfRange {
        value: String,
        target: String,
    },
    /// A string-to-typed parse failed.
    ParseError(String),
}

impl fmt::Display for CoercionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoercionError::IncompatibleTypes { from, to } => {
                write!(f, "cannot convert from {} to {}", from, to)
            }
            CoercionError::LossyConversion { from, to } => {
                write!(f, "lossy conversion from {} to {}", from, to)
            }
            CoercionError::OutOfRange { value, target } => {
                write!(f, "value {} out of range for {}", value, target)
            }
            CoercionError::ParseError(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for CoercionError {}

// ---------------------------------------------------------------------------
// ValueType
// ---------------------------------------------------------------------------

/// Describes a value's type tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueType {
    String,
    Integer,
    Float,
    Bool,
    List,
    Map,
    Bytes,
    Null,
    /// Matches any value type.
    Any,
}

impl ValueType {
    /// Returns `true` if a [`DynamicValue`] matches this type tag.
    pub fn matches(&self, val: &DynamicValue) -> bool {
        match self {
            ValueType::Any => true,
            ValueType::String => matches!(val, DynamicValue::String(_)),
            ValueType::Integer => matches!(val, DynamicValue::Integer(_)),
            ValueType::Float => matches!(val, DynamicValue::Float(_)),
            ValueType::Bool => matches!(val, DynamicValue::Bool(_)),
            ValueType::List => matches!(val, DynamicValue::List(_)),
            ValueType::Map => matches!(val, DynamicValue::Map(_)),
            ValueType::Bytes => matches!(val, DynamicValue::Bytes(_)),
            ValueType::Null => matches!(val, DynamicValue::Null),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ValueType::String => "String",
            ValueType::Integer => "Integer",
            ValueType::Float => "Float",
            ValueType::Bool => "Bool",
            ValueType::List => "List",
            ValueType::Map => "Map",
            ValueType::Bytes => "Bytes",
            ValueType::Null => "Null",
            ValueType::Any => "Any",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// DynamicValue
// ---------------------------------------------------------------------------

/// A dynamically-typed value used across the LLM pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<DynamicValue>),
    Map(HashMap<String, DynamicValue>),
    Bytes(Vec<u8>),
    Null,
}

impl DynamicValue {
    /// Returns a human-readable name for the value's type.
    pub fn type_name(&self) -> &'static str {
        match self {
            DynamicValue::String(_) => "String",
            DynamicValue::Integer(_) => "Integer",
            DynamicValue::Float(_) => "Float",
            DynamicValue::Bool(_) => "Bool",
            DynamicValue::List(_) => "List",
            DynamicValue::Map(_) => "Map",
            DynamicValue::Bytes(_) => "Bytes",
            DynamicValue::Null => "Null",
        }
    }

    /// Returns `true` if this value is [`DynamicValue::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, DynamicValue::Null)
    }

    /// Returns `true` if the value is "truthy" (non-null, non-zero, non-empty).
    pub fn is_truthy(&self) -> bool {
        match self {
            DynamicValue::Null => false,
            DynamicValue::Bool(b) => *b,
            DynamicValue::Integer(i) => *i != 0,
            DynamicValue::Float(f) => *f != 0.0,
            DynamicValue::String(s) => !s.is_empty(),
            DynamicValue::List(l) => !l.is_empty(),
            DynamicValue::Map(m) => !m.is_empty(),
            DynamicValue::Bytes(b) => !b.is_empty(),
        }
    }

    /// Converts this value to a [`serde_json::Value`].
    pub fn to_json(&self) -> Value {
        match self {
            DynamicValue::String(s) => Value::String(s.clone()),
            DynamicValue::Integer(i) => Value::Number((*i).into()),
            DynamicValue::Float(f) => {
                serde_json::Number::from_f64(*f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }
            DynamicValue::Bool(b) => Value::Bool(*b),
            DynamicValue::List(l) => Value::Array(l.iter().map(|v| v.to_json()).collect()),
            DynamicValue::Map(m) => {
                let obj = m.iter().map(|(k, v)| (k.clone(), v.to_json())).collect();
                Value::Object(obj)
            }
            DynamicValue::Bytes(b) => {
                Value::String(base64_encode(b))
            }
            DynamicValue::Null => Value::Null,
        }
    }

    /// Creates a [`DynamicValue`] from a [`serde_json::Value`].
    pub fn from_json(value: &Value) -> Self {
        match value {
            Value::Null => DynamicValue::Null,
            Value::Bool(b) => DynamicValue::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    DynamicValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    DynamicValue::Float(f)
                } else {
                    DynamicValue::Null
                }
            }
            Value::String(s) => DynamicValue::String(s.clone()),
            Value::Array(arr) => {
                DynamicValue::List(arr.iter().map(DynamicValue::from_json).collect())
            }
            Value::Object(obj) => {
                let map = obj.iter().map(|(k, v)| (k.clone(), DynamicValue::from_json(v))).collect();
                DynamicValue::Map(map)
            }
        }
    }

    // -- typed accessors --

    /// Returns the inner string if this is a [`DynamicValue::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DynamicValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner i64 if this is a [`DynamicValue::Integer`].
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            DynamicValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the inner f64 if this is a [`DynamicValue::Float`].
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DynamicValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the inner bool if this is a [`DynamicValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DynamicValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns a reference to the inner list if this is a [`DynamicValue::List`].
    pub fn as_list(&self) -> Option<&Vec<DynamicValue>> {
        match self {
            DynamicValue::List(l) => Some(l),
            _ => None,
        }
    }

    /// Returns a reference to the inner map if this is a [`DynamicValue::Map`].
    pub fn as_map(&self) -> Option<&HashMap<String, DynamicValue>> {
        match self {
            DynamicValue::Map(m) => Some(m),
            _ => None,
        }
    }
}

// -- From impls --

impl From<String> for DynamicValue {
    fn from(s: String) -> Self {
        DynamicValue::String(s)
    }
}

impl From<&str> for DynamicValue {
    fn from(s: &str) -> Self {
        DynamicValue::String(s.to_owned())
    }
}

impl From<i64> for DynamicValue {
    fn from(i: i64) -> Self {
        DynamicValue::Integer(i)
    }
}

impl From<f64> for DynamicValue {
    fn from(f: f64) -> Self {
        DynamicValue::Float(f)
    }
}

impl From<bool> for DynamicValue {
    fn from(b: bool) -> Self {
        DynamicValue::Bool(b)
    }
}

impl From<Vec<DynamicValue>> for DynamicValue {
    fn from(v: Vec<DynamicValue>) -> Self {
        DynamicValue::List(v)
    }
}

impl From<Value> for DynamicValue {
    fn from(v: Value) -> Self {
        DynamicValue::from_json(&v)
    }
}

// ---------------------------------------------------------------------------
// TypeCoercer
// ---------------------------------------------------------------------------

/// Converts between [`DynamicValue`] types with configurable strictness.
#[derive(Debug, Clone)]
pub struct TypeCoercer {
    allow_lossy: bool,
}

impl TypeCoercer {
    /// Creates a new coercer with default settings (lossy conversions allowed).
    pub fn new() -> Self {
        Self { allow_lossy: true }
    }

    /// Creates a strict coercer that rejects lossy conversions.
    pub fn strict() -> Self {
        Self { allow_lossy: false }
    }

    /// Creates a lenient coercer that allows lossy conversions.
    pub fn lenient() -> Self {
        Self { allow_lossy: true }
    }

    /// Coerce the value to a `String`.
    pub fn to_string(&self, val: &DynamicValue) -> Result<String, CoercionError> {
        match val {
            DynamicValue::String(s) => Ok(s.clone()),
            DynamicValue::Integer(i) => Ok(i.to_string()),
            DynamicValue::Float(f) => Ok(f.to_string()),
            DynamicValue::Bool(b) => Ok(b.to_string()),
            DynamicValue::Null => Ok("null".to_owned()),
            DynamicValue::Bytes(b) => Ok(base64_encode(b)),
            DynamicValue::List(_) | DynamicValue::Map(_) => {
                if self.allow_lossy {
                    Ok(serde_json::to_string(&val.to_json()).unwrap_or_default())
                } else {
                    Err(CoercionError::LossyConversion {
                        from: val.type_name().to_owned(),
                        to: "String".to_owned(),
                    })
                }
            }
        }
    }

    /// Coerce the value to an `i64`.
    pub fn to_integer(&self, val: &DynamicValue) -> Result<i64, CoercionError> {
        match val {
            DynamicValue::Integer(i) => Ok(*i),
            DynamicValue::Float(f) => {
                if !self.allow_lossy && *f != (*f as i64) as f64 {
                    return Err(CoercionError::LossyConversion {
                        from: "Float".to_owned(),
                        to: "Integer".to_owned(),
                    });
                }
                let i = *f as i64;
                Ok(i)
            }
            DynamicValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            DynamicValue::String(s) => {
                s.parse::<i64>().map_err(|_| CoercionError::ParseError(
                    format!("cannot parse '{}' as integer", s),
                ))
            }
            _ => Err(CoercionError::IncompatibleTypes {
                from: val.type_name().to_owned(),
                to: "Integer".to_owned(),
            }),
        }
    }

    /// Coerce the value to an `f64`.
    pub fn to_float(&self, val: &DynamicValue) -> Result<f64, CoercionError> {
        match val {
            DynamicValue::Float(f) => Ok(*f),
            DynamicValue::Integer(i) => Ok(*i as f64),
            DynamicValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            DynamicValue::String(s) => {
                s.parse::<f64>().map_err(|_| CoercionError::ParseError(
                    format!("cannot parse '{}' as float", s),
                ))
            }
            _ => Err(CoercionError::IncompatibleTypes {
                from: val.type_name().to_owned(),
                to: "Float".to_owned(),
            }),
        }
    }

    /// Coerce the value to a `bool`.
    pub fn to_bool(&self, val: &DynamicValue) -> Result<bool, CoercionError> {
        match val {
            DynamicValue::Bool(b) => Ok(*b),
            DynamicValue::Integer(i) => Ok(*i != 0),
            DynamicValue::Float(f) => Ok(*f != 0.0),
            DynamicValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" => Ok(true),
                "false" | "0" | "no" => Ok(false),
                _ => Err(CoercionError::ParseError(
                    format!("cannot parse '{}' as bool", s),
                )),
            },
            DynamicValue::Null => Ok(false),
            _ => Err(CoercionError::IncompatibleTypes {
                from: val.type_name().to_owned(),
                to: "Bool".to_owned(),
            }),
        }
    }

    /// Coerce the value to the given target [`ValueType`].
    pub fn coerce(
        &self,
        val: &DynamicValue,
        target: ValueType,
    ) -> Result<DynamicValue, CoercionError> {
        match target {
            ValueType::Any => Ok(val.clone()),
            ValueType::String => self.to_string(val).map(DynamicValue::String),
            ValueType::Integer => self.to_integer(val).map(DynamicValue::Integer),
            ValueType::Float => self.to_float(val).map(DynamicValue::Float),
            ValueType::Bool => self.to_bool(val).map(DynamicValue::Bool),
            ValueType::Null => Ok(DynamicValue::Null),
            ValueType::List => {
                if matches!(val, DynamicValue::List(_)) {
                    Ok(val.clone())
                } else {
                    Err(CoercionError::IncompatibleTypes {
                        from: val.type_name().to_owned(),
                        to: "List".to_owned(),
                    })
                }
            }
            ValueType::Map => {
                if matches!(val, DynamicValue::Map(_)) {
                    Ok(val.clone())
                } else {
                    Err(CoercionError::IncompatibleTypes {
                        from: val.type_name().to_owned(),
                        to: "Map".to_owned(),
                    })
                }
            }
            ValueType::Bytes => {
                if matches!(val, DynamicValue::Bytes(_)) {
                    Ok(val.clone())
                } else {
                    Err(CoercionError::IncompatibleTypes {
                        from: val.type_name().to_owned(),
                        to: "Bytes".to_owned(),
                    })
                }
            }
        }
    }
}

impl Default for TypeCoercer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ValueConstraint
// ---------------------------------------------------------------------------

/// A constraint that can be applied to a [`DynamicValue`] during validation.
#[derive(Debug, Clone)]
pub enum ValueConstraint {
    /// Minimum length for strings, lists, bytes.
    MinLength(usize),
    /// Maximum length for strings, lists, bytes.
    MaxLength(usize),
    /// Minimum numeric value (inclusive).
    MinValue(f64),
    /// Maximum numeric value (inclusive).
    MaxValue(f64),
    /// Regex pattern the string value must match.
    Pattern(String),
    /// The value must be one of the given options.
    OneOf(Vec<DynamicValue>),
}

impl ValueConstraint {
    /// Returns `true` if the given value satisfies this constraint.
    pub fn check(&self, val: &DynamicValue) -> bool {
        match self {
            ValueConstraint::MinLength(min) => {
                let len = value_length(val);
                len.map_or(true, |l| l >= *min)
            }
            ValueConstraint::MaxLength(max) => {
                let len = value_length(val);
                len.map_or(true, |l| l <= *max)
            }
            ValueConstraint::MinValue(min) => {
                let num = value_as_f64(val);
                num.map_or(true, |n| n >= *min)
            }
            ValueConstraint::MaxValue(max) => {
                let num = value_as_f64(val);
                num.map_or(true, |n| n <= *max)
            }
            ValueConstraint::Pattern(pat) => {
                if let DynamicValue::String(s) = val {
                    regex::Regex::new(pat).map_or(false, |re| re.is_match(s))
                } else {
                    true
                }
            }
            ValueConstraint::OneOf(options) => options.contains(val),
        }
    }
}

impl fmt::Display for ValueConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueConstraint::MinLength(n) => write!(f, "min_length({})", n),
            ValueConstraint::MaxLength(n) => write!(f, "max_length({})", n),
            ValueConstraint::MinValue(v) => write!(f, "min_value({})", v),
            ValueConstraint::MaxValue(v) => write!(f, "max_value({})", v),
            ValueConstraint::Pattern(p) => write!(f, "pattern({})", p),
            ValueConstraint::OneOf(opts) => write!(f, "one_of({} options)", opts.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// ValueSchema
// ---------------------------------------------------------------------------

/// Describes the expected shape and constraints of a value.
#[derive(Debug, Clone)]
pub struct ValueSchema {
    /// The expected type.
    pub value_type: ValueType,
    /// Whether the value is required.
    pub is_required: bool,
    /// Optional default for non-required values.
    pub default: Option<DynamicValue>,
    /// Additional constraints.
    pub constraints: Vec<ValueConstraint>,
}

impl ValueSchema {
    /// Create a new schema that expects the given type, required by default.
    pub fn new(value_type: ValueType) -> Self {
        Self {
            value_type,
            is_required: true,
            default: None,
            constraints: Vec::new(),
        }
    }

    /// Mark this schema as required (the default).
    pub fn required(mut self) -> Self {
        self.is_required = true;
        self.default = None;
        self
    }

    /// Mark this schema as optional with a default value.
    pub fn optional(mut self, default: DynamicValue) -> Self {
        self.is_required = false;
        self.default = Some(default);
        self
    }

    /// Add a constraint.
    pub fn with_constraint(mut self, constraint: ValueConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Validate a value against this schema, returning a list of error messages on failure.
    pub fn validate(&self, val: &DynamicValue) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check null / required
        if val.is_null() {
            if self.is_required {
                errors.push("value is required but got Null".to_owned());
            }
            // If null is accepted (optional), skip further checks
            if !errors.is_empty() {
                return Err(errors);
            }
            return Ok(());
        }

        // Type check
        if self.value_type != ValueType::Any && !self.value_type.matches(val) {
            errors.push(format!(
                "expected type {} but got {}",
                self.value_type,
                val.type_name()
            ));
        }

        // Constraints
        for c in &self.constraints {
            if !c.check(val) {
                errors.push(format!("constraint {} not satisfied", c));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// ValuePath
// ---------------------------------------------------------------------------

/// A dot-notation path for accessing nested values in a [`DynamicValue`].
///
/// Supports map keys separated by `.` and list indices as numeric segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValuePath {
    raw: String,
    parts: Vec<String>,
}

impl ValuePath {
    /// Parse a dot-notation path string (e.g. `"a.b.0.c"`).
    pub fn new(path: &str) -> Self {
        let parts: Vec<String> = path.split('.').map(|s| s.to_owned()).collect();
        Self {
            raw: path.to_owned(),
            parts,
        }
    }

    /// Returns the path segments.
    pub fn segments(&self) -> Vec<String> {
        self.parts.clone()
    }

    /// Returns the number of segments.
    pub fn depth(&self) -> usize {
        self.parts.len()
    }

    /// Traverse the value tree and return a reference to the target.
    pub fn get<'a>(&self, val: &'a DynamicValue) -> Option<&'a DynamicValue> {
        let mut current = val;
        for part in &self.parts {
            match current {
                DynamicValue::Map(m) => {
                    current = m.get(part)?;
                }
                DynamicValue::List(l) => {
                    let idx: usize = part.parse().ok()?;
                    current = l.get(idx)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Traverse the value tree and set the target, returning `true` on success.
    pub fn set(&self, val: &mut DynamicValue, new_val: DynamicValue) -> bool {
        if self.parts.is_empty() {
            return false;
        }
        if self.parts.len() == 1 {
            return set_direct(val, &self.parts[0], new_val);
        }
        // Navigate to parent
        let mut current = val;
        for part in &self.parts[..self.parts.len() - 1] {
            match current {
                DynamicValue::Map(m) => {
                    current = match m.get_mut(part) {
                        Some(v) => v,
                        None => return false,
                    };
                }
                DynamicValue::List(l) => {
                    let idx: usize = match part.parse() {
                        Ok(i) => i,
                        Err(_) => return false,
                    };
                    current = match l.get_mut(idx) {
                        Some(v) => v,
                        None => return false,
                    };
                }
                _ => return false,
            }
        }
        let last = &self.parts[self.parts.len() - 1];
        set_direct(current, last, new_val)
    }
}

impl fmt::Display for ValuePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

fn set_direct(target: &mut DynamicValue, key: &str, new_val: DynamicValue) -> bool {
    match target {
        DynamicValue::Map(m) => {
            m.insert(key.to_owned(), new_val);
            true
        }
        DynamicValue::List(l) => {
            if let Ok(idx) = key.parse::<usize>() {
                if idx < l.len() {
                    l[idx] = new_val;
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// TypeRegistry
// ---------------------------------------------------------------------------

/// A registry of expected types for named fields, enabling batch validation.
#[derive(Debug, Clone)]
pub struct TypeRegistry {
    schemas: HashMap<String, ValueSchema>,
}

impl TypeRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    /// Register a schema for a field name.
    pub fn register(&mut self, field: &str, schema: ValueSchema) {
        self.schemas.insert(field.to_owned(), schema);
    }

    /// Validate all fields against their registered schemas.
    ///
    /// Returns field names paired with their validation errors.
    pub fn validate_all(
        &self,
        values: &HashMap<String, DynamicValue>,
    ) -> Result<(), Vec<(String, Vec<String>)>> {
        let mut all_errors = Vec::new();

        for (field, schema) in &self.schemas {
            let val = values.get(field).cloned().unwrap_or(DynamicValue::Null);
            if let Err(errs) = schema.validate(&val) {
                all_errors.push((field.clone(), errs));
            }
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            all_errors.sort_by(|a, b| a.0.cmp(&b.0));
            Err(all_errors)
        }
    }

    /// Returns the list of registered field names.
    pub fn registered_fields(&self) -> Vec<String> {
        let mut fields: Vec<String> = self.schemas.keys().cloned().collect();
        fields.sort();
        fields
    }

    /// Returns the number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn value_length(val: &DynamicValue) -> Option<usize> {
    match val {
        DynamicValue::String(s) => Some(s.len()),
        DynamicValue::List(l) => Some(l.len()),
        DynamicValue::Bytes(b) => Some(b.len()),
        _ => None,
    }
}

fn value_as_f64(val: &DynamicValue) -> Option<f64> {
    match val {
        DynamicValue::Integer(i) => Some(*i as f64),
        DynamicValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Simple base64 encoder (no padding variant) using only stdlib.
/// We use the `base64` crate available in the workspace.
fn base64_encode(data: &[u8]) -> String {
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- DynamicValue basics --

    #[test]
    fn test_type_name() {
        assert_eq!(DynamicValue::String("hi".into()).type_name(), "String");
        assert_eq!(DynamicValue::Integer(1).type_name(), "Integer");
        assert_eq!(DynamicValue::Float(1.0).type_name(), "Float");
        assert_eq!(DynamicValue::Bool(true).type_name(), "Bool");
        assert_eq!(DynamicValue::List(vec![]).type_name(), "List");
        assert_eq!(DynamicValue::Map(HashMap::new()).type_name(), "Map");
        assert_eq!(DynamicValue::Bytes(vec![]).type_name(), "Bytes");
        assert_eq!(DynamicValue::Null.type_name(), "Null");
    }

    #[test]
    fn test_is_null() {
        assert!(DynamicValue::Null.is_null());
        assert!(!DynamicValue::Integer(0).is_null());
    }

    #[test]
    fn test_is_truthy() {
        assert!(!DynamicValue::Null.is_truthy());
        assert!(!DynamicValue::Bool(false).is_truthy());
        assert!(DynamicValue::Bool(true).is_truthy());
        assert!(!DynamicValue::Integer(0).is_truthy());
        assert!(DynamicValue::Integer(1).is_truthy());
        assert!(!DynamicValue::Float(0.0).is_truthy());
        assert!(DynamicValue::Float(0.1).is_truthy());
        assert!(!DynamicValue::String("".into()).is_truthy());
        assert!(DynamicValue::String("x".into()).is_truthy());
        assert!(!DynamicValue::List(vec![]).is_truthy());
        assert!(DynamicValue::List(vec![DynamicValue::Null]).is_truthy());
        assert!(!DynamicValue::Map(HashMap::new()).is_truthy());
        assert!(!DynamicValue::Bytes(vec![]).is_truthy());
        assert!(DynamicValue::Bytes(vec![1]).is_truthy());
    }

    #[test]
    fn test_to_json_primitives() {
        assert_eq!(DynamicValue::String("hello".into()).to_json(), json!("hello"));
        assert_eq!(DynamicValue::Integer(42).to_json(), json!(42));
        assert_eq!(DynamicValue::Float(3.14).to_json(), json!(3.14));
        assert_eq!(DynamicValue::Bool(true).to_json(), json!(true));
        assert_eq!(DynamicValue::Null.to_json(), json!(null));
    }

    #[test]
    fn test_to_json_list() {
        let list = DynamicValue::List(vec![DynamicValue::Integer(1), DynamicValue::Integer(2)]);
        assert_eq!(list.to_json(), json!([1, 2]));
    }

    #[test]
    fn test_to_json_map() {
        let mut m = HashMap::new();
        m.insert("key".to_owned(), DynamicValue::String("val".into()));
        let map = DynamicValue::Map(m);
        assert_eq!(map.to_json(), json!({"key": "val"}));
    }

    #[test]
    fn test_to_json_bytes() {
        let val = DynamicValue::Bytes(vec![72, 101, 108, 108, 111]); // "Hello"
        let j = val.to_json();
        assert!(j.is_string());
    }

    #[test]
    fn test_from_json() {
        assert_eq!(DynamicValue::from_json(&json!(null)), DynamicValue::Null);
        assert_eq!(DynamicValue::from_json(&json!(true)), DynamicValue::Bool(true));
        assert_eq!(DynamicValue::from_json(&json!(42)), DynamicValue::Integer(42));
        assert_eq!(DynamicValue::from_json(&json!(3.14)), DynamicValue::Float(3.14));
        assert_eq!(DynamicValue::from_json(&json!("hi")), DynamicValue::String("hi".into()));
    }

    #[test]
    fn test_from_json_array() {
        let v = DynamicValue::from_json(&json!([1, "two"]));
        match v {
            DynamicValue::List(l) => {
                assert_eq!(l.len(), 2);
                assert_eq!(l[0], DynamicValue::Integer(1));
                assert_eq!(l[1], DynamicValue::String("two".into()));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_from_json_object() {
        let v = DynamicValue::from_json(&json!({"a": 1}));
        match v {
            DynamicValue::Map(m) => {
                assert_eq!(m.get("a"), Some(&DynamicValue::Integer(1)));
            }
            _ => panic!("expected map"),
        }
    }

    #[test]
    fn test_roundtrip_json() {
        let original = json!({"name": "test", "values": [1, 2, 3], "active": true});
        let dv = DynamicValue::from_json(&original);
        let back = dv.to_json();
        assert_eq!(original, back);
    }

    // -- typed accessors --

    #[test]
    fn test_as_str() {
        assert_eq!(DynamicValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(DynamicValue::Integer(1).as_str(), None);
    }

    #[test]
    fn test_as_i64() {
        assert_eq!(DynamicValue::Integer(42).as_i64(), Some(42));
        assert_eq!(DynamicValue::String("x".into()).as_i64(), None);
    }

    #[test]
    fn test_as_f64() {
        assert_eq!(DynamicValue::Float(1.5).as_f64(), Some(1.5));
        assert_eq!(DynamicValue::Null.as_f64(), None);
    }

    #[test]
    fn test_as_bool() {
        assert_eq!(DynamicValue::Bool(true).as_bool(), Some(true));
        assert_eq!(DynamicValue::Integer(1).as_bool(), None);
    }

    #[test]
    fn test_as_list() {
        let list = DynamicValue::List(vec![DynamicValue::Null]);
        assert!(list.as_list().is_some());
        assert!(DynamicValue::Null.as_list().is_none());
    }

    #[test]
    fn test_as_map() {
        let map = DynamicValue::Map(HashMap::new());
        assert!(map.as_map().is_some());
        assert!(DynamicValue::Null.as_map().is_none());
    }

    // -- From impls --

    #[test]
    fn test_from_string() {
        let v: DynamicValue = "hello".to_string().into();
        assert_eq!(v, DynamicValue::String("hello".into()));
    }

    #[test]
    fn test_from_str_ref() {
        let v: DynamicValue = "world".into();
        assert_eq!(v, DynamicValue::String("world".into()));
    }

    #[test]
    fn test_from_i64() {
        let v: DynamicValue = 99i64.into();
        assert_eq!(v, DynamicValue::Integer(99));
    }

    #[test]
    fn test_from_f64() {
        let v: DynamicValue = 2.718f64.into();
        assert_eq!(v, DynamicValue::Float(2.718));
    }

    #[test]
    fn test_from_bool() {
        let v: DynamicValue = true.into();
        assert_eq!(v, DynamicValue::Bool(true));
    }

    #[test]
    fn test_from_vec() {
        let v: DynamicValue = vec![DynamicValue::Integer(1)].into();
        assert_eq!(v.as_list().unwrap().len(), 1);
    }

    #[test]
    fn test_from_value() {
        let v: DynamicValue = json!(42).into();
        assert_eq!(v, DynamicValue::Integer(42));
    }

    // -- TypeCoercer --

    #[test]
    fn test_coerce_to_string_from_int() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_string(&DynamicValue::Integer(42)).unwrap(), "42");
    }

    #[test]
    fn test_coerce_to_string_from_bool() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_string(&DynamicValue::Bool(true)).unwrap(), "true");
    }

    #[test]
    fn test_coerce_to_string_from_null() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_string(&DynamicValue::Null).unwrap(), "null");
    }

    #[test]
    fn test_coerce_to_string_from_list_lenient() {
        let c = TypeCoercer::lenient();
        let list = DynamicValue::List(vec![DynamicValue::Integer(1)]);
        let result = c.to_string(&list).unwrap();
        assert!(result.contains('1'));
    }

    #[test]
    fn test_coerce_to_string_from_list_strict() {
        let c = TypeCoercer::strict();
        let list = DynamicValue::List(vec![]);
        assert!(c.to_string(&list).is_err());
    }

    #[test]
    fn test_coerce_to_integer_from_float() {
        let c = TypeCoercer::lenient();
        assert_eq!(c.to_integer(&DynamicValue::Float(3.7)).unwrap(), 3);
    }

    #[test]
    fn test_coerce_to_integer_from_float_strict() {
        let c = TypeCoercer::strict();
        // 3.0 can be losslessly converted
        assert_eq!(c.to_integer(&DynamicValue::Float(3.0)).unwrap(), 3);
        // 3.7 cannot
        assert!(c.to_integer(&DynamicValue::Float(3.7)).is_err());
    }

    #[test]
    fn test_coerce_to_integer_from_string() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_integer(&DynamicValue::String("123".into())).unwrap(), 123);
    }

    #[test]
    fn test_coerce_to_integer_from_bad_string() {
        let c = TypeCoercer::new();
        assert!(c.to_integer(&DynamicValue::String("abc".into())).is_err());
    }

    #[test]
    fn test_coerce_to_integer_from_bool() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_integer(&DynamicValue::Bool(true)).unwrap(), 1);
        assert_eq!(c.to_integer(&DynamicValue::Bool(false)).unwrap(), 0);
    }

    #[test]
    fn test_coerce_to_float_from_int() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_float(&DynamicValue::Integer(5)).unwrap(), 5.0);
    }

    #[test]
    fn test_coerce_to_float_from_string() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_float(&DynamicValue::String("3.14".into())).unwrap(), 3.14);
    }

    #[test]
    fn test_coerce_to_float_incompatible() {
        let c = TypeCoercer::new();
        assert!(c.to_float(&DynamicValue::List(vec![])).is_err());
    }

    #[test]
    fn test_coerce_to_bool_from_string() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_bool(&DynamicValue::String("yes".into())).unwrap(), true);
        assert_eq!(c.to_bool(&DynamicValue::String("false".into())).unwrap(), false);
    }

    #[test]
    fn test_coerce_to_bool_from_null() {
        let c = TypeCoercer::new();
        assert_eq!(c.to_bool(&DynamicValue::Null).unwrap(), false);
    }

    #[test]
    fn test_coerce_to_bool_parse_error() {
        let c = TypeCoercer::new();
        assert!(c.to_bool(&DynamicValue::String("maybe".into())).is_err());
    }

    #[test]
    fn test_coerce_method_any() {
        let c = TypeCoercer::new();
        let val = DynamicValue::Integer(10);
        let result = c.coerce(&val, ValueType::Any).unwrap();
        assert_eq!(result, val);
    }

    #[test]
    fn test_coerce_method_to_string() {
        let c = TypeCoercer::new();
        let result = c.coerce(&DynamicValue::Integer(7), ValueType::String).unwrap();
        assert_eq!(result, DynamicValue::String("7".into()));
    }

    #[test]
    fn test_coerce_to_list_incompatible() {
        let c = TypeCoercer::new();
        assert!(c.coerce(&DynamicValue::Integer(1), ValueType::List).is_err());
    }

    #[test]
    fn test_coerce_to_map_ok() {
        let c = TypeCoercer::new();
        let map = DynamicValue::Map(HashMap::new());
        assert!(c.coerce(&map, ValueType::Map).is_ok());
    }

    // -- ValueType --

    #[test]
    fn test_value_type_matches() {
        assert!(ValueType::String.matches(&DynamicValue::String("x".into())));
        assert!(!ValueType::String.matches(&DynamicValue::Integer(1)));
        assert!(ValueType::Any.matches(&DynamicValue::Null));
        assert!(ValueType::Null.matches(&DynamicValue::Null));
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", ValueType::String), "String");
        assert_eq!(format!("{}", ValueType::Any), "Any");
    }

    // -- CoercionError --

    #[test]
    fn test_coercion_error_display() {
        let e = CoercionError::IncompatibleTypes {
            from: "List".into(),
            to: "Integer".into(),
        };
        assert!(format!("{}", e).contains("List"));
        assert!(format!("{}", e).contains("Integer"));

        let e2 = CoercionError::ParseError("bad".into());
        assert!(format!("{}", e2).contains("bad"));
    }

    #[test]
    fn test_coercion_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(CoercionError::ParseError("x".into()));
        assert!(format!("{}", e).contains("x"));
    }

    // -- ValueConstraint --

    #[test]
    fn test_constraint_min_length() {
        let c = ValueConstraint::MinLength(3);
        assert!(c.check(&DynamicValue::String("abc".into())));
        assert!(!c.check(&DynamicValue::String("ab".into())));
        // Non-applicable type returns true
        assert!(c.check(&DynamicValue::Integer(1)));
    }

    #[test]
    fn test_constraint_max_length() {
        let c = ValueConstraint::MaxLength(2);
        assert!(c.check(&DynamicValue::String("ab".into())));
        assert!(!c.check(&DynamicValue::String("abc".into())));
    }

    #[test]
    fn test_constraint_min_value() {
        let c = ValueConstraint::MinValue(10.0);
        assert!(c.check(&DynamicValue::Integer(10)));
        assert!(!c.check(&DynamicValue::Integer(9)));
        assert!(c.check(&DynamicValue::Float(10.5)));
    }

    #[test]
    fn test_constraint_max_value() {
        let c = ValueConstraint::MaxValue(100.0);
        assert!(c.check(&DynamicValue::Integer(100)));
        assert!(!c.check(&DynamicValue::Integer(101)));
    }

    #[test]
    fn test_constraint_pattern() {
        let c = ValueConstraint::Pattern(r"^\d+$".into());
        assert!(c.check(&DynamicValue::String("123".into())));
        assert!(!c.check(&DynamicValue::String("abc".into())));
        // Non-string: vacuously true
        assert!(c.check(&DynamicValue::Integer(1)));
    }

    #[test]
    fn test_constraint_one_of() {
        let c = ValueConstraint::OneOf(vec![
            DynamicValue::String("a".into()),
            DynamicValue::String("b".into()),
        ]);
        assert!(c.check(&DynamicValue::String("a".into())));
        assert!(!c.check(&DynamicValue::String("c".into())));
    }

    #[test]
    fn test_constraint_display() {
        assert_eq!(format!("{}", ValueConstraint::MinLength(5)), "min_length(5)");
        assert_eq!(format!("{}", ValueConstraint::MaxValue(10.0)), "max_value(10)");
    }

    // -- ValueSchema --

    #[test]
    fn test_schema_validate_ok() {
        let schema = ValueSchema::new(ValueType::String);
        assert!(schema.validate(&DynamicValue::String("hi".into())).is_ok());
    }

    #[test]
    fn test_schema_validate_wrong_type() {
        let schema = ValueSchema::new(ValueType::String);
        assert!(schema.validate(&DynamicValue::Integer(1)).is_err());
    }

    #[test]
    fn test_schema_validate_required_null() {
        let schema = ValueSchema::new(ValueType::String).required();
        let result = schema.validate(&DynamicValue::Null);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_validate_optional_null() {
        let schema = ValueSchema::new(ValueType::String)
            .optional(DynamicValue::String("default".into()));
        assert!(schema.validate(&DynamicValue::Null).is_ok());
    }

    #[test]
    fn test_schema_with_constraint() {
        let schema = ValueSchema::new(ValueType::String)
            .with_constraint(ValueConstraint::MinLength(3));
        assert!(schema.validate(&DynamicValue::String("abc".into())).is_ok());
        assert!(schema.validate(&DynamicValue::String("ab".into())).is_err());
    }

    // -- ValuePath --

    #[test]
    fn test_path_segments() {
        let p = ValuePath::new("a.b.c");
        assert_eq!(p.segments(), vec!["a", "b", "c"]);
        assert_eq!(p.depth(), 3);
    }

    #[test]
    fn test_path_display() {
        let p = ValuePath::new("x.y");
        assert_eq!(format!("{}", p), "x.y");
    }

    #[test]
    fn test_path_get_simple() {
        let mut m = HashMap::new();
        m.insert("name".to_owned(), DynamicValue::String("Alice".into()));
        let val = DynamicValue::Map(m);
        let p = ValuePath::new("name");
        assert_eq!(p.get(&val), Some(&DynamicValue::String("Alice".into())));
    }

    #[test]
    fn test_path_get_nested() {
        let inner = {
            let mut m = HashMap::new();
            m.insert("x".to_owned(), DynamicValue::Integer(42));
            DynamicValue::Map(m)
        };
        let outer = {
            let mut m = HashMap::new();
            m.insert("inner".to_owned(), inner);
            DynamicValue::Map(m)
        };
        let p = ValuePath::new("inner.x");
        assert_eq!(p.get(&outer), Some(&DynamicValue::Integer(42)));
    }

    #[test]
    fn test_path_get_list_index() {
        let list = DynamicValue::List(vec![
            DynamicValue::String("a".into()),
            DynamicValue::String("b".into()),
        ]);
        let mut m = HashMap::new();
        m.insert("items".to_owned(), list);
        let val = DynamicValue::Map(m);
        let p = ValuePath::new("items.1");
        assert_eq!(p.get(&val), Some(&DynamicValue::String("b".into())));
    }

    #[test]
    fn test_path_get_missing() {
        let val = DynamicValue::Map(HashMap::new());
        let p = ValuePath::new("nonexistent");
        assert_eq!(p.get(&val), None);
    }

    #[test]
    fn test_path_set_simple() {
        let mut m = HashMap::new();
        m.insert("name".to_owned(), DynamicValue::String("old".into()));
        let mut val = DynamicValue::Map(m);
        let p = ValuePath::new("name");
        assert!(p.set(&mut val, DynamicValue::String("new".into())));
        assert_eq!(
            p.get(&val),
            Some(&DynamicValue::String("new".into()))
        );
    }

    #[test]
    fn test_path_set_nested() {
        let inner = {
            let mut m = HashMap::new();
            m.insert("x".to_owned(), DynamicValue::Integer(0));
            DynamicValue::Map(m)
        };
        let mut outer = {
            let mut m = HashMap::new();
            m.insert("inner".to_owned(), inner);
            DynamicValue::Map(m)
        };
        let p = ValuePath::new("inner.x");
        assert!(p.set(&mut outer, DynamicValue::Integer(99)));
        assert_eq!(p.get(&outer), Some(&DynamicValue::Integer(99)));
    }

    #[test]
    fn test_path_set_list_index() {
        let mut list = DynamicValue::List(vec![DynamicValue::Integer(1), DynamicValue::Integer(2)]);
        let p = ValuePath::new("0");
        assert!(p.set(&mut list, DynamicValue::Integer(10)));
        assert_eq!(
            list,
            DynamicValue::List(vec![DynamicValue::Integer(10), DynamicValue::Integer(2)])
        );
    }

    #[test]
    fn test_path_set_returns_false_on_missing_parent() {
        let mut val = DynamicValue::Map(HashMap::new());
        let p = ValuePath::new("a.b.c");
        assert!(!p.set(&mut val, DynamicValue::Integer(1)));
    }

    // -- TypeRegistry --

    #[test]
    fn test_registry_new() {
        let reg = TypeRegistry::new();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_register_and_len() {
        let mut reg = TypeRegistry::new();
        reg.register("name", ValueSchema::new(ValueType::String));
        reg.register("age", ValueSchema::new(ValueType::Integer));
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_registered_fields() {
        let mut reg = TypeRegistry::new();
        reg.register("b_field", ValueSchema::new(ValueType::String));
        reg.register("a_field", ValueSchema::new(ValueType::Integer));
        assert_eq!(reg.registered_fields(), vec!["a_field", "b_field"]);
    }

    #[test]
    fn test_registry_validate_all_ok() {
        let mut reg = TypeRegistry::new();
        reg.register("name", ValueSchema::new(ValueType::String));
        reg.register("age", ValueSchema::new(ValueType::Integer));

        let mut values = HashMap::new();
        values.insert("name".to_owned(), DynamicValue::String("Alice".into()));
        values.insert("age".to_owned(), DynamicValue::Integer(30));
        assert!(reg.validate_all(&values).is_ok());
    }

    #[test]
    fn test_registry_validate_all_errors() {
        let mut reg = TypeRegistry::new();
        reg.register("name", ValueSchema::new(ValueType::String));
        reg.register("age", ValueSchema::new(ValueType::Integer));

        let mut values = HashMap::new();
        values.insert("name".to_owned(), DynamicValue::Integer(123));
        // age missing -> Null -> required fails

        let result = reg.validate_all(&values);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_registry_validate_optional_field_missing() {
        let mut reg = TypeRegistry::new();
        reg.register(
            "nickname",
            ValueSchema::new(ValueType::String).optional(DynamicValue::String("none".into())),
        );
        let values = HashMap::new();
        assert!(reg.validate_all(&values).is_ok());
    }

    #[test]
    fn test_coercer_default() {
        let c = TypeCoercer::default();
        // default is lenient
        let list = DynamicValue::List(vec![]);
        assert!(c.to_string(&list).is_ok());
    }
}
