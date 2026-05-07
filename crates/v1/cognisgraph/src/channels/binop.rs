//! BinaryOperatorAggregate channel — applies a binary operator to aggregate values.
//!
//! This channel maintains an accumulated value that is updated by folding incoming
//! values using a configurable binary operator. Supports common operations like
//! addition (for numbers), array extension, and replacement, as well as custom
//! operator functions.

use std::sync::Arc;

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// Binary operator used to combine channel values.
///
/// When the channel receives an update, it folds each incoming value into the
/// current accumulated value using this operator.
#[derive(Clone)]
pub enum BinOp {
    /// Add numeric values. For non-numeric values, falls back to string concatenation.
    Add,
    /// Extend arrays. If both values are arrays, the right array is appended.
    /// If the left is an array, the right value is pushed. Otherwise, creates
    /// a new array with both values.
    Extend,
    /// Replace the current value entirely with the new value.
    Replace,
    /// A custom binary operator function.
    Custom(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
}

impl std::fmt::Debug for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "BinOp::Add"),
            BinOp::Extend => write!(f, "BinOp::Extend"),
            BinOp::Replace => write!(f, "BinOp::Replace"),
            BinOp::Custom(_) => write!(f, "BinOp::Custom(...)"),
        }
    }
}

impl BinOp {
    /// Apply the binary operator to two values.
    pub fn apply(&self, left: Value, right: Value) -> Value {
        match self {
            BinOp::Add => binop_add(left, right),
            BinOp::Extend => binop_extend(left, right),
            BinOp::Replace => right,
            BinOp::Custom(f) => f(left, right),
        }
    }
}

/// Add two values. Numbers are added numerically; strings are concatenated.
fn binop_add(left: Value, right: Value) -> Value {
    match (&left, &right) {
        (Value::Number(a), Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                Value::from(ai + bi)
            } else if let (Some(af), Some(bf)) = (a.as_f64(), b.as_f64()) {
                Value::from(af + bf)
            } else {
                right
            }
        }
        (Value::String(a), Value::String(b)) => Value::from(format!("{}{}", a, b)),
        (Value::Array(a), Value::Array(b)) => {
            let mut result = a.clone();
            result.extend(b.iter().cloned());
            Value::Array(result)
        }
        _ => right,
    }
}

/// Extend arrays. If left is an array, right elements are appended.
fn binop_extend(left: Value, right: Value) -> Value {
    match (left, right) {
        (Value::Array(mut a), Value::Array(b)) => {
            a.extend(b);
            Value::Array(a)
        }
        (Value::Array(mut a), other) => {
            a.push(other);
            Value::Array(a)
        }
        (other, Value::Array(b)) => {
            let mut result = vec![other];
            result.extend(b);
            Value::Array(result)
        }
        (a, b) => Value::Array(vec![a, b]),
    }
}

/// Check if a value is an [`Overwrite`] wrapper. If so, return the inner value.
fn unwrap_overwrite(value: &Value) -> Option<Value> {
    if let Value::Object(map) = value {
        if map.len() == 1 {
            if let Some(inner) = map.get("value") {
                // Check if this looks like a serialized Overwrite struct.
                // We use a simple heuristic: an object with a single "value" key
                // that was tagged as an Overwrite.
                return Some(inner.clone());
            }
        }
    }
    None
}

/// A channel that aggregates values using a binary operator.
///
/// Each update folds incoming values into the accumulated result using the
/// configured [`BinOp`]. If an incoming value is an [`Overwrite`] wrapper,
/// the accumulated value is replaced rather than folded.
#[derive(Debug, Clone)]
pub struct BinaryOperatorAggregate {
    /// The key name of this channel.
    key: String,
    /// The current accumulated value.
    value: Option<Value>,
    /// The binary operator to apply.
    operator: BinOp,
}

impl BinaryOperatorAggregate {
    /// Create a new `BinaryOperatorAggregate` channel.
    pub fn new(key: impl Into<String>, operator: BinOp) -> Self {
        Self {
            key: key.into(),
            value: None,
            operator,
        }
    }

    /// Create a new channel with an initial value.
    pub fn with_value(key: impl Into<String>, operator: BinOp, value: Value) -> Self {
        Self {
            key: key.into(),
            value: Some(value),
            operator,
        }
    }

    /// Apply the overwrite protocol: if the value is an Overwrite wrapper,
    /// replace the accumulator; otherwise, fold using the operator.
    fn apply_value(&mut self, new_value: Value) {
        // Check for Overwrite wrapper.
        if let Some(inner) = unwrap_overwrite(&new_value) {
            self.value = Some(inner);
            return;
        }

        match self.value.take() {
            Some(current) => {
                self.value = Some(self.operator.apply(current, new_value));
            }
            None => {
                self.value = Some(new_value);
            }
        }
    }
}

impl BaseChannel for BinaryOperatorAggregate {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        self.value.clone()
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        Box::new(BinaryOperatorAggregate {
            key: self.key.clone(),
            value: checkpoint,
            operator: self.operator.clone(),
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        self.value.clone().ok_or(LangGraphError::EmptyChannelError)
    }

    fn is_available(&self) -> bool {
        self.value.is_some()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }

        let mut seen_overwrite = false;
        for v in values {
            if unwrap_overwrite(&v).is_some() {
                if seen_overwrite {
                    return Err(LangGraphError::InvalidUpdateError(
                        "Can receive only one Overwrite value per super-step.".to_string(),
                    ));
                }
                seen_overwrite = true;
            }
            self.apply_value(v);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_numbers() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(0));
        ch.update(vec![Value::from(5)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(5));

        ch.update(vec![Value::from(3)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(8));
    }

    #[test]
    fn test_add_floats() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(1.5));
        ch.update(vec![Value::from(2.5)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(4.0));
    }

    #[test]
    fn test_add_strings() {
        let mut ch = BinaryOperatorAggregate::with_value("text", BinOp::Add, Value::from("hello"));
        ch.update(vec![Value::from(" world")]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from("hello world"));
    }

    #[test]
    fn test_add_arrays() {
        let mut ch = BinaryOperatorAggregate::with_value(
            "list",
            BinOp::Add,
            Value::Array(vec![Value::from(1)]),
        );
        ch.update(vec![Value::Array(vec![Value::from(2), Value::from(3)])])
            .unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from(1), Value::from(2), Value::from(3)])
        );
    }

    #[test]
    fn test_extend_arrays() {
        let mut ch = BinaryOperatorAggregate::with_value(
            "list",
            BinOp::Extend,
            Value::Array(vec![Value::from("a")]),
        );
        ch.update(vec![Value::Array(vec![Value::from("b")])])
            .unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("a"), Value::from("b")])
        );
    }

    #[test]
    fn test_extend_non_array_into_array() {
        let mut ch = BinaryOperatorAggregate::with_value(
            "list",
            BinOp::Extend,
            Value::Array(vec![Value::from(1)]),
        );
        ch.update(vec![Value::from(2)]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from(1), Value::from(2)])
        );
    }

    #[test]
    fn test_replace() {
        let mut ch = BinaryOperatorAggregate::with_value("val", BinOp::Replace, Value::from("old"));
        ch.update(vec![Value::from("new")]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from("new"));
    }

    #[test]
    fn test_custom_operator() {
        let max_op = BinOp::Custom(Arc::new(|a, b| {
            let a_num = a.as_i64().unwrap_or(0);
            let b_num = b.as_i64().unwrap_or(0);
            Value::from(a_num.max(b_num))
        }));
        let mut ch = BinaryOperatorAggregate::with_value("max", max_op, Value::from(5));
        ch.update(vec![Value::from(3)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(5));

        ch.update(vec![Value::from(10)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(10));
    }

    #[test]
    fn test_overwrite_wrapper() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(100));

        // Send an Overwrite-style value.
        let overwrite = serde_json::json!({"value": 0});
        ch.update(vec![overwrite]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(0));
    }

    #[test]
    fn test_multiple_values_in_single_update() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(0));
        ch.update(vec![Value::from(1), Value::from(2), Value::from(3)])
            .unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(6));
    }

    #[test]
    fn test_empty_update() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(5));
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
        assert_eq!(ch.get().unwrap(), Value::from(5));
    }

    #[test]
    fn test_update_on_empty_channel() {
        let mut ch = BinaryOperatorAggregate::new("counter", BinOp::Add);
        assert!(!ch.is_available());

        ch.update(vec![Value::from(42)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(42));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = BinaryOperatorAggregate::new("counter", BinOp::Add);
        ch.update(vec![Value::from(10)]).unwrap();

        let ckpt = ch.checkpoint();
        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(restored.get().unwrap(), Value::from(10));
    }

    #[test]
    fn test_multiple_overwrites_error() {
        let mut ch = BinaryOperatorAggregate::with_value("counter", BinOp::Add, Value::from(100));
        let ow1 = serde_json::json!({"value": 0});
        let ow2 = serde_json::json!({"value": 50});
        let result = ch.update(vec![ow1, ow2]);
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::InvalidUpdateError(msg) => {
                assert!(msg.contains("Overwrite"));
            }
            other => panic!("Expected InvalidUpdateError, got: {:?}", other),
        }
    }
}
