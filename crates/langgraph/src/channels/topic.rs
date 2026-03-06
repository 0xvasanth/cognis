//! Topic channel — pub/sub style channel that collects values into a list.
//!
//! The topic channel stores a list of values. If `accumulate` is `false` (default),
//! the list is cleared at the start of each step before new values are added.
//! If `accumulate` is `true`, values persist across steps.

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A pub/sub channel that collects values into a list.
///
/// When `accumulate` is `false`, the values list is cleared at each step via
/// [`consume`](BaseChannel::consume) before new values are added. When `true`,
/// values persist and accumulate across steps.
#[derive(Debug, Clone)]
pub struct Topic {
    /// The key name of this channel.
    key: String,
    /// The collected values.
    values: Vec<Value>,
    /// Whether to accumulate values across steps.
    accumulate: bool,
}

impl Topic {
    /// Create a new `Topic` channel with the given key.
    ///
    /// By default, `accumulate` is `false`, meaning the channel clears
    /// at each step.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            values: Vec::new(),
            accumulate: false,
        }
    }

    /// Create a new accumulating `Topic` channel.
    pub fn accumulating(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            values: Vec::new(),
            accumulate: true,
        }
    }

    /// Set whether the channel should accumulate values.
    pub fn with_accumulate(mut self, accumulate: bool) -> Self {
        self.accumulate = accumulate;
        self
    }
}

impl BaseChannel for Topic {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        if self.values.is_empty() {
            None
        } else {
            Some(Value::Array(self.values.clone()))
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        let values = match checkpoint {
            Some(Value::Array(arr)) => arr,
            Some(other) => vec![other],
            None => Vec::new(),
        };
        Box::new(Topic {
            key: self.key.clone(),
            values,
            accumulate: self.accumulate,
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        if self.values.is_empty() {
            Err(LangGraphError::EmptyChannelError)
        } else {
            Ok(Value::Array(self.values.clone()))
        }
    }

    fn is_available(&self) -> bool {
        !self.values.is_empty()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }

        if !self.accumulate {
            self.values.clear();
        }

        // Flatten any arrays in the input values.
        for v in values {
            match v {
                Value::Array(arr) => self.values.extend(arr),
                other => self.values.push(other),
            }
        }
        Ok(true)
    }

    fn consume(&mut self) -> bool {
        if !self.accumulate && !self.values.is_empty() {
            self.values.clear();
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_channel_is_empty() {
        let ch = Topic::new("events");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_update_single_value() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("event1")]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("event1")])
        );
    }

    #[test]
    fn test_update_multiple_values() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("a"), Value::from("b")])
        );
    }

    #[test]
    fn test_flatten_arrays_in_input() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::Array(vec![
            Value::from(1),
            Value::from(2),
        ])])
        .unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from(1), Value::from(2)])
        );
    }

    #[test]
    fn test_non_accumulate_clears_on_update() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("first")]).unwrap();
        ch.update(vec![Value::from("second")]).unwrap();
        // Non-accumulate clears before adding new values.
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("second")])
        );
    }

    #[test]
    fn test_accumulate_mode() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("first")]).unwrap();
        ch.update(vec![Value::from("second")]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("first"), Value::from("second")])
        );
    }

    #[test]
    fn test_consume_clears_non_accumulate() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(!ch.values.is_empty());

        let changed = ch.consume();
        assert!(changed);
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_consume_noop_on_accumulate() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("data")]).unwrap();

        let changed = ch.consume();
        assert!(!changed);
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("data")])
        );
    }

    #[test]
    fn test_consume_noop_on_empty() {
        let mut ch = Topic::new("events");
        let changed = ch.consume();
        assert!(!changed);
    }

    #[test]
    fn test_empty_update() {
        let mut ch = Topic::new("events");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();

        let ckpt = ch.checkpoint();
        assert!(ckpt.is_some());

        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(
            restored.get().unwrap(),
            Value::Array(vec![Value::from("a"), Value::from("b")])
        );
    }

    #[test]
    fn test_checkpoint_empty() {
        let ch = Topic::new("events");
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_from_checkpoint_none() {
        let ch = Topic::new("events");
        let restored = ch.from_checkpoint(None);
        assert!(!restored.is_available());
        assert!(restored.get().is_err());
    }

    #[test]
    fn test_with_accumulate_builder() {
        let ch = Topic::new("events").with_accumulate(true);
        assert!(ch.accumulate);
    }
}
