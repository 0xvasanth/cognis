//! Channel reading for the Pregel execution engine.
//!
//! Provides [`PregelChannelReader`] which reads the current state from channels
//! and presents it as a `serde_json::Value` for consumption by node actions.

use std::collections::HashMap;

use serde_json::Value;

use crate::channels::base::BaseChannel;
use crate::errors::{LangGraphError, Result};

/// Reads state from a set of channels and returns it as a JSON value.
///
/// `PregelChannelReader` is a stateless helper that maps channel keys to their
/// current values. It supports reading a single channel (returning its raw value)
/// or multiple channels (returning a JSON object keyed by channel name).
#[derive(Debug)]
pub struct PregelChannelReader {
    /// The channel names to read from.
    pub channel_names: Vec<String>,
    /// Whether to skip channels that are empty rather than returning an error.
    pub skip_empty: bool,
}

impl PregelChannelReader {
    /// Create a new reader for the specified channel names.
    pub fn new(channel_names: Vec<String>) -> Self {
        Self {
            channel_names,
            skip_empty: true,
        }
    }

    /// Create a new reader for a single channel.
    pub fn single(channel_name: impl Into<String>) -> Self {
        Self {
            channel_names: vec![channel_name.into()],
            skip_empty: true,
        }
    }

    /// Set whether to skip empty channels.
    pub fn with_skip_empty(mut self, skip_empty: bool) -> Self {
        self.skip_empty = skip_empty;
        self
    }

    /// Read the current state from the given channels.
    ///
    /// If the reader was configured with a single channel name, returns that
    /// channel's raw value (or `Value::Null` if empty and `skip_empty` is true).
    ///
    /// If configured with multiple channel names, returns a JSON object keyed
    /// by channel name.
    ///
    /// # Arguments
    ///
    /// * `channels` - The map of channel name to channel instance.
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::EmptyChannelError`] if a required channel is empty
    /// and `skip_empty` is false.
    pub fn read(
        &self,
        channels: &HashMap<String, Box<dyn BaseChannel>>,
    ) -> Result<Value> {
        if self.channel_names.len() == 1 {
            return self.read_single(channels, &self.channel_names[0]);
        }

        let mut result = serde_json::Map::new();
        for name in &self.channel_names {
            match self.read_single(channels, name) {
                Ok(value) if value.is_null() && self.skip_empty => {
                    // Skip null/empty values when skip_empty is true.
                }
                Ok(value) => {
                    result.insert(name.clone(), value);
                }
                Err(LangGraphError::EmptyChannelError) if self.skip_empty => {
                    // Skip empty channels.
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Value::Object(result))
    }

    /// Read the current state, applying an optional mapper function to transform
    /// the result.
    pub fn read_with_mapper(
        &self,
        channels: &HashMap<String, Box<dyn BaseChannel>>,
        mapper: Option<&dyn Fn(Value) -> Value>,
    ) -> Result<Value> {
        let value = self.read(channels)?;
        match mapper {
            Some(f) => Ok(f(value)),
            None => Ok(value),
        }
    }

    /// Read a "fresh" view of the channels — identical to [`read`](Self::read)
    /// but conceptually signals that the caller wants the latest values rather
    /// than a cached view.
    pub fn read_fresh(
        &self,
        channels: &HashMap<String, Box<dyn BaseChannel>>,
    ) -> Result<Value> {
        // In the Rust implementation, channels are always read directly, so
        // "fresh" is the same as a normal read. In Python, this distinction
        // matters because of caching in the config.
        self.read(channels)
    }

    /// Read a single channel by name.
    fn read_single(
        &self,
        channels: &HashMap<String, Box<dyn BaseChannel>>,
        name: &str,
    ) -> Result<Value> {
        match channels.get(name) {
            Some(channel) => match channel.get() {
                Ok(value) => Ok(value),
                Err(LangGraphError::EmptyChannelError) if self.skip_empty => Ok(Value::Null),
                Err(e) => Err(e),
            },
            None if self.skip_empty => Ok(Value::Null),
            None => Err(LangGraphError::Other(format!(
                "Channel '{}' not found",
                name
            ))),
        }
    }
}

/// Convenience function to read a single channel value.
pub fn read_channel_value(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    channel_name: &str,
) -> Result<Value> {
    PregelChannelReader::single(channel_name).read(channels)
}

/// Convenience function to read multiple channel values into a JSON object.
pub fn read_channel_values(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    channel_names: &[String],
) -> Result<Value> {
    PregelChannelReader::new(channel_names.to_vec()).read(channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A simple test channel.
    #[derive(Debug, Clone)]
    struct TestChannel {
        key: String,
        value: Option<Value>,
    }

    impl BaseChannel for TestChannel {
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
            Box::new(TestChannel {
                key: self.key.clone(),
                value: checkpoint,
            })
        }

        fn get(&self) -> std::result::Result<Value, LangGraphError> {
            self.value
                .clone()
                .ok_or(LangGraphError::EmptyChannelError)
        }

        fn is_available(&self) -> bool {
            self.value.is_some()
        }

        fn update(&mut self, values: Vec<Value>) -> std::result::Result<bool, LangGraphError> {
            if let Some(v) = values.into_iter().last() {
                self.value = Some(v);
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    fn make_channels(
        entries: Vec<(&str, Option<Value>)>,
    ) -> HashMap<String, Box<dyn BaseChannel>> {
        entries
            .into_iter()
            .map(|(name, value)| {
                let ch: Box<dyn BaseChannel> = Box::new(TestChannel {
                    key: name.to_string(),
                    value,
                });
                (name.to_string(), ch)
            })
            .collect()
    }

    #[test]
    fn test_read_single_channel() {
        let channels = make_channels(vec![("foo", Some(json!(42)))]);
        let reader = PregelChannelReader::single("foo");
        let result = reader.read(&channels).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_read_single_channel_empty_skip() {
        let channels = make_channels(vec![("foo", None)]);
        let reader = PregelChannelReader::single("foo");
        let result = reader.read(&channels).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_read_single_channel_empty_no_skip() {
        let channels = make_channels(vec![("foo", None)]);
        let reader = PregelChannelReader::single("foo").with_skip_empty(false);
        let result = reader.read(&channels);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_multiple_channels() {
        let channels = make_channels(vec![
            ("a", Some(json!(1))),
            ("b", Some(json!("two"))),
            ("c", Some(json!(true))),
        ]);
        let reader =
            PregelChannelReader::new(vec!["a".into(), "b".into(), "c".into()]);
        let result = reader.read(&channels).unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!("two"));
        assert_eq!(result["c"], json!(true));
    }

    #[test]
    fn test_read_multiple_channels_some_empty() {
        let channels = make_channels(vec![
            ("a", Some(json!(1))),
            ("b", None),
        ]);
        let reader = PregelChannelReader::new(vec!["a".into(), "b".into()]);
        let result = reader.read(&channels).unwrap();
        assert_eq!(result["a"], json!(1));
        // "b" should be absent (skipped).
        assert!(result.get("b").is_none());
    }

    #[test]
    fn test_read_missing_channel_skip() {
        let channels = make_channels(vec![]);
        let reader = PregelChannelReader::single("missing");
        let result = reader.read(&channels).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_read_missing_channel_no_skip() {
        let channels = make_channels(vec![]);
        let reader = PregelChannelReader::single("missing").with_skip_empty(false);
        let result = reader.read(&channels);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_with_mapper() {
        let channels = make_channels(vec![("x", Some(json!(5)))]);
        let reader = PregelChannelReader::single("x");

        let mapper = |v: Value| {
            let n = v.as_i64().unwrap_or(0);
            json!(n * 2)
        };

        let result = reader.read_with_mapper(&channels, Some(&mapper)).unwrap();
        assert_eq!(result, json!(10));
    }

    #[test]
    fn test_read_with_mapper_none() {
        let channels = make_channels(vec![("x", Some(json!(5)))]);
        let reader = PregelChannelReader::single("x");
        let result = reader.read_with_mapper(&channels, None).unwrap();
        assert_eq!(result, json!(5));
    }

    #[test]
    fn test_read_fresh() {
        let channels = make_channels(vec![("x", Some(json!("fresh")))]);
        let reader = PregelChannelReader::single("x");
        let result = reader.read_fresh(&channels).unwrap();
        assert_eq!(result, json!("fresh"));
    }

    #[test]
    fn test_convenience_read_channel_value() {
        let channels = make_channels(vec![("key", Some(json!(123)))]);
        let result = read_channel_value(&channels, "key").unwrap();
        assert_eq!(result, json!(123));
    }

    #[test]
    fn test_convenience_read_channel_values() {
        let channels = make_channels(vec![
            ("a", Some(json!(1))),
            ("b", Some(json!(2))),
        ]);
        let names = vec!["a".to_string(), "b".to_string()];
        let result = read_channel_values(&channels, &names).unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
    }
}
