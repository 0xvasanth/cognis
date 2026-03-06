//! Input/output channel operations for the Pregel execution engine.
//!
//! This module provides functions for:
//! - Reading single or multiple channels
//! - Mapping input chunks to pending writes
//! - Mapping pending writes to output values and updates

use std::collections::HashMap;

use serde_json::Value;

use crate::channels::base::BaseChannel;
use crate::constants::{START, TASKS};
use crate::errors::LangGraphError;
use crate::types::CommandGoto;

/// Read a single channel by name, returning `None` if the channel is empty
/// and `catch` is true.
///
/// # Arguments
///
/// * `channels` - The map of channel name to channel instance.
/// * `chan` - The name of the channel to read.
/// * `catch` - If `true`, returns `Ok(None)` for empty channels instead of an error.
pub fn read_channel(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    chan: &str,
    catch: bool,
) -> Result<Option<Value>, LangGraphError> {
    match channels.get(chan) {
        Some(channel) => match channel.get() {
            Ok(value) => Ok(Some(value)),
            Err(LangGraphError::EmptyChannelError) if catch => Ok(None),
            Err(e) => Err(e),
        },
        None => {
            if catch {
                Ok(None)
            } else {
                Err(LangGraphError::Other(format!(
                    "Channel '{chan}' not found"
                )))
            }
        }
    }
}

/// Read multiple channels, returning a map of channel name to value.
///
/// Channels that are empty are skipped when `skip_empty` is true.
///
/// # Arguments
///
/// * `channels` - The map of channel name to channel instance.
/// * `select` - The channel names to read.
/// * `skip_empty` - If true, empty channels are silently omitted from the result.
pub fn read_channels(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    select: &[String],
    skip_empty: bool,
) -> Result<HashMap<String, Value>, LangGraphError> {
    let mut values = HashMap::new();
    for key in select {
        match read_channel(channels, key, skip_empty) {
            Ok(Some(value)) => {
                values.insert(key.clone(), value);
            }
            Ok(None) => {
                // Skipped empty channel.
            }
            Err(e) => return Err(e),
        }
    }
    Ok(values)
}

/// Read a single channel or multiple channels, returning either a single `Value`
/// or a JSON object with channel name keys.
///
/// When `select` is a single-element slice, returns the raw value.
/// When `select` has multiple elements, returns a JSON object.
pub fn read_channels_as_value(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    select: &[String],
    skip_empty: bool,
) -> Result<Value, LangGraphError> {
    if select.len() == 1 {
        read_channel(channels, &select[0], skip_empty)
            .map(|opt| opt.unwrap_or(Value::Null))
    } else {
        let map = read_channels(channels, select, skip_empty)?;
        Ok(serde_json::to_value(map).unwrap_or(Value::Null))
    }
}

/// A pending write: (channel_name, value).
pub type PendingWrite = (String, Value);

/// Map an input chunk to a sequence of pending writes.
///
/// If `input_channels` is a single channel name, the entire `chunk` is written to it.
/// If `input_channels` is a list, `chunk` must be a JSON object whose keys match
/// the channel names.
///
/// # Errors
///
/// Returns [`LangGraphError::InvalidUpdateError`] if `chunk` is not an object when
/// multiple input channels are specified.
pub fn map_input(
    input_channels: &[String],
    chunk: Option<Value>,
) -> Result<Vec<PendingWrite>, LangGraphError> {
    let Some(chunk) = chunk else {
        return Ok(Vec::new());
    };

    if input_channels.len() == 1 {
        return Ok(vec![(input_channels[0].clone(), chunk)]);
    }

    // Multiple input channels: chunk must be a JSON object.
    let obj = chunk.as_object().ok_or_else(|| {
        LangGraphError::InvalidUpdateError(
            "Expected input chunk to be a JSON object for multiple input channels".into(),
        )
    })?;

    let mut writes = Vec::new();
    for (key, value) in obj {
        if input_channels.iter().any(|c| c == key) {
            writes.push((key.clone(), value.clone()));
        }
    }
    Ok(writes)
}

/// Map a [`Command`](crate::types::Command) to a sequence of pending writes.
///
/// This handles the `goto`, `resume`, and `update` fields of a command.
///
/// # Errors
///
/// Returns [`LangGraphError::InvalidUpdateError`] if the command targets the parent
/// graph and there is no parent.
pub fn map_command(cmd: &crate::types::Command) -> Result<Vec<PendingWrite>, LangGraphError> {
    if cmd.graph.as_deref() == Some(crate::types::Command::PARENT) {
        return Err(LangGraphError::InvalidUpdateError(
            "There is no parent graph".into(),
        ));
    }

    let mut writes = Vec::new();

    // Process goto targets.
    for goto in &cmd.goto {
        match goto {
            CommandGoto::SendTo(send) => {
                writes.push((
                    TASKS.to_string(),
                    serde_json::to_value(send).unwrap_or(Value::Null),
                ));
            }
            CommandGoto::Node(name) => {
                writes.push((
                    format!("branch:to:{}", name),
                    Value::String(START.to_string()),
                ));
            }
        }
    }

    // Process resume.
    if let Some(ref resume) = cmd.resume {
        writes.push(("__pregel_resume".to_string(), resume.clone()));
    }

    // Process update.
    if let Some(ref update) = cmd.update {
        if let Some(obj) = update.as_object() {
            for (key, value) in obj {
                writes.push((key.clone(), value.clone()));
            }
        }
    }

    Ok(writes)
}

/// Map pending writes to output values by reading the affected output channels.
///
/// Returns an iterator of values read from the channels that were written to.
pub fn map_output_values(
    output_channels: &[String],
    pending_writes: &[PendingWrite],
    channels: &HashMap<String, Box<dyn BaseChannel>>,
) -> Vec<Value> {
    let mut results = Vec::new();

    if output_channels.len() == 1 {
        let chan = &output_channels[0];
        if pending_writes.iter().any(|(c, _)| c == chan) {
            if let Ok(Some(value)) = read_channel(channels, chan, true) {
                results.push(value);
            }
        }
    } else {
        let written_channels: std::collections::HashSet<&str> =
            pending_writes.iter().map(|(c, _)| c.as_str()).collect();

        if output_channels
            .iter()
            .any(|c| written_channels.contains(c.as_str()))
        {
            let mut output = serde_json::Map::new();
            for chan in output_channels {
                if let Ok(Some(value)) = read_channel(channels, chan, true) {
                    output.insert(chan.clone(), value);
                }
            }
            if !output.is_empty() {
                results.push(Value::Object(output));
            }
        }
    }

    results
}

/// Map pending writes to output update format.
///
/// Groups writes by task name and returns a map of node -> update.
pub fn map_output_updates(
    output_channels: &[String],
    task_writes: &[(String, Vec<PendingWrite>)],
) -> Option<HashMap<String, Value>> {
    if task_writes.is_empty() {
        return None;
    }

    let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();

    for (task_name, writes) in task_writes {
        let entry = grouped.entry(task_name.clone()).or_default();

        if output_channels.len() == 1 {
            for (chan, value) in writes {
                if chan == &output_channels[0] {
                    entry.push(value.clone());
                }
            }
        } else {
            let relevant: HashMap<String, Value> = writes
                .iter()
                .filter(|(chan, _)| output_channels.iter().any(|c| c == chan))
                .map(|(chan, value)| (chan.clone(), value.clone()))
                .collect();
            if !relevant.is_empty() {
                entry.push(serde_json::to_value(relevant).unwrap_or(Value::Null));
            }
        }
    }

    // Simplify single-element lists.
    let result: HashMap<String, Value> = grouped
        .into_iter()
        .map(|(name, values)| {
            let value = match values.len() {
                0 => Value::Null,
                1 => values.into_iter().next().unwrap(),
                _ => Value::Array(values),
            };
            (name, value)
        })
        .collect();

    if result.values().all(|v| v.is_null()) {
        None
    } else {
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper: create a simple in-memory channel for testing.
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

        fn get(&self) -> Result<Value, LangGraphError> {
            self.value
                .clone()
                .ok_or(LangGraphError::EmptyChannelError)
        }

        fn is_available(&self) -> bool {
            self.value.is_some()
        }

        fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
            if let Some(v) = values.into_iter().last() {
                self.value = Some(v);
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    fn make_channels(entries: Vec<(&str, Option<Value>)>) -> HashMap<String, Box<dyn BaseChannel>> {
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
    fn test_read_channel_available() {
        let channels = make_channels(vec![("foo", Some(json!(42)))]);
        let result = read_channel(&channels, "foo", true).unwrap();
        assert_eq!(result, Some(json!(42)));
    }

    #[test]
    fn test_read_channel_empty_catch() {
        let channels = make_channels(vec![("foo", None)]);
        let result = read_channel(&channels, "foo", true).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_read_channel_empty_no_catch() {
        let channels = make_channels(vec![("foo", None)]);
        let result = read_channel(&channels, "foo", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_channel_missing_catch() {
        let channels = make_channels(vec![]);
        let result = read_channel(&channels, "missing", true).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_read_channel_missing_no_catch() {
        let channels = make_channels(vec![]);
        let result = read_channel(&channels, "missing", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_channels_multiple() {
        let channels = make_channels(vec![
            ("a", Some(json!(1))),
            ("b", Some(json!(2))),
            ("c", None),
        ]);
        let select = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = read_channels(&channels, &select, true).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
    }

    #[test]
    fn test_read_channels_as_value_single() {
        let channels = make_channels(vec![("x", Some(json!("hello")))]);
        let select = vec!["x".to_string()];
        let result = read_channels_as_value(&channels, &select, true).unwrap();
        assert_eq!(result, json!("hello"));
    }

    #[test]
    fn test_read_channels_as_value_multiple() {
        let channels = make_channels(vec![
            ("a", Some(json!(1))),
            ("b", Some(json!(2))),
        ]);
        let select = vec!["a".to_string(), "b".to_string()];
        let result = read_channels_as_value(&channels, &select, true).unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
    }

    #[test]
    fn test_map_input_none() {
        let channels = vec!["input".to_string()];
        let result = map_input(&channels, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_map_input_single_channel() {
        let channels = vec!["input".to_string()];
        let result = map_input(&channels, Some(json!({"key": "val"}))).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "input");
        assert_eq!(result[0].1, json!({"key": "val"}));
    }

    #[test]
    fn test_map_input_multiple_channels() {
        let channels = vec!["a".to_string(), "b".to_string()];
        let result = map_input(&channels, Some(json!({"a": 1, "b": 2, "c": 3}))).unwrap();
        assert_eq!(result.len(), 2);
        let keys: Vec<&str> = result.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"a"));
        assert!(keys.contains(&"b"));
    }

    #[test]
    fn test_map_input_multiple_channels_non_object() {
        let channels = vec!["a".to_string(), "b".to_string()];
        let result = map_input(&channels, Some(json!(42)));
        assert!(result.is_err());
    }

    #[test]
    fn test_map_command_goto_node() {
        let cmd = crate::types::Command {
            graph: None,
            update: None,
            resume: None,
            goto: vec![CommandGoto::Node("target".to_string())],
        };
        let writes = map_command(&cmd).unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "branch:to:target");
        assert_eq!(writes[0].1, Value::String(START.to_string()));
    }

    #[test]
    fn test_map_command_goto_send() {
        let send = crate::types::Send {
            node: "target".to_string(),
            arg: json!({"data": 1}),
        };
        let cmd = crate::types::Command {
            graph: None,
            update: None,
            resume: None,
            goto: vec![CommandGoto::SendTo(send)],
        };
        let writes = map_command(&cmd).unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, TASKS);
    }

    #[test]
    fn test_map_command_with_update() {
        let cmd = crate::types::Command {
            graph: None,
            update: Some(json!({"x": 1, "y": 2})),
            resume: None,
            goto: vec![],
        };
        let writes = map_command(&cmd).unwrap();
        assert_eq!(writes.len(), 2);
        let keys: Vec<&str> = writes.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"x"));
        assert!(keys.contains(&"y"));
    }

    #[test]
    fn test_map_command_with_resume() {
        let cmd = crate::types::Command {
            graph: None,
            update: None,
            resume: Some(json!("continue")),
            goto: vec![],
        };
        let writes = map_command(&cmd).unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "__pregel_resume");
        assert_eq!(writes[0].1, json!("continue"));
    }

    #[test]
    fn test_map_command_parent_error() {
        let cmd = crate::types::Command {
            graph: Some(crate::types::Command::PARENT.to_string()),
            update: None,
            resume: None,
            goto: vec![],
        };
        let result = map_command(&cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_map_output_values_single_channel() {
        let channels = make_channels(vec![("out", Some(json!(99)))]);
        let output_channels = vec!["out".to_string()];
        let pending = vec![("out".to_string(), json!(99))];
        let results = map_output_values(&output_channels, &pending, &channels);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], json!(99));
    }

    #[test]
    fn test_map_output_values_no_matching_writes() {
        let channels = make_channels(vec![("out", Some(json!(99)))]);
        let output_channels = vec!["out".to_string()];
        let pending = vec![("other".to_string(), json!(1))];
        let results = map_output_values(&output_channels, &pending, &channels);
        assert!(results.is_empty());
    }

    #[test]
    fn test_map_output_updates_empty() {
        let output_channels = vec!["out".to_string()];
        let result = map_output_updates(&output_channels, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_map_output_updates_single_channel() {
        let output_channels = vec!["out".to_string()];
        let task_writes = vec![(
            "node_a".to_string(),
            vec![("out".to_string(), json!(42))],
        )];
        let result = map_output_updates(&output_channels, &task_writes);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map["node_a"], json!(42));
    }

    #[test]
    fn test_map_output_updates_multiple_channels() {
        let output_channels = vec!["a".to_string(), "b".to_string()];
        let task_writes = vec![(
            "node_x".to_string(),
            vec![
                ("a".to_string(), json!(1)),
                ("b".to_string(), json!(2)),
            ],
        )];
        let result = map_output_updates(&output_channels, &task_writes);
        assert!(result.is_some());
        let map = result.unwrap();
        let update = &map["node_x"];
        assert_eq!(update["a"], json!(1));
        assert_eq!(update["b"], json!(2));
    }

    #[test]
    fn test_map_output_updates_all_null() {
        let output_channels = vec!["out".to_string()];
        let task_writes = vec![(
            "node_a".to_string(),
            vec![("other".to_string(), json!(1))],
        )];
        let result = map_output_updates(&output_channels, &task_writes);
        assert!(result.is_none());
    }
}
