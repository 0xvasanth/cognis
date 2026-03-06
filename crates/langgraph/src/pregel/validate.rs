//! Graph structure validation for the Pregel execution engine.
//!
//! Validates that a Pregel graph is well-formed before execution:
//! - All referenced channels exist
//! - No reserved names are used for nodes or channels
//! - Input/output channels are properly subscribed
//! - Interrupt node references are valid

use std::collections::{HashMap, HashSet};

use crate::channels::base::BaseChannel;
use crate::constants::{END, START};
use crate::errors::LangGraphError;

/// Reserved names that cannot be used for nodes or channels.
const RESERVED_NAMES: &[&str] = &[
    "__start__",
    "__end__",
    "__interrupt__",
    "__pregel_tasks",
    "__pregel_resume",
];

/// Specification of a node's channel connections for validation purposes.
#[derive(Debug, Clone)]
pub struct NodeChannelSpec {
    /// The channels this node reads from (its input).
    pub input_channels: Vec<String>,
    /// The channels that trigger this node when written to.
    pub trigger_channels: Vec<String>,
}

/// Configuration for graph validation.
#[derive(Debug)]
pub struct ValidationConfig {
    /// Map of node name to its channel specification.
    pub nodes: HashMap<String, NodeChannelSpec>,
    /// The defined channels in the graph.
    pub channels: HashMap<String, Box<dyn BaseChannel>>,
    /// Input channels for the graph.
    pub input_channels: Vec<String>,
    /// Output channels for the graph.
    pub output_channels: Vec<String>,
    /// Channels to stream (if any).
    pub stream_channels: Option<Vec<String>>,
    /// Nodes to interrupt after (empty for none, or list of node names).
    pub interrupt_after_nodes: InterruptSpec,
    /// Nodes to interrupt before (empty for none, or list of node names).
    pub interrupt_before_nodes: InterruptSpec,
}

/// Specification for which nodes to interrupt on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterruptSpec {
    /// Interrupt on all nodes.
    All,
    /// Interrupt on specific nodes.
    Nodes(Vec<String>),
}

impl InterruptSpec {
    /// Create an interrupt spec for no nodes.
    pub fn none() -> Self {
        Self::Nodes(Vec::new())
    }
}

/// Validate the complete graph structure.
///
/// Checks:
/// 1. No reserved names used for channels or nodes
/// 2. All node input channels reference known channels
/// 3. All trigger channels reference known channels
/// 4. Input channels exist and are subscribed to by at least one node
/// 5. Output and stream channels exist
/// 6. Interrupt node references are valid
///
/// # Errors
///
/// Returns a descriptive [`LangGraphError::Other`] on the first validation failure.
pub fn validate_graph(config: &ValidationConfig) -> Result<(), LangGraphError> {
    validate_no_reserved_channels(&config.channels)?;
    validate_no_reserved_nodes(&config.nodes)?;
    validate_node_channels(&config.nodes, &config.channels)?;
    validate_input_channels(&config.input_channels, &config.channels, &config.nodes)?;
    validate_output_channels(&config.output_channels, &config.channels)?;

    if let Some(ref stream_channels) = config.stream_channels {
        validate_output_channels(stream_channels, &config.channels)?;
    }

    validate_interrupt_nodes(&config.interrupt_after_nodes, &config.nodes)?;
    validate_interrupt_nodes(&config.interrupt_before_nodes, &config.nodes)?;

    Ok(())
}

/// Validate that no channel uses a reserved name.
fn validate_no_reserved_channels(
    channels: &HashMap<String, Box<dyn BaseChannel>>,
) -> Result<(), LangGraphError> {
    for name in channels.keys() {
        if RESERVED_NAMES.contains(&name.as_str()) {
            return Err(LangGraphError::Other(format!(
                "Channel name '{}' is reserved",
                name
            )));
        }
    }
    Ok(())
}

/// Validate that no node uses a reserved name.
fn validate_no_reserved_nodes(
    nodes: &HashMap<String, NodeChannelSpec>,
) -> Result<(), LangGraphError> {
    for name in nodes.keys() {
        if RESERVED_NAMES.contains(&name.as_str()) {
            return Err(LangGraphError::Other(format!(
                "Node name '{}' is reserved",
                name
            )));
        }
    }
    Ok(())
}

/// Validate that all node channel references point to known channels.
fn validate_node_channels(
    nodes: &HashMap<String, NodeChannelSpec>,
    channels: &HashMap<String, Box<dyn BaseChannel>>,
) -> Result<(), LangGraphError> {
    for (node_name, spec) in nodes {
        // Check input channels.
        for chan in &spec.input_channels {
            if !channels.contains_key(chan) {
                return Err(LangGraphError::Other(format!(
                    "Node '{}' reads channel '{}' not in known channels",
                    node_name, chan
                )));
            }
        }

        // Check trigger channels.
        for chan in &spec.trigger_channels {
            if !channels.contains_key(chan) {
                return Err(LangGraphError::Other(format!(
                    "Subscribed channel '{}' (trigger for node '{}') not in known channels",
                    chan, node_name
                )));
            }
        }
    }
    Ok(())
}

/// Validate that input channels exist and are subscribed to.
fn validate_input_channels(
    input_channels: &[String],
    channels: &HashMap<String, Box<dyn BaseChannel>>,
    nodes: &HashMap<String, NodeChannelSpec>,
) -> Result<(), LangGraphError> {
    // Collect all subscribed (trigger) channels.
    let subscribed: HashSet<&str> = nodes
        .values()
        .flat_map(|spec| spec.trigger_channels.iter().map(String::as_str))
        .collect();

    for chan in input_channels {
        if !channels.contains_key(chan) {
            return Err(LangGraphError::Other(format!(
                "Input channel '{}' not in known channels",
                chan
            )));
        }
    }

    if input_channels.len() == 1 {
        if !subscribed.contains(input_channels[0].as_str()) {
            return Err(LangGraphError::Other(format!(
                "Input channel '{}' is not subscribed to by any node",
                input_channels[0]
            )));
        }
    } else if !input_channels.is_empty()
        && input_channels
            .iter()
            .all(|c| !subscribed.contains(c.as_str()))
    {
        return Err(LangGraphError::Other(format!(
            "None of the input channels {:?} are subscribed to by any node",
            input_channels
        )));
    }

    Ok(())
}

/// Validate that output channels exist.
fn validate_output_channels(
    output_channels: &[String],
    channels: &HashMap<String, Box<dyn BaseChannel>>,
) -> Result<(), LangGraphError> {
    for chan in output_channels {
        if !channels.contains_key(chan) {
            return Err(LangGraphError::Other(format!(
                "Output channel '{}' not in known channels",
                chan
            )));
        }
    }
    Ok(())
}

/// Validate interrupt node references.
fn validate_interrupt_nodes(
    spec: &InterruptSpec,
    nodes: &HashMap<String, NodeChannelSpec>,
) -> Result<(), LangGraphError> {
    if let InterruptSpec::Nodes(names) = spec {
        for name in names {
            if !nodes.contains_key(name) {
                return Err(LangGraphError::Other(format!(
                    "Interrupt node '{}' not in nodes",
                    name
                )));
            }
        }
    }
    Ok(())
}

/// Validate that a set of keys all exist in the given channels.
///
/// This is a simpler validation helper for checking arbitrary key sets against
/// a channel map.
pub fn validate_keys(
    keys: &[String],
    channels: &HashMap<String, Box<dyn BaseChannel>>,
) -> Result<(), LangGraphError> {
    for key in keys {
        if !channels.contains_key(key) {
            return Err(LangGraphError::Other(format!(
                "Key '{}' not in channels",
                key
            )));
        }
    }
    Ok(())
}

/// Check that all nodes have at least one outgoing edge.
///
/// This ensures no node is a dead-end (except nodes that lead to END).
pub fn validate_all_nodes_reachable(
    nodes: &[String],
    edges: &HashMap<String, Vec<String>>,
) -> Result<(), LangGraphError> {
    for node in nodes {
        if node == END || node == START {
            continue;
        }
        if !edges.contains_key(node) || edges[node].is_empty() {
            return Err(LangGraphError::Other(format!(
                "Node '{}' has no outgoing edges",
                node
            )));
        }
    }
    Ok(())
}

/// Check for unreachable nodes — nodes not reachable from any entry point.
///
/// Performs a BFS from the start node and reports any nodes not visited.
pub fn validate_no_unreachable_nodes(
    start_node: &str,
    all_nodes: &[String],
    edges: &HashMap<String, Vec<String>>,
) -> Result<(), LangGraphError> {
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(start_node.to_string());
    visited.insert(start_node.to_string());

    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = edges.get(&current) {
            for neighbor in neighbors {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }

    for node in all_nodes {
        if node != START && node != END && !visited.contains(node) {
            return Err(LangGraphError::Other(format!(
                "Node '{}' is not reachable from '{}'",
                node, start_node
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Minimal test channel.
    #[derive(Debug, Clone)]
    struct TestChannel {
        key: String,
    }

    impl BaseChannel for TestChannel {
        fn key(&self) -> &str {
            &self.key
        }
        fn box_clone(&self) -> Box<dyn BaseChannel> {
            Box::new(self.clone())
        }
        fn checkpoint(&self) -> Option<Value> {
            None
        }
        fn from_checkpoint(&self, _: Option<Value>) -> Box<dyn BaseChannel> {
            Box::new(self.clone())
        }
        fn get(&self) -> Result<Value, LangGraphError> {
            Err(LangGraphError::EmptyChannelError)
        }
        fn is_available(&self) -> bool {
            false
        }
        fn update(&mut self, _: Vec<Value>) -> Result<bool, LangGraphError> {
            Ok(false)
        }
    }

    fn make_channel(name: &str) -> Box<dyn BaseChannel> {
        Box::new(TestChannel {
            key: name.to_string(),
        })
    }

    fn make_channel_map(names: &[&str]) -> HashMap<String, Box<dyn BaseChannel>> {
        names
            .iter()
            .map(|n| (n.to_string(), make_channel(n)))
            .collect()
    }

    #[test]
    fn test_validate_graph_success() {
        let channels = make_channel_map(&["input", "output", "state"]);

        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec!["state".to_string()],
                trigger_channels: vec!["input".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec!["output".to_string()],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        assert!(validate_graph(&config).is_ok());
    }

    #[test]
    fn test_validate_reserved_channel_name() {
        let mut channels = make_channel_map(&["ok_chan"]);
        channels.insert("__start__".to_string(), make_channel("__start__"));

        let config = ValidationConfig {
            nodes: HashMap::new(),
            channels,
            input_channels: vec![],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("reserved"));
    }

    #[test]
    fn test_validate_reserved_node_name() {
        let channels = make_channel_map(&["ch"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "__end__".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec![],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec![],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_node_reads_unknown_channel() {
        let channels = make_channel_map(&["known"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec!["unknown_chan".to_string()],
                trigger_channels: vec!["known".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["known".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown_chan"));
    }

    #[test]
    fn test_validate_trigger_unknown_channel() {
        let channels = make_channel_map(&["known"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec!["known".to_string()],
                trigger_channels: vec!["missing_trigger".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["known".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("missing_trigger"));
    }

    #[test]
    fn test_validate_input_channel_not_in_channels() {
        let channels = make_channel_map(&["other"]);
        let nodes = HashMap::new();

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["nonexistent".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_input_channel_not_subscribed() {
        let channels = make_channel_map(&["input"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec![], // Not subscribed to "input"
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not subscribed"));
    }

    #[test]
    fn test_validate_output_channel_missing() {
        let channels = make_channel_map(&["input"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec!["input".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec!["missing_output".to_string()],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_stream_channel_missing() {
        let channels = make_channel_map(&["input"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec!["input".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec![],
            stream_channels: Some(vec!["missing_stream".to_string()]),
            interrupt_after_nodes: InterruptSpec::none(),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_interrupt_unknown_node() {
        let channels = make_channel_map(&["input"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec!["input".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::Nodes(vec!["nonexistent".to_string()]),
            interrupt_before_nodes: InterruptSpec::none(),
        };

        let result = validate_graph(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_interrupt_all_is_ok() {
        let channels = make_channel_map(&["input"]);
        let mut nodes = HashMap::new();
        nodes.insert(
            "node_a".to_string(),
            NodeChannelSpec {
                input_channels: vec![],
                trigger_channels: vec!["input".to_string()],
            },
        );

        let config = ValidationConfig {
            nodes,
            channels,
            input_channels: vec!["input".to_string()],
            output_channels: vec![],
            stream_channels: None,
            interrupt_after_nodes: InterruptSpec::All,
            interrupt_before_nodes: InterruptSpec::All,
        };

        assert!(validate_graph(&config).is_ok());
    }

    #[test]
    fn test_validate_keys_success() {
        let channels = make_channel_map(&["a", "b", "c"]);
        let keys = vec!["a".to_string(), "b".to_string()];
        assert!(validate_keys(&keys, &channels).is_ok());
    }

    #[test]
    fn test_validate_keys_missing() {
        let channels = make_channel_map(&["a"]);
        let keys = vec!["a".to_string(), "missing".to_string()];
        assert!(validate_keys(&keys, &channels).is_err());
    }

    #[test]
    fn test_validate_all_nodes_reachable_success() {
        let nodes = vec!["a".to_string(), "b".to_string()];
        let mut edges = HashMap::new();
        edges.insert("a".to_string(), vec!["b".to_string()]);
        edges.insert("b".to_string(), vec![END.to_string()]);

        assert!(validate_all_nodes_reachable(&nodes, &edges).is_ok());
    }

    #[test]
    fn test_validate_all_nodes_reachable_dead_end() {
        let nodes = vec!["a".to_string(), "b".to_string()];
        let mut edges = HashMap::new();
        edges.insert("a".to_string(), vec!["b".to_string()]);
        // "b" has no outgoing edges.

        let result = validate_all_nodes_reachable(&nodes, &edges);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("b"));
    }

    #[test]
    fn test_validate_no_unreachable_nodes_success() {
        let all_nodes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut edges = HashMap::new();
        edges.insert(START.to_string(), vec!["a".to_string()]);
        edges.insert("a".to_string(), vec!["b".to_string()]);
        edges.insert("b".to_string(), vec!["c".to_string()]);

        assert!(validate_no_unreachable_nodes(START, &all_nodes, &edges).is_ok());
    }

    #[test]
    fn test_validate_no_unreachable_nodes_unreachable() {
        let all_nodes = vec!["a".to_string(), "b".to_string(), "isolated".to_string()];
        let mut edges = HashMap::new();
        edges.insert(START.to_string(), vec!["a".to_string()]);
        edges.insert("a".to_string(), vec!["b".to_string()]);
        // "isolated" is not reachable.

        let result = validate_no_unreachable_nodes(START, &all_nodes, &edges);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("isolated"));
    }

    #[test]
    fn test_interrupt_spec_none() {
        let spec = InterruptSpec::none();
        assert_eq!(spec, InterruptSpec::Nodes(vec![]));
    }
}
