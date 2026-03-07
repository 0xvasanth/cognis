//! Message graph — a convenience wrapper for message-based state graphs.
//!
//! Provides [`MessageGraph`], a specialized graph builder for message-based workflows
//! where state is a conversation (list of [`GraphMessage`]s) that accumulates over time.
//!
//! Also provides the [`add_messages`] reducer function which merges two message
//! lists with ID-based deduplication and `RemoveMessage` support, as well as
//! [`message_graph`] for creating a pre-configured [`StateGraph`].

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::state::StateGraph;
use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// MessageRole
// ---------------------------------------------------------------------------

/// The role of a message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MessageRole {
    /// A system prompt or instruction.
    System,
    /// A message from the human user.
    Human,
    /// A message from the AI assistant.
    Ai,
    /// A tool/function result.
    Tool,
    /// A function call (legacy OpenAI style).
    Function,
    /// A custom role not covered by the standard variants.
    Custom(String),
}

impl MessageRole {
    /// Return the role as a string slice.
    pub fn as_str(&self) -> &str {
        match self {
            MessageRole::System => "system",
            MessageRole::Human => "human",
            MessageRole::Ai => "ai",
            MessageRole::Tool => "tool",
            MessageRole::Function => "function",
            MessageRole::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for MessageRole {
    fn from(s: &str) -> Self {
        match s {
            "system" => MessageRole::System,
            "human" | "user" => MessageRole::Human,
            "ai" | "assistant" => MessageRole::Ai,
            "tool" => MessageRole::Tool,
            "function" => MessageRole::Function,
            other => MessageRole::Custom(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// GraphMessage
// ---------------------------------------------------------------------------

/// A single message in a message-based workflow.
#[derive(Debug, Clone)]
pub struct GraphMessage {
    /// The role of the message author.
    pub role: MessageRole,
    /// The text content of the message.
    pub content: String,
    /// An optional name for the message author.
    pub name: Option<String>,
    /// An optional tool call ID (for tool result messages).
    pub tool_call_id: Option<String>,
    /// Arbitrary metadata attached to this message.
    pub metadata: HashMap<String, Value>,
    /// ISO-8601 timestamp string for when the message was created.
    pub timestamp: String,
}

impl GraphMessage {
    /// Create a new `GraphMessage` with the given role and content.
    ///
    /// Uses the current UTC time as the timestamp.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Set the author name (builder pattern).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the tool call ID (builder pattern).
    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Insert a metadata key-value pair (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Override the timestamp (builder pattern).
    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = ts.into();
        self
    }

    /// Serialize this message to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("role".into(), Value::String(self.role.to_string()));
        map.insert("content".into(), Value::String(self.content.clone()));
        if let Some(ref name) = self.name {
            map.insert("name".into(), Value::String(name.clone()));
        }
        if let Some(ref id) = self.tool_call_id {
            map.insert("tool_call_id".into(), Value::String(id.clone()));
        }
        if !self.metadata.is_empty() {
            map.insert(
                "metadata".into(),
                Value::Object(
                    self.metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
            );
        }
        map.insert("timestamp".into(), Value::String(self.timestamp.clone()));
        Value::Object(map)
    }

    /// Returns `true` if this is a human message.
    pub fn is_human(&self) -> bool {
        self.role == MessageRole::Human
    }

    /// Returns `true` if this is an AI message.
    pub fn is_ai(&self) -> bool {
        self.role == MessageRole::Ai
    }

    /// Returns `true` if this is a system message.
    pub fn is_system(&self) -> bool {
        self.role == MessageRole::System
    }

    /// Returns `true` if this is a tool message.
    pub fn is_tool(&self) -> bool {
        self.role == MessageRole::Tool
    }
}

// ---------------------------------------------------------------------------
// MessageState
// ---------------------------------------------------------------------------

/// The state of a message-based conversation.
///
/// Holds an ordered list of [`GraphMessage`]s representing the conversation history.
#[derive(Debug, Clone)]
pub struct MessageState {
    messages: Vec<GraphMessage>,
}

impl MessageState {
    /// Create an empty `MessageState`.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Append a message to the conversation.
    pub fn add(&mut self, message: GraphMessage) {
        self.messages.push(message);
    }

    /// Return a slice of all messages.
    pub fn messages(&self) -> &[GraphMessage] {
        &self.messages
    }

    /// Return the last message, if any.
    pub fn last(&self) -> Option<&GraphMessage> {
        self.messages.last()
    }

    /// Return the last human message, if any.
    pub fn last_human(&self) -> Option<&GraphMessage> {
        self.messages.iter().rev().find(|m| m.is_human())
    }

    /// Return the last AI message, if any.
    pub fn last_ai(&self) -> Option<&GraphMessage> {
        self.messages.iter().rev().find(|m| m.is_ai())
    }

    /// Return all messages matching the given role.
    pub fn filter_by_role(&self, role: &MessageRole) -> Vec<&GraphMessage> {
        self.messages.iter().filter(|m| &m.role == role).collect()
    }

    /// Return the number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Serialize all messages to a JSON array.
    pub fn to_json(&self) -> Value {
        Value::Array(self.messages.iter().map(|m| m.to_json()).collect())
    }

    /// Remove all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl Default for MessageState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MessageNode
// ---------------------------------------------------------------------------

/// A node in a [`MessageGraph`].
///
/// Each node has a name and a synchronous handler that receives the current
/// [`MessageState`] and returns zero or more new messages.
pub struct MessageNode {
    /// The name of this node.
    pub name: String,
    #[allow(clippy::type_complexity)]
    handler: Box<dyn Fn(&MessageState) -> Vec<GraphMessage> + Send + Sync>,
}

impl MessageNode {
    /// Create a new `MessageNode` with the given name and handler.
    pub fn new(
        name: impl Into<String>,
        handler: impl Fn(&MessageState) -> Vec<GraphMessage> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            handler: Box::new(handler),
        }
    }

    /// Execute this node against the given state, returning new messages.
    pub fn execute(&self, state: &MessageState) -> Vec<GraphMessage> {
        (self.handler)(state)
    }
}

impl fmt::Debug for MessageNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessageNode")
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MessageEdge
// ---------------------------------------------------------------------------

/// An edge in a [`MessageGraph`].
pub enum MessageEdge {
    /// A direct edge from one node to another.
    Direct {
        /// Source node name.
        from: String,
        /// Target node name.
        to: String,
    },
    /// A conditional edge whose target is determined at runtime by a router function.
    Conditional {
        /// Source node name.
        from: String,
        /// Router function: receives the current state and returns the name of the next node.
        router: Box<dyn Fn(&MessageState) -> String + Send + Sync>,
    },
}

impl MessageEdge {
    /// Return the source node name of this edge.
    pub fn source(&self) -> &str {
        match self {
            MessageEdge::Direct { from, .. } => from,
            MessageEdge::Conditional { from, .. } => from,
        }
    }
}

impl fmt::Debug for MessageEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageEdge::Direct { from, to } => f
                .debug_struct("Direct")
                .field("from", from)
                .field("to", to)
                .finish(),
            MessageEdge::Conditional { from, .. } => {
                f.debug_struct("Conditional").field("from", from).finish()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MessageGraph
// ---------------------------------------------------------------------------

/// A graph builder for message-based workflows.
///
/// Nodes consume a [`MessageState`] and produce new [`GraphMessage`]s.
/// The graph executes by following edges from the entry point to the finish point,
/// appending each node's output messages to the state.
pub struct MessageGraph {
    nodes: HashMap<String, MessageNode>,
    edges: Vec<MessageEdge>,
    entry_point: Option<String>,
    finish_point: Option<String>,
}

impl MessageGraph {
    /// Create a new empty `MessageGraph`.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: None,
            finish_point: None,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: MessageNode) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Add a direct edge from one node to another.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.edges.push(MessageEdge::Direct {
            from: from.into(),
            to: to.into(),
        });
    }

    /// Add a conditional edge from a node, using a router function to decide the target.
    pub fn add_conditional_edge(
        &mut self,
        from: impl Into<String>,
        router: impl Fn(&MessageState) -> String + Send + Sync + 'static,
    ) {
        self.edges.push(MessageEdge::Conditional {
            from: from.into(),
            router: Box::new(router),
        });
    }

    /// Set the entry point (first node to execute).
    pub fn set_entry_point(&mut self, name: impl Into<String>) {
        self.entry_point = Some(name.into());
    }

    /// Set the finish point (node whose completion ends execution).
    pub fn set_finish_point(&mut self, name: impl Into<String>) {
        self.finish_point = Some(name.into());
    }

    /// Execute the graph starting from the entry point with the given initial state.
    ///
    /// Returns the final [`MessageState`] after all nodes have been processed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No entry point has been set
    /// - A node referenced by an edge does not exist
    /// - The graph exceeds 100 steps (cycle protection)
    pub fn execute(&self, mut state: MessageState) -> Result<MessageState> {
        let entry = self
            .entry_point
            .as_ref()
            .ok_or_else(|| LangGraphError::Other("No entry point set".into()))?;

        let mut current = entry.clone();
        let max_steps = 100;

        for _ in 0..max_steps {
            // Execute the current node
            let node = self
                .nodes
                .get(&current)
                .ok_or_else(|| LangGraphError::Other(format!("Node not found: {}", current)))?;

            let new_messages = node.execute(&state);
            for msg in new_messages {
                state.add(msg);
            }

            // Check if we've reached the finish point
            if self.finish_point.as_ref() == Some(&current) {
                return Ok(state);
            }

            // Find the outgoing edge from the current node
            let next = self.resolve_next(&current, &state)?;
            match next {
                Some(n) => current = n,
                None => return Ok(state), // No outgoing edge — done
            }
        }

        Err(LangGraphError::GraphRecursionError(
            "MessageGraph exceeded 100 steps".into(),
        ))
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Validate the graph structure.
    ///
    /// Checks that:
    /// - An entry point is set and refers to an existing node
    /// - All direct edge targets refer to existing nodes
    pub fn validate(&self) -> Result<()> {
        // Check entry point
        let entry = self
            .entry_point
            .as_ref()
            .ok_or_else(|| LangGraphError::Other("No entry point set".into()))?;
        if !self.nodes.contains_key(entry) {
            return Err(LangGraphError::Other(format!(
                "Entry point '{}' does not exist as a node",
                entry
            )));
        }

        // Check finish point if set
        if let Some(ref fp) = self.finish_point {
            if !self.nodes.contains_key(fp) {
                return Err(LangGraphError::Other(format!(
                    "Finish point '{}' does not exist as a node",
                    fp
                )));
            }
        }

        // Check all direct edge endpoints exist
        for edge in &self.edges {
            match edge {
                MessageEdge::Direct { from, to } => {
                    if !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Edge source '{}' does not exist as a node",
                            from
                        )));
                    }
                    if !self.nodes.contains_key(to) {
                        return Err(LangGraphError::Other(format!(
                            "Edge target '{}' does not exist as a node",
                            to
                        )));
                    }
                }
                MessageEdge::Conditional { from, .. } => {
                    if !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Conditional edge source '{}' does not exist as a node",
                            from
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Resolve the next node to execute after `current`, given the current state.
    fn resolve_next(&self, current: &str, state: &MessageState) -> Result<Option<String>> {
        for edge in &self.edges {
            match edge {
                MessageEdge::Direct { from, to } if from == current => {
                    return Ok(Some(to.clone()));
                }
                MessageEdge::Conditional { from, router } if from == current => {
                    let target = router(state);
                    return Ok(Some(target));
                }
                _ => {}
            }
        }
        Ok(None)
    }
}

impl Default for MessageGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MessageGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessageGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("entry_point", &self.entry_point)
            .field("finish_point", &self.finish_point)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MessageGraphBuilder
// ---------------------------------------------------------------------------

/// A fluent builder for constructing a [`MessageGraph`].
pub struct MessageGraphBuilder {
    graph: MessageGraph,
}

impl MessageGraphBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            graph: MessageGraph::new(),
        }
    }

    /// Add a node with the given name and handler.
    pub fn node(
        mut self,
        name: impl Into<String>,
        handler: impl Fn(&MessageState) -> Vec<GraphMessage> + Send + Sync + 'static,
    ) -> Self {
        self.graph.add_node(MessageNode::new(name, handler));
        self
    }

    /// Add a direct edge.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.graph.add_edge(from, to);
        self
    }

    /// Add a conditional edge.
    pub fn conditional_edge(
        mut self,
        from: impl Into<String>,
        router: impl Fn(&MessageState) -> String + Send + Sync + 'static,
    ) -> Self {
        self.graph.add_conditional_edge(from, router);
        self
    }

    /// Set the entry point.
    pub fn entry(mut self, name: impl Into<String>) -> Self {
        self.graph.set_entry_point(name);
        self
    }

    /// Set the finish point.
    pub fn finish(mut self, name: impl Into<String>) -> Self {
        self.graph.set_finish_point(name);
        self
    }

    /// Validate and build the [`MessageGraph`].
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails (e.g. missing entry point, unknown edge target).
    pub fn build(self) -> Result<MessageGraph> {
        self.graph.validate()?;
        Ok(self.graph)
    }
}

impl Default for MessageGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// add_messages / message_graph (existing public API)
// ---------------------------------------------------------------------------

/// Merge two message lists with ID-based deduplication.
///
/// Messages are matched by their `"id"` field. If a message in `right` has the same
/// id as one in `left`, it replaces the original. If a message has `type: "remove"`
/// (or `type: "RemoveMessage"`), the message with the matching id is removed from the
/// result.
///
/// Messages without ids are always appended.
///
/// # Arguments
///
/// * `left` — The existing message list (a JSON array, or a single message value).
/// * `right` — The new messages to merge in (a JSON array, or a single message value).
///
/// # Returns
///
/// A `Value::Array` containing the merged messages.
///
/// # Examples
///
/// ```
/// use langgraph::graph::message::add_messages;
/// use serde_json::json;
///
/// let left = json!([
///     {"type": "human", "content": "hi", "id": "1"}
/// ]);
/// let right = json!([
///     {"type": "ai", "content": "hello!", "id": "2"}
/// ]);
/// let result = add_messages(&left, &right);
/// assert_eq!(result.as_array().unwrap().len(), 2);
/// ```
pub fn add_messages(left: &Value, right: &Value) -> Value {
    let left_msgs = match left {
        Value::Array(arr) => arr.clone(),
        _ => vec![left.clone()],
    };
    let right_msgs = match right {
        Value::Array(arr) => arr.clone(),
        _ => vec![right.clone()],
    };

    // Collect IDs to remove
    let mut ids_to_remove: HashSet<String> = HashSet::new();
    for msg in &right_msgs {
        if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
            if msg_type == "remove" || msg_type == "RemoveMessage" {
                if let Some(id) = msg.get("id").and_then(|id| id.as_str()) {
                    ids_to_remove.insert(id.to_string());
                }
            }
        }
    }

    // Build result: start with left, filtering out removed messages
    let mut result: Vec<Value> = Vec::new();
    let mut left_ids: HashMap<String, usize> = HashMap::new();

    for msg in left_msgs {
        let id = msg
            .get("id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string());
        if let Some(ref id_str) = id {
            if ids_to_remove.contains(id_str) {
                continue; // Skip removed messages
            }
            left_ids.insert(id_str.clone(), result.len());
        }
        result.push(msg);
    }

    // Add right messages, replacing by ID if one already exists in left
    for msg in right_msgs {
        let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if msg_type == "remove" || msg_type == "RemoveMessage" {
            continue; // Skip remove markers — they have already been applied
        }

        let id = msg
            .get("id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string());
        if let Some(ref id_str) = id {
            if let Some(&idx) = left_ids.get(id_str) {
                result[idx] = msg; // Replace existing
                continue;
            }
        }
        result.push(msg);
    }

    Value::Array(result)
}

/// Create a state graph pre-configured for message passing.
///
/// The resulting graph uses [`add_messages`] as the reducer for the `"messages"`
/// key in the state. Node outputs that include a `"messages"` key will have
/// their values merged using ID-based deduplication.
///
/// # Example
///
/// ```rust
/// use langgraph::graph::message::message_graph;
///
/// let graph = message_graph();
/// // Add nodes and edges, then compile.
/// ```
pub fn message_graph() -> StateGraph {
    StateGraph::new()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::state::AsyncNodeAction;
    use serde_json::json;
    use std::sync::Arc;

    // ---- MessageRole tests ----

    #[test]
    fn test_message_role_as_str() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::Human.as_str(), "human");
        assert_eq!(MessageRole::Ai.as_str(), "ai");
        assert_eq!(MessageRole::Tool.as_str(), "tool");
        assert_eq!(MessageRole::Function.as_str(), "function");
        assert_eq!(MessageRole::Custom("narrator".into()).as_str(), "narrator");
    }

    #[test]
    fn test_message_role_display() {
        assert_eq!(format!("{}", MessageRole::Human), "human");
        assert_eq!(format!("{}", MessageRole::Ai), "ai");
        assert_eq!(
            format!("{}", MessageRole::Custom("moderator".into())),
            "moderator"
        );
    }

    #[test]
    fn test_message_role_from_str() {
        assert_eq!(MessageRole::from("system"), MessageRole::System);
        assert_eq!(MessageRole::from("human"), MessageRole::Human);
        assert_eq!(MessageRole::from("user"), MessageRole::Human);
        assert_eq!(MessageRole::from("ai"), MessageRole::Ai);
        assert_eq!(MessageRole::from("assistant"), MessageRole::Ai);
        assert_eq!(MessageRole::from("tool"), MessageRole::Tool);
        assert_eq!(MessageRole::from("function"), MessageRole::Function);
        assert_eq!(
            MessageRole::from("custom_role"),
            MessageRole::Custom("custom_role".into())
        );
    }

    #[test]
    fn test_message_role_equality() {
        assert_eq!(MessageRole::Human, MessageRole::Human);
        assert_ne!(MessageRole::Human, MessageRole::Ai);
        assert_eq!(
            MessageRole::Custom("x".into()),
            MessageRole::Custom("x".into())
        );
        assert_ne!(
            MessageRole::Custom("x".into()),
            MessageRole::Custom("y".into())
        );
    }

    #[test]
    fn test_message_role_clone() {
        let role = MessageRole::Custom("test".into());
        let cloned = role.clone();
        assert_eq!(role, cloned);
    }

    // ---- GraphMessage tests ----

    #[test]
    fn test_graph_message_new() {
        let msg = GraphMessage::new(MessageRole::Human, "hello");
        assert_eq!(msg.role, MessageRole::Human);
        assert_eq!(msg.content, "hello");
        assert!(msg.name.is_none());
        assert!(msg.tool_call_id.is_none());
        assert!(msg.metadata.is_empty());
        assert!(!msg.timestamp.is_empty());
    }

    #[test]
    fn test_graph_message_builder_pattern() {
        let msg = GraphMessage::new(MessageRole::Tool, "result")
            .with_name("calculator")
            .with_tool_call_id("tc_123")
            .with_metadata("confidence", json!(0.95))
            .with_timestamp("2026-01-01T00:00:00Z");

        assert_eq!(msg.name.as_deref(), Some("calculator"));
        assert_eq!(msg.tool_call_id.as_deref(), Some("tc_123"));
        assert_eq!(msg.metadata.get("confidence"), Some(&json!(0.95)));
        assert_eq!(msg.timestamp, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn test_graph_message_to_json() {
        let msg = GraphMessage::new(MessageRole::Ai, "hi")
            .with_name("bot")
            .with_timestamp("2026-01-01T00:00:00Z");
        let json = msg.to_json();
        assert_eq!(json["role"], "ai");
        assert_eq!(json["content"], "hi");
        assert_eq!(json["name"], "bot");
        assert_eq!(json["timestamp"], "2026-01-01T00:00:00Z");
    }

    #[test]
    fn test_graph_message_to_json_minimal() {
        let msg = GraphMessage::new(MessageRole::Human, "test").with_timestamp("ts");
        let json = msg.to_json();
        assert!(json.get("name").is_none());
        assert!(json.get("tool_call_id").is_none());
        assert!(json.get("metadata").is_none());
    }

    #[test]
    fn test_graph_message_to_json_with_metadata() {
        let msg = GraphMessage::new(MessageRole::Ai, "answer")
            .with_metadata("source", json!("web"))
            .with_timestamp("ts");
        let json = msg.to_json();
        assert_eq!(json["metadata"]["source"], "web");
    }

    #[test]
    fn test_graph_message_is_human() {
        assert!(GraphMessage::new(MessageRole::Human, "hi").is_human());
        assert!(!GraphMessage::new(MessageRole::Ai, "hi").is_human());
    }

    #[test]
    fn test_graph_message_is_ai() {
        assert!(GraphMessage::new(MessageRole::Ai, "hi").is_ai());
        assert!(!GraphMessage::new(MessageRole::Human, "hi").is_ai());
    }

    #[test]
    fn test_graph_message_is_system() {
        assert!(GraphMessage::new(MessageRole::System, "prompt").is_system());
        assert!(!GraphMessage::new(MessageRole::Human, "hi").is_system());
    }

    #[test]
    fn test_graph_message_is_tool() {
        assert!(GraphMessage::new(MessageRole::Tool, "result").is_tool());
        assert!(!GraphMessage::new(MessageRole::Ai, "hi").is_tool());
    }

    #[test]
    fn test_graph_message_clone() {
        let msg = GraphMessage::new(MessageRole::Human, "hello")
            .with_name("user1")
            .with_metadata("k", json!("v"));
        let cloned = msg.clone();
        assert_eq!(cloned.content, "hello");
        assert_eq!(cloned.name.as_deref(), Some("user1"));
    }

    // ---- MessageState tests ----

    #[test]
    fn test_message_state_new_is_empty() {
        let state = MessageState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_message_state_default() {
        let state = MessageState::default();
        assert!(state.is_empty());
    }

    #[test]
    fn test_message_state_add_and_len() {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "hi"));
        state.add(GraphMessage::new(MessageRole::Ai, "hello"));
        assert_eq!(state.len(), 2);
        assert!(!state.is_empty());
    }

    #[test]
    fn test_message_state_messages() {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "msg1"));
        state.add(GraphMessage::new(MessageRole::Ai, "msg2"));
        let msgs = state.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "msg1");
        assert_eq!(msgs[1].content, "msg2");
    }

    #[test]
    fn test_message_state_last() {
        let mut state = MessageState::new();
        assert!(state.last().is_none());
        state.add(GraphMessage::new(MessageRole::Human, "first"));
        state.add(GraphMessage::new(MessageRole::Ai, "second"));
        assert_eq!(state.last().unwrap().content, "second");
    }

    #[test]
    fn test_message_state_last_human() {
        let mut state = MessageState::new();
        assert!(state.last_human().is_none());
        state.add(GraphMessage::new(MessageRole::Human, "q1"));
        state.add(GraphMessage::new(MessageRole::Ai, "a1"));
        state.add(GraphMessage::new(MessageRole::Human, "q2"));
        state.add(GraphMessage::new(MessageRole::Ai, "a2"));
        assert_eq!(state.last_human().unwrap().content, "q2");
    }

    #[test]
    fn test_message_state_last_ai() {
        let mut state = MessageState::new();
        assert!(state.last_ai().is_none());
        state.add(GraphMessage::new(MessageRole::Ai, "a1"));
        state.add(GraphMessage::new(MessageRole::Human, "q1"));
        state.add(GraphMessage::new(MessageRole::Ai, "a2"));
        assert_eq!(state.last_ai().unwrap().content, "a2");
    }

    #[test]
    fn test_message_state_filter_by_role() {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "q1"));
        state.add(GraphMessage::new(MessageRole::Ai, "a1"));
        state.add(GraphMessage::new(MessageRole::Human, "q2"));
        state.add(GraphMessage::new(MessageRole::System, "sys"));
        let humans = state.filter_by_role(&MessageRole::Human);
        assert_eq!(humans.len(), 2);
        assert_eq!(humans[0].content, "q1");
        assert_eq!(humans[1].content, "q2");
    }

    #[test]
    fn test_message_state_filter_by_role_empty_result() {
        let state = MessageState::new();
        let tools = state.filter_by_role(&MessageRole::Tool);
        assert!(tools.is_empty());
    }

    #[test]
    fn test_message_state_to_json() {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "hi").with_timestamp("ts1"));
        let json = state.to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"], "hi");
    }

    #[test]
    fn test_message_state_clear() {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "hi"));
        state.add(GraphMessage::new(MessageRole::Ai, "hello"));
        assert_eq!(state.len(), 2);
        state.clear();
        assert!(state.is_empty());
    }

    // ---- MessageNode tests ----

    #[test]
    fn test_message_node_execute() {
        let node = MessageNode::new("echo", |state: &MessageState| {
            let last = state.last().map(|m| m.content.as_str()).unwrap_or("empty");
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Echo: {}", last),
            )]
        });
        assert_eq!(node.name, "echo");

        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "hello"));
        let result = node.execute(&state);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Echo: hello");
    }

    #[test]
    fn test_message_node_execute_empty_state() {
        let node = MessageNode::new("fallback", |_state: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "No messages yet")]
        });
        let state = MessageState::new();
        let result = node.execute(&state);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "No messages yet");
    }

    #[test]
    fn test_message_node_returns_multiple_messages() {
        let node = MessageNode::new("multi", |_state: &MessageState| {
            vec![
                GraphMessage::new(MessageRole::Ai, "part 1"),
                GraphMessage::new(MessageRole::Ai, "part 2"),
            ]
        });
        let state = MessageState::new();
        let result = node.execute(&state);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_message_node_debug() {
        let node = MessageNode::new("test_node", |_: &MessageState| vec![]);
        let debug = format!("{:?}", node);
        assert!(debug.contains("test_node"));
    }

    // ---- MessageEdge tests ----

    #[test]
    fn test_message_edge_direct_source() {
        let edge = MessageEdge::Direct {
            from: "a".into(),
            to: "b".into(),
        };
        assert_eq!(edge.source(), "a");
    }

    #[test]
    fn test_message_edge_conditional_source() {
        let edge = MessageEdge::Conditional {
            from: "router".into(),
            router: Box::new(|_| "target".into()),
        };
        assert_eq!(edge.source(), "router");
    }

    #[test]
    fn test_message_edge_debug() {
        let edge = MessageEdge::Direct {
            from: "a".into(),
            to: "b".into(),
        };
        let debug = format!("{:?}", edge);
        assert!(debug.contains("Direct"));
        assert!(debug.contains("a"));
        assert!(debug.contains("b"));
    }

    // ---- MessageGraph tests ----

    #[test]
    fn test_message_graph_new() {
        let graph = MessageGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_message_graph_add_node_and_edge() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
        graph.add_node(MessageNode::new("b", |_: &MessageState| vec![]));
        graph.add_edge("a", "b");
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_message_graph_validate_no_entry() {
        let graph = MessageGraph::new();
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_message_graph_validate_entry_not_found() {
        let mut graph = MessageGraph::new();
        graph.set_entry_point("nonexistent");
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_message_graph_validate_edge_target_not_found() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
        graph.set_entry_point("a");
        graph.add_edge("a", "nonexistent");
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_message_graph_validate_success() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
        graph.add_node(MessageNode::new("b", |_: &MessageState| vec![]));
        graph.set_entry_point("a");
        graph.set_finish_point("b");
        graph.add_edge("a", "b");
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_message_graph_validate_finish_point_not_found() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
        graph.set_entry_point("a");
        graph.set_finish_point("nonexistent");
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_message_graph_execute_no_entry() {
        let graph = MessageGraph::new();
        let state = MessageState::new();
        assert!(graph.execute(state).is_err());
    }

    #[test]
    fn test_message_graph_execute_single_node() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("greet", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Hello!")]
        }));
        graph.set_entry_point("greet");
        graph.set_finish_point("greet");

        let state = MessageState::new();
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result.last().unwrap().content, "Hello!");
    }

    #[test]
    fn test_message_graph_execute_linear_flow() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("step1", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Step 1 done")]
        }));
        graph.add_node(MessageNode::new("step2", |state: &MessageState| {
            let prev = state.last().map(|m| m.content.clone()).unwrap_or_default();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("{} -> Step 2 done", prev),
            )]
        }));
        graph.add_node(MessageNode::new("step3", |state: &MessageState| {
            let prev = state.last().map(|m| m.content.clone()).unwrap_or_default();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("{} -> Step 3 done", prev),
            )]
        }));
        graph.set_entry_point("step1");
        graph.set_finish_point("step3");
        graph.add_edge("step1", "step2");
        graph.add_edge("step2", "step3");

        let state = MessageState::new();
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.last().unwrap().content.contains("Step 3 done"));
    }

    #[test]
    fn test_message_graph_execute_with_initial_messages() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("reply", |state: &MessageState| {
            let q = state
                .last_human()
                .map(|m| m.content.as_str())
                .unwrap_or("?");
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Answer to: {}", q),
            )]
        }));
        graph.set_entry_point("reply");
        graph.set_finish_point("reply");

        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "What is Rust?"));
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.last().unwrap().content, "Answer to: What is Rust?");
    }

    #[test]
    fn test_message_graph_execute_conditional_edge() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("router_node", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Routing...")]
        }));
        graph.add_node(MessageNode::new("path_a", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Took path A")]
        }));
        graph.add_node(MessageNode::new("path_b", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Took path B")]
        }));
        graph.set_entry_point("router_node");
        graph.set_finish_point("path_a");
        graph.add_conditional_edge("router_node", |state: &MessageState| {
            // Route based on whether the last human message contains "A"
            if state
                .last_human()
                .map_or(false, |m| m.content.contains("A"))
            {
                "path_a".into()
            } else {
                "path_b".into()
            }
        });
        // path_b also needs a finish so it doesn't loop — we set finish to path_a
        // but path_b has no outgoing edge so execution stops there too.

        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "Go to A"));
        let result = graph.execute(state).unwrap();
        assert_eq!(result.last().unwrap().content, "Took path A");

        let mut state2 = MessageState::new();
        state2.add(GraphMessage::new(MessageRole::Human, "Go to B"));
        // Reset finish to also accept path_b
        graph.set_finish_point("path_b");
        let result2 = graph.execute(state2).unwrap();
        assert_eq!(result2.last().unwrap().content, "Took path B");
    }

    #[test]
    fn test_message_graph_execute_no_outgoing_edge_stops() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("only", |_: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "done")]
        }));
        graph.set_entry_point("only");
        // No finish point, no edges — should still stop after the node
        let state = MessageState::new();
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_message_graph_debug() {
        let mut graph = MessageGraph::new();
        graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
        graph.set_entry_point("a");
        let debug = format!("{:?}", graph);
        assert!(debug.contains("MessageGraph"));
    }

    // ---- MessageGraphBuilder tests ----

    #[test]
    fn test_builder_basic() {
        let graph = MessageGraphBuilder::new()
            .node("a", |_: &MessageState| {
                vec![GraphMessage::new(MessageRole::Ai, "a")]
            })
            .node("b", |_: &MessageState| {
                vec![GraphMessage::new(MessageRole::Ai, "b")]
            })
            .edge("a", "b")
            .entry("a")
            .finish("b")
            .build()
            .unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_builder_no_entry_fails() {
        let result = MessageGraphBuilder::new()
            .node("a", |_: &MessageState| vec![])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_conditional_edge() {
        let graph = MessageGraphBuilder::new()
            .node("start", |_: &MessageState| {
                vec![GraphMessage::new(MessageRole::Ai, "start")]
            })
            .node("end", |_: &MessageState| {
                vec![GraphMessage::new(MessageRole::Ai, "end")]
            })
            .conditional_edge("start", |_: &MessageState| "end".into())
            .entry("start")
            .finish("end")
            .build()
            .unwrap();

        let state = MessageState::new();
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_builder_execute_full_pipeline() {
        let graph = MessageGraphBuilder::new()
            .node("intake", |_: &MessageState| {
                vec![GraphMessage::new(MessageRole::System, "Processing...")]
            })
            .node("respond", |state: &MessageState| {
                let q = state
                    .last_human()
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                vec![GraphMessage::new(
                    MessageRole::Ai,
                    format!("Reply to: {}", q),
                )]
            })
            .edge("intake", "respond")
            .entry("intake")
            .finish("respond")
            .build()
            .unwrap();

        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, "question"));
        let result = graph.execute(state).unwrap();
        assert_eq!(result.len(), 3); // human + system + ai
        assert!(result
            .last()
            .unwrap()
            .content
            .contains("Reply to: question"));
    }

    #[test]
    fn test_builder_default() {
        let builder = MessageGraphBuilder::default();
        // Should fail because no entry point
        assert!(builder.build().is_err());
    }

    // ---- add_messages tests (preserved from original) ----

    #[test]
    fn test_add_messages_append() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"}
        ]);
        let right = json!([
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_replace_by_id() {
        let left = json!([
            {"type": "human", "content": "old", "id": "1"},
            {"type": "ai", "content": "response", "id": "2"}
        ]);
        let right = json!([
            {"type": "human", "content": "updated", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["content"], "updated");
        assert_eq!(arr[1]["content"], "response");
    }

    #[test]
    fn test_add_messages_remove() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"},
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let right = json!([
            {"type": "remove", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "2");
    }

    #[test]
    fn test_add_messages_no_id_always_appends() {
        let left = json!([
            {"type": "human", "content": "hi"}
        ]);
        let right = json!([
            {"type": "ai", "content": "hello!"}
        ]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_empty_left() {
        let left = json!([]);
        let right = json!([{"type": "human", "content": "hi", "id": "1"}]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_add_messages_empty_right() {
        let left = json!([{"type": "human", "content": "hi", "id": "1"}]);
        let right = json!([]);
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_add_messages_mixed_operations() {
        let left = json!([
            {"type": "human", "content": "msg1", "id": "1"},
            {"type": "ai", "content": "msg2", "id": "2"},
            {"type": "human", "content": "msg3", "id": "3"}
        ]);
        let right = json!([
            {"type": "remove", "id": "2"},
            {"type": "human", "content": "msg1-updated", "id": "1"},
            {"type": "ai", "content": "new-msg", "id": "4"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["content"], "msg1-updated");
        assert_eq!(arr[1]["content"], "msg3");
        assert_eq!(arr[2]["content"], "new-msg");
    }

    #[test]
    fn test_add_messages_single_value_not_array() {
        let left = json!({"type": "human", "content": "hi", "id": "1"});
        let right = json!({"type": "ai", "content": "hello!", "id": "2"});
        let result = add_messages(&left, &right);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_add_messages_remove_message_variant() {
        let left = json!([
            {"type": "human", "content": "hi", "id": "1"},
            {"type": "ai", "content": "hello!", "id": "2"}
        ]);
        let right = json!([
            {"type": "RemoveMessage", "id": "1"}
        ]);
        let result = add_messages(&left, &right);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "2");
    }

    #[tokio::test]
    async fn test_message_graph_basic() {
        let graph = message_graph()
            .add_node(
                "greeter",
                Arc::new(
                    |state: Value| -> std::pin::Pin<
                        Box<
                            dyn std::future::Future<
                                    Output = std::result::Result<Value, LangGraphError>,
                                > + Send,
                        >,
                    > {
                        Box::pin(async move {
                            let name = state
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("world");
                            Ok(json!({"greeting": format!("Hello, {name}!")}))
                        })
                    },
                ) as AsyncNodeAction,
            )
            .set_entry_point("greeter")
            .set_finish_point("greeter")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"name": "Alice"})).await.unwrap();
        assert_eq!(result["greeting"], json!("Hello, Alice!"));
    }
}
