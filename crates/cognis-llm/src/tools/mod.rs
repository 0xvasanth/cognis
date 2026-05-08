//! Tool trait + ergonomic tiers + supporting types.

pub mod schema_based;
pub mod simple;
pub mod types;
pub mod validation;

pub use schema_based::SchemaBasedTool;
pub use simple::__simple_async_trait;
pub use types::{ToolInput, ToolOutput};
pub use validation::{Format, ValidateArgs};

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};

/// Tier-1 tool trait. The most general contract — manual JSON schema,
/// `serde_json::Value` arg deserialization is the tool's responsibility.
///
/// `BaseTool` is a type alias for callers (especially cognis-macros
/// generated code) that prefer the v1 name.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name as registered with the LLM.
    fn name(&self) -> &str;

    /// Description shown to the LLM.
    fn description(&self) -> &str;

    /// Optional JSON Schema for the parameters. None = no parameters.
    fn args_schema(&self) -> Option<serde_json::Value>;

    /// Hint to the agent: if true, return the tool result directly
    /// instead of looping back to the LLM.
    fn return_direct(&self) -> bool {
        false
    }

    /// Execute the tool with the given input.
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput>;
}

/// Alias for cognis-macros-generated code that emits paths to `BaseTool`.
pub use Tool as BaseTool;

/// Serializable form of a tool — what gets sent to the LLM API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Description.
    pub description: String,
    /// JSON Schema for parameters (None if the tool takes no params).
    pub parameters: Option<serde_json::Value>,
}

impl ToolDefinition {
    /// Build a `ToolDefinition` from any `&dyn Tool`.
    pub fn from_tool(t: &dyn Tool) -> Self {
        Self {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.args_schema(),
        }
    }
}

/// Per-tool runtime state — enabled/disabled flag + call counter.
#[derive(Default)]
struct ToolEntry {
    tool: Option<Arc<dyn Tool>>,
    enabled: bool,
    /// Total successful + failed calls served via [`ToolRegistry::execute`].
    calls: std::sync::atomic::AtomicUsize,
    /// Optional permission predicate: `agent_id → allowed?`.
    #[allow(clippy::type_complexity)]
    permission: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl Clone for ToolEntry {
    fn clone(&self) -> Self {
        Self {
            tool: self.tool.clone(),
            enabled: self.enabled,
            calls: std::sync::atomic::AtomicUsize::new(
                self.calls.load(std::sync::atomic::Ordering::Relaxed),
            ),
            permission: self.permission.clone(),
        }
    }
}

/// HashMap-backed tool registry. The agent layer uses this to dispatch
/// tool calls returned by the LLM.
///
/// Per-tool controls (added in V2):
/// - [`ToolRegistry::enable`] / [`ToolRegistry::disable`] toggle
///   availability without unregistering.
/// - [`ToolRegistry::set_permission`] attaches an `agent_id → allowed`
///   predicate. [`ToolRegistry::is_allowed`] checks; [`ToolRegistry::execute_for`]
///   enforces.
/// - [`ToolRegistry::call_count`] returns the cumulative dispatch count
///   for any tool (incremented on every `execute*` call).
#[derive(Default, Clone)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,
}

impl ToolRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. Replaces any existing tool with the same name
    /// (preserving its enabled flag and call counter? no — replaces
    /// wholesale; the new tool starts enabled with count = 0).
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.entries.insert(
            name,
            ToolEntry {
                tool: Some(tool),
                enabled: true,
                calls: std::sync::atomic::AtomicUsize::new(0),
                permission: None,
            },
        );
    }

    /// Register `alias` to point at the same tool as `name`. No-op when
    /// `name` is not registered.
    pub fn register_alias(&mut self, alias: impl Into<String>, name: &str) {
        if let Some(t) = self.entries.get(name).and_then(|e| e.tool.clone()) {
            self.entries.insert(
                alias.into(),
                ToolEntry {
                    tool: Some(t),
                    enabled: true,
                    calls: std::sync::atomic::AtomicUsize::new(0),
                    permission: None,
                },
            );
        }
    }

    /// Remove a tool by name. Returns `true` if it was present.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Filter the registry: keep only tools whose name passes `predicate`.
    /// Returns the names of removed tools.
    pub fn retain<F>(&mut self, mut predicate: F) -> Vec<String>
    where
        F: FnMut(&str) -> bool,
    {
        let mut removed = Vec::new();
        self.entries.retain(|k, _| {
            let keep = predicate(k);
            if !keep {
                removed.push(k.clone());
            }
            keep
        });
        removed
    }

    /// Get a tool by name. Returns the inner `Arc` only if the tool is
    /// **enabled** — disabled tools are invisible to dispatch.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        let e = self.entries.get(name)?;
        if !e.enabled {
            return None;
        }
        e.tool.as_ref()
    }

    /// True if a tool with this name is registered (regardless of enabled state).
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// True if a tool with this name is registered AND enabled.
    pub fn is_enabled(&self, name: &str) -> bool {
        self.entries.get(name).is_some_and(|e| e.enabled)
    }

    /// Disable a tool without unregistering it. Disabled tools are
    /// hidden from `get` / `tool_names` / `definitions` / `execute`,
    /// but their call counters and permissions persist for when they're
    /// re-enabled. No-op if the tool isn't registered.
    pub fn disable(&mut self, name: &str) -> bool {
        match self.entries.get_mut(name) {
            Some(e) => {
                e.enabled = false;
                true
            }
            None => false,
        }
    }

    /// Re-enable a previously-disabled tool. No-op if not registered.
    pub fn enable(&mut self, name: &str) -> bool {
        match self.entries.get_mut(name) {
            Some(e) => {
                e.enabled = true;
                true
            }
            None => false,
        }
    }

    /// Attach a permission predicate `agent_id → allowed`. Replace by
    /// calling again. Pass [`ToolRegistry::clear_permission`] to remove.
    pub fn set_permission<F>(&mut self, name: &str, predicate: F) -> bool
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        match self.entries.get_mut(name) {
            Some(e) => {
                e.permission = Some(Arc::new(predicate));
                true
            }
            None => false,
        }
    }

    /// Drop the permission predicate. The tool reverts to "any agent allowed".
    pub fn clear_permission(&mut self, name: &str) {
        if let Some(e) = self.entries.get_mut(name) {
            e.permission = None;
        }
    }

    /// True if the tool exists, is enabled, and `agent_id` passes the
    /// permission predicate (or no predicate is set). Disabled tools
    /// always return false.
    pub fn is_allowed(&self, name: &str, agent_id: &str) -> bool {
        let Some(e) = self.entries.get(name) else {
            return false;
        };
        if !e.enabled {
            return false;
        }
        match &e.permission {
            Some(p) => p(agent_id),
            None => true,
        }
    }

    /// Number of times the tool was dispatched via `execute` /
    /// `execute_for` (across all agents, including failed dispatches).
    /// Returns 0 if the tool isn't registered.
    pub fn call_count(&self, name: &str) -> usize {
        self.entries
            .get(name)
            .map(|e| e.calls.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// All registered + enabled tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, e)| e.enabled)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Build `ToolDefinition`s for every registered + enabled tool.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.entries
            .values()
            .filter(|e| e.enabled)
            .filter_map(|e| e.tool.as_ref())
            .map(|t| ToolDefinition::from_tool(t.as_ref()))
            .collect()
    }

    /// Execute a tool by name with the given input. Increments the
    /// per-tool call counter regardless of success/failure. Errors if
    /// the tool isn't registered or is disabled.
    pub async fn execute(&self, name: &str, input: ToolInput) -> Result<ToolOutput> {
        let entry = self.entries.get(name).ok_or_else(|| CognisError::Tool {
            name: name.to_string(),
            reason: "not registered".into(),
        })?;
        if !entry.enabled {
            return Err(CognisError::Tool {
                name: name.to_string(),
                reason: "disabled".into(),
            });
        }
        let t = entry.tool.as_ref().ok_or_else(|| CognisError::Tool {
            name: name.to_string(),
            reason: "no implementation".into(),
        })?;
        entry
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        t._run(input).await
    }

    /// Like [`ToolRegistry::execute`] but also enforces the permission
    /// predicate against `agent_id`. Errors with a permission error if
    /// the predicate returns false.
    pub async fn execute_for(
        &self,
        name: &str,
        agent_id: &str,
        input: ToolInput,
    ) -> Result<ToolOutput> {
        if !self.is_allowed(name, agent_id) {
            return Err(CognisError::Tool {
                name: name.to_string(),
                reason: format!("not allowed for agent `{agent_id}`"),
            });
        }
        self.execute(name, input).await
    }

    /// Number of registered tools (including disabled ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes input"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            Some(json!({"type": "object", "properties": {"text": {"type": "string"}}}))
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Content(input.into_json()))
        }
    }

    #[tokio::test]
    async fn registry_register_get_execute() {
        let mut reg = ToolRegistry::new();
        assert!(reg.is_empty());
        reg.register(Arc::new(Echo));
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("echo"));

        let mut m = HashMap::new();
        m.insert("text".into(), json!("hi"));
        let out = reg.execute("echo", ToolInput::Structured(m)).await.unwrap();
        match out {
            ToolOutput::Content(v) => assert_eq!(v["text"], "hi"),
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let reg = ToolRegistry::new();
        let err = reg
            .execute("missing", ToolInput::Text("x".into()))
            .await
            .unwrap_err();
        assert_eq!(err.category(), "tool");
    }

    #[test]
    fn definition_from_tool() {
        let d = ToolDefinition::from_tool(&Echo);
        assert_eq!(d.name, "echo");
        assert_eq!(d.description, "echoes input");
        assert!(d.parameters.is_some());
    }

    #[tokio::test]
    async fn disable_hides_from_dispatch_and_listing() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Echo));
        assert!(reg.disable("echo"));
        assert!(reg.contains("echo"), "still registered");
        assert!(!reg.is_enabled("echo"));
        assert!(reg.tool_names().is_empty());
        assert!(reg.definitions().is_empty());
        let err = reg
            .execute("echo", ToolInput::Text("x".into()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("disabled"), "got: {err}");
    }

    #[tokio::test]
    async fn enable_restores() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Echo));
        reg.disable("echo");
        reg.enable("echo");
        assert!(reg.is_enabled("echo"));
        assert!(reg
            .execute("echo", ToolInput::Text("x".into()))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn call_count_increments_on_execute() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Echo));
        assert_eq!(reg.call_count("echo"), 0);
        for _ in 0..3 {
            reg.execute("echo", ToolInput::Text("hi".into()))
                .await
                .unwrap();
        }
        assert_eq!(reg.call_count("echo"), 3);
        assert_eq!(reg.call_count("missing"), 0);
    }

    #[tokio::test]
    async fn permission_predicate_blocks_disallowed_agents() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Echo));
        reg.set_permission("echo", |agent_id: &str| agent_id == "writer");
        assert!(reg.is_allowed("echo", "writer"));
        assert!(!reg.is_allowed("echo", "intruder"));
        let ok = reg
            .execute_for("echo", "writer", ToolInput::Text("hi".into()))
            .await;
        assert!(ok.is_ok());
        let denied = reg
            .execute_for("echo", "intruder", ToolInput::Text("hi".into()))
            .await
            .unwrap_err();
        assert!(denied.to_string().contains("not allowed"), "got: {denied}");
    }

    #[tokio::test]
    async fn clear_permission_reopens_dispatch() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Echo));
        reg.set_permission("echo", |_: &str| false);
        assert!(!reg.is_allowed("echo", "anyone"));
        reg.clear_permission("echo");
        assert!(reg.is_allowed("echo", "anyone"));
    }
}
