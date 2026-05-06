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

use cognis2_core::{CognisError, Result};

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

/// HashMap-backed tool registry. The agent layer uses this to dispatch
/// tool calls returned by the LLM.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. Replaces any existing tool with the same name.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// True if a tool with this name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// All registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Build `ToolDefinition`s for every registered tool.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition::from_tool(t.as_ref()))
            .collect()
    }

    /// Execute a tool by name with the given input.
    pub async fn execute(&self, name: &str, input: ToolInput) -> Result<ToolOutput> {
        let t = self.get(name).ok_or_else(|| CognisError::Tool {
            name: name.to_string(),
            reason: "not registered".into(),
        })?;
        t._run(input).await
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
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
}
