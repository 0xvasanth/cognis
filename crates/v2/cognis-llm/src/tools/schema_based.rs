//! Tier-2 ergonomic: SchemaBasedTool with typed Params + auto-derived schema.
//!
//! The blanket impl `impl<T: SchemaBasedTool> Tool for T` means anything
//! implementing this trait gets the `Tool` surface for free.

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;

use cognis2_core::{CognisError, JsonSchema, Result};

use super::types::{ToolInput, ToolOutput};
use super::Tool;

/// Tier-2 tool: typed args + auto-derived JSON Schema. Implementers only
/// write `execute_typed`; deserialization, schema, and the Tool surface
/// come for free via the blanket impl below.
#[async_trait]
pub trait SchemaBasedTool: Send + Sync {
    /// Typed parameter struct (must derive `JsonSchema` + `Deserialize`).
    type Params: JsonSchema + DeserializeOwned + Send + Sync + 'static;

    /// Typed output (must be `Serialize`).
    type Output: Serialize + Send + 'static;

    /// Tool name.
    fn name(&self) -> &str;

    /// Tool description.
    fn description(&self) -> &str;

    /// Execute with typed params.
    async fn execute_typed(&self, params: Self::Params) -> Result<Self::Output>;

    /// Optional override — return true to short-circuit the agent loop.
    fn return_direct(&self) -> bool {
        false
    }
}

#[async_trait]
impl<T> Tool for T
where
    T: SchemaBasedTool,
{
    fn name(&self) -> &str {
        SchemaBasedTool::name(self)
    }

    fn description(&self) -> &str {
        SchemaBasedTool::description(self)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        // Use the OpenAI-strict helper from schema.rs for compat.
        Some(crate::schema::schema_for_tool::<T::Params>())
    }

    fn return_direct(&self) -> bool {
        SchemaBasedTool::return_direct(self)
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let json = input.into_json();
        let params: T::Params =
            serde_json::from_value(json).map_err(|e| CognisError::ToolValidation(e.to_string()))?;
        let out = self.execute_typed(params).await?;
        let v = serde_json::to_value(out)?;
        Ok(ToolOutput::Content(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, JsonSchema, Deserialize)]
    struct AddParams {
        a: f64,
        b: f64,
    }

    #[derive(Debug, Serialize)]
    struct AddResult {
        sum: f64,
    }

    struct Adder;

    #[async_trait]
    impl SchemaBasedTool for Adder {
        type Params = AddParams;
        type Output = AddResult;
        fn name(&self) -> &str {
            "add"
        }
        fn description(&self) -> &str {
            "Add two numbers"
        }
        async fn execute_typed(&self, p: AddParams) -> Result<AddResult> {
            Ok(AddResult { sum: p.a + p.b })
        }
    }

    #[tokio::test]
    async fn blanket_impl_provides_tool_surface() {
        let a = Adder;
        // Tool methods all work via blanket impl
        assert_eq!(<Adder as Tool>::name(&a), "add");
        assert_eq!(<Adder as Tool>::description(&a), "Add two numbers");
        let schema = <Adder as Tool>::args_schema(&a).unwrap();
        assert_eq!(schema["type"], "object");
        let s = schema.to_string();
        assert!(!s.contains("$ref"), "OpenAI-strict: no $ref");

        let mut m = std::collections::HashMap::new();
        m.insert("a".into(), serde_json::json!(2.5));
        m.insert("b".into(), serde_json::json!(3.5));
        let out = <Adder as Tool>::_run(&a, ToolInput::Structured(m))
            .await
            .unwrap();
        match out {
            ToolOutput::Content(v) => assert_eq!(v["sum"], 6.0),
            _ => panic!("wrong variant"),
        }
    }
}
