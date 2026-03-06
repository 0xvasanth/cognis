use std::sync::Arc;

use serde_json::{json, Value};

use super::base::BaseTool;
use super::simple::SimpleTool;
use super::structured::StructuredTool;
use crate::runnables::base::Runnable;

/// Convert a single tool to OpenAI function-calling format.
pub fn convert_to_openai_tool(tool: &dyn BaseTool) -> Value {
    let mut function = json!({
        "name": tool.name(),
        "description": tool.description()
    });
    if let Some(schema) = tool.args_schema() {
        function["parameters"] = schema;
    }
    json!({
        "type": "function",
        "function": function
    })
}

/// Convert multiple tools to OpenAI function-calling format.
pub fn convert_to_openai_tools(tools: &[&dyn BaseTool]) -> Vec<Value> {
    tools.iter().map(|t| convert_to_openai_tool(*t)).collect()
}

/// Convert a `Runnable` into a `BaseTool`.
///
/// This is the Rust equivalent of Python's `convert_runnable_to_tool()` from
/// `langchain_core.tools.convert`. It wraps a `Runnable` (which takes `Value`
/// input and produces `Value` output) into a tool that can be used by agents.
///
/// If the runnable accepts a simple string input, a [`SimpleTool`] is returned.
/// Otherwise, a [`StructuredTool`] is returned that parses structured JSON arguments.
///
/// # Arguments
///
/// * `runnable` — The `Runnable` to convert.
/// * `name` — Name for the tool. If `None`, uses the runnable's name.
/// * `description` — Description for the tool. If `None`, generates a placeholder.
/// * `schema` — Optional JSON Schema for structured input. If provided and the schema
///   type is `"object"`, a `StructuredTool` is created with that schema.
///
/// # Example
///
/// ```ignore
/// use rustchain_core::tools::convert_runnable_to_tool;
///
/// let tool = convert_runnable_to_tool(my_runnable, Some("my_tool"), Some("Does something"), None);
/// ```
pub fn convert_runnable_to_tool(
    runnable: Arc<dyn Runnable>,
    name: Option<&str>,
    description: Option<&str>,
    schema: Option<Value>,
) -> Box<dyn BaseTool> {
    let tool_name = name
        .map(|n| n.to_string())
        .unwrap_or_else(|| runnable.name().to_string());
    let tool_description = description
        .map(|d| d.to_string())
        .unwrap_or_else(|| format!("Wrapper around {}", tool_name));

    // Determine whether the schema indicates a string input or structured input.
    let is_string_schema = schema
        .as_ref()
        .and_then(|s| s.get("type"))
        .and_then(|t| t.as_str())
        .map(|t| t == "string")
        .unwrap_or(false);

    if is_string_schema || schema.is_none() {
        // Wrap as a SimpleTool for string input.
        let runnable_clone = Arc::clone(&runnable);
        // SimpleTool uses a sync function; we block on the async invoke.
        // For true async usage, callers should use the async variant or StructuredTool.
        let simple = SimpleTool::new_async(tool_name, tool_description, move |input: String| {
            let r = Arc::clone(&runnable_clone);
            async move {
                let result = r.invoke(Value::String(input), None).await?;
                match result {
                    Value::String(s) => Ok(s),
                    other => Ok(other.to_string()),
                }
            }
        });
        Box::new(simple)
    } else {
        // Wrap as a StructuredTool for object input.
        let tool_schema = schema.unwrap_or_else(|| {
            json!({
                "type": "object",
                "properties": {}
            })
        });
        let runnable_clone = Arc::clone(&runnable);
        let structured =
            StructuredTool::new(tool_name, tool_description, tool_schema, move |args| {
                let r = Arc::clone(&runnable_clone);
                async move {
                    let input =
                        Value::Object(args.into_iter().collect::<serde_json::Map<String, Value>>());
                    r.invoke(input, None).await
                }
            });
        Box::new(structured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::runnables::config::RunnableConfig;
    use async_trait::async_trait;
    use serde_json::json;

    struct EchoRunnable;

    #[async_trait]
    impl Runnable for EchoRunnable {
        fn name(&self) -> &str {
            "echo"
        }

        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Ok(input)
        }
    }

    struct AddRunnable;

    #[async_trait]
    impl Runnable for AddRunnable {
        fn name(&self) -> &str {
            "add"
        }

        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            let a = input.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = input.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(json!(a + b))
        }
    }

    #[test]
    fn test_convert_to_openai_tool_format() {
        let tool = SimpleTool::new("search", "Search the web", |_: &str| {
            Ok("result".to_string())
        });
        let openai_tool = convert_to_openai_tool(&tool);
        assert_eq!(openai_tool["type"], "function");
        assert_eq!(openai_tool["function"]["name"], "search");
        assert_eq!(openai_tool["function"]["description"], "Search the web");
        assert!(openai_tool["function"]["parameters"].is_object());
    }

    #[test]
    fn test_convert_to_openai_tools_multiple() {
        let tool1 = SimpleTool::new("t1", "Tool 1", |_: &str| Ok("r1".to_string()));
        let tool2 = SimpleTool::new("t2", "Tool 2", |_: &str| Ok("r2".to_string()));
        let tools: Vec<&dyn BaseTool> = vec![&tool1, &tool2];
        let result = convert_to_openai_tools(&tools);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["function"]["name"], "t1");
        assert_eq!(result[1]["function"]["name"], "t2");
    }

    #[tokio::test]
    async fn test_convert_runnable_to_simple_tool() {
        let runnable = Arc::new(EchoRunnable);
        let tool =
            convert_runnable_to_tool(runnable, Some("echo_tool"), Some("Echoes input"), None);

        assert_eq!(tool.name(), "echo_tool");
        assert_eq!(tool.description(), "Echoes input");

        use crate::tools::types::ToolInput;
        let result = tool
            ._run(ToolInput::Text("hello".to_string()))
            .await
            .unwrap();
        match result {
            crate::tools::types::ToolOutput::Content(v) => {
                assert_eq!(v, Value::String("hello".to_string()));
            }
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_convert_runnable_to_structured_tool() {
        let runnable = Arc::new(AddRunnable);
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        });
        let tool = convert_runnable_to_tool(
            runnable,
            Some("add_tool"),
            Some("Add numbers"),
            Some(schema),
        );

        assert_eq!(tool.name(), "add_tool");

        use crate::tools::types::ToolInput;
        let mut args = std::collections::HashMap::new();
        args.insert("a".to_string(), json!(3));
        args.insert("b".to_string(), json!(7));
        let result = tool._run(ToolInput::Structured(args)).await.unwrap();
        match result {
            crate::tools::types::ToolOutput::Content(v) => {
                assert_eq!(v, json!(10.0));
            }
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_convert_runnable_defaults_name_and_description() {
        let runnable = Arc::new(EchoRunnable);
        let tool = convert_runnable_to_tool(runnable, None, None, None);

        assert_eq!(tool.name(), "echo");
        assert_eq!(tool.description(), "Wrapper around echo");
    }
}
