//! Integration tests for #[tools_impl] using v1 cognis_core paths.
//! Plan #2 will add v2 equivalents pointing at cognis2_core.

use std::collections::HashMap;
use std::sync::Arc;

use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_macros::tools_impl;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct AddParams {
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct MulParams {
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

struct Calculator {
    precision: u32,
}

impl Calculator {
    fn round(&self, x: f64) -> f64 {
        let factor = 10f64.powi(self.precision as i32);
        (x * factor).round() / factor
    }
}

#[tools_impl]
impl Calculator {
    /// Add two numbers (doc-comment description).
    #[tool]
    async fn add(&self, p: AddParams) -> cognis_core::error::Result<ToolOutput> {
        Ok(ToolOutput::Content(serde_json::json!(self.round(p.a + p.b))))
    }

    #[tool(description = "Multiply two numbers")]
    async fn mul(&self, p: MulParams) -> cognis_core::error::Result<ToolOutput> {
        Ok(ToolOutput::Content(serde_json::json!(self.round(p.a * p.b))))
    }

    fn helper(&self) -> u32 {
        // No #[tool] — passes through untouched.
        self.precision
    }
}

#[tokio::test]
async fn into_tools_yields_two_tools() {
    let calc = Arc::new(Calculator { precision: 2 });
    let tools = calc.clone().into_tools();
    assert_eq!(tools.len(), 2);

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"mul"));

    let add_tool = tools.iter().find(|t| t.name() == "add").unwrap();
    assert_eq!(add_tool.description(), "Add two numbers (doc-comment description).");
    let mul_tool = tools.iter().find(|t| t.name() == "mul").unwrap();
    assert_eq!(mul_tool.description(), "Multiply two numbers");
}

#[tokio::test]
async fn schema_unwraps_single_params_struct() {
    // The LLM should see AddParams's fields directly at the top level —
    // not wrapped under a `"p"` key as the v1 #[cognis::tool] would do.
    let calc = Arc::new(Calculator { precision: 0 });
    let tools = calc.into_tools();
    let add = tools.iter().find(|t| t.name() == "add").unwrap();

    let schema = add.args_schema().expect("schema present");
    let props = schema["properties"].as_object().expect("properties");
    assert!(props.contains_key("a"), "expected `a` at top level: {schema}");
    assert!(props.contains_key("b"), "expected `b` at top level: {schema}");
    assert!(!props.contains_key("p"), "did not expect `p` wrapper: {schema}");
}

#[tokio::test]
async fn add_tool_uses_shared_state() {
    let calc = Arc::new(Calculator { precision: 1 });
    let tools = calc.clone().into_tools();
    let add = tools.iter().find(|t| t.name() == "add").unwrap();

    let map: HashMap<String, serde_json::Value> =
        serde_json::from_value(serde_json::json!({"a": 1.234, "b": 2.345})).unwrap();
    let out = add
        ._run(ToolInput::Structured(map))
        .await
        .unwrap();
    let val = match out { ToolOutput::Content(v) => v, _ => panic!("wrong variant") };
    // 1.234 + 2.345 = 3.579, rounded to precision 1 = 3.6.
    assert_eq!(val.as_f64().unwrap(), 3.6);
}

#[tokio::test]
async fn mul_tool_uses_shared_state() {
    let calc = Arc::new(Calculator { precision: 0 });
    let tools = calc.clone().into_tools();
    let mul = tools.iter().find(|t| t.name() == "mul").unwrap();

    let map: HashMap<String, serde_json::Value> =
        serde_json::from_value(serde_json::json!({"a": 2.4, "b": 3.0})).unwrap();
    let out = mul
        ._run(ToolInput::Structured(map))
        .await
        .unwrap();
    let val = match out { ToolOutput::Content(v) => v, _ => panic!("wrong variant") };
    // 2.4 * 3.0 = 7.2, rounded to precision 0 = 7.0.
    assert_eq!(val.as_f64().unwrap(), 7.0);
}

#[test]
fn helper_method_passes_through() {
    let calc = Calculator { precision: 5 };
    assert_eq!(calc.helper(), 5);
}
