//! Regression tests proving #[tool(crate_path = "cognis_core", ...)] is
//! semantically identical to #[tool(...)] without crate_path. v2 retargets
//! by passing crate_path = "cognis2_core" instead — verified in Plan #2.

use cognis_core::error::Result;
use cognis_core::tool;
use cognis_core::tools::BaseTool;
use cognis_core::tools::ToolInput;
use cognis_core::tools::ToolOutput;
use serde_json::json;
use std::collections::HashMap;

#[tool(
    name = "add_explicit",
    description = "Add two numbers",
    crate_path = "cognis_core"
)]
async fn add_explicit(
    /// First number
    a: f64,
    /// Second number
    b: f64,
) -> Result<ToolOutput> {
    Ok(ToolOutput::Content(json!(a + b)))
}

#[tokio::test]
async fn explicit_crate_path_works() {
    let t = AddExplicit;
    assert_eq!(t.name(), "add_explicit");
    assert_eq!(t.description(), "Add two numbers");

    let schema = t.args_schema().expect("schema present");
    assert_eq!(schema["properties"]["a"]["description"], "First number");
    assert_eq!(schema["properties"]["b"]["description"], "Second number");

    let mut m = HashMap::new();
    m.insert("a".to_string(), json!(2.0));
    m.insert("b".to_string(), json!(3.0));
    let out = t._run(ToolInput::Structured(m)).await.unwrap();
    let val = match out {
        ToolOutput::Content(v) => v,
        _ => panic!("wrong variant"),
    };
    assert_eq!(val.as_f64().unwrap(), 5.0);
}

#[tokio::test]
async fn schema_has_no_dollar_ref() {
    // OpenAI strict mode requires no $ref / $defs in schemas.
    let t = AddExplicit;
    let schema = t.args_schema().expect("schema present");
    let s = schema.to_string();
    assert!(!s.contains("$ref"), "schema must not contain $ref: {s}");
    assert!(!s.contains("$defs"), "schema must not contain $defs: {s}");
}
