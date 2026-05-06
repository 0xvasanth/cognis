//! Verify #[tool] with crate_path = "cognis2_llm" works end-to-end.

use cognis2_llm::error::Result;
use cognis2_llm::tools::{BaseTool, ToolInput, ToolOutput};
use cognis_macros::tool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct AddArgs {
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

#[tool(
    name = "add",
    description = "Add two numbers",
    crate_path = "cognis2_llm"
)]
async fn add(args: AddArgs) -> Result<ToolOutput> {
    Ok(ToolOutput::Content(serde_json::json!(args.a + args.b)))
}

// Note: v1 #[cognis::tool] one-arg-per-fn-param semantics still apply here —
// the v2 single-struct unwrap is only in #[tools_impl]. So `args` will appear
// as a top-level "args" key in the schema. That's fine for a regression test.

#[tokio::test]
async fn tool_v2_path_resolves_and_runs() {
    let t = Add;
    assert_eq!(t.name(), "add");
    assert_eq!(t.description(), "Add two numbers");

    // The free-fn #[tool] macro uses schemars::schema_for! (not our
    // schema_for_tool), so the schema may include $ref for nested types.
    // The key invariant is that the schema is present and the tool runs.
    let schema = t.args_schema().expect("schema present");
    assert!(schema.is_object(), "schema must be a JSON object");

    let mut m = std::collections::HashMap::new();
    m.insert("args".into(), serde_json::json!({"a": 2.0, "b": 3.0}));
    let out = t._run(ToolInput::Structured(m)).await.unwrap();
    if let ToolOutput::Content(v) = out {
        assert_eq!(v.as_f64().unwrap(), 5.0);
    } else {
        panic!("wrong variant");
    }
}
