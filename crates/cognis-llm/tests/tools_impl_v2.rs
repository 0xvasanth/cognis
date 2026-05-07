//! Verify #[tools_impl] with crate_path = "cognis_llm" works end-to-end.

use std::sync::Arc;

use cognis_llm::error::Result;
use cognis_llm::tools::{ToolInput, ToolOutput};
use cognis_macros::tools_impl;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct AddParams {
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

struct Calculator {
    precision: u32,
}

#[tools_impl(crate_path = "cognis_llm")]
impl Calculator {
    /// Add two numbers (doc-comment description).
    #[tool]
    async fn add(&self, p: AddParams) -> Result<ToolOutput> {
        let factor = 10f64.powi(self.precision as i32);
        let r = ((p.a + p.b) * factor).round() / factor;
        Ok(ToolOutput::Content(serde_json::json!(r)))
    }
}

#[tokio::test]
async fn tools_impl_v2_path_resolves_and_runs() {
    let calc = Arc::new(Calculator { precision: 2 });
    let tools = calc.into_tools();
    assert_eq!(tools.len(), 1);

    let add = &tools[0];
    assert_eq!(add.name(), "add");
    assert_eq!(
        add.description(),
        "Add two numbers (doc-comment description)."
    );

    // Single-arg unwrap: schema has a/b at top level (NOT under "p").
    let schema = add.args_schema().unwrap();
    assert!(schema["properties"]["a"].is_object());
    assert!(schema["properties"]["b"].is_object());
    assert!(schema["properties"]["p"].is_null());

    let mut m = std::collections::HashMap::new();
    m.insert("a".into(), serde_json::json!(1.234));
    m.insert("b".into(), serde_json::json!(2.345));
    let out = add._run(ToolInput::Structured(m)).await.unwrap();
    if let ToolOutput::Content(v) = out {
        assert_eq!(v.as_f64().unwrap(), 3.58); // (1.234+2.345)=3.579 → precision 2 → 3.58
    } else {
        panic!("wrong variant");
    }
}
