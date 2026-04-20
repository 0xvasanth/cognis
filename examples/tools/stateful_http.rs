//! `#[cognis::tool]` on an `impl` block — a stateful tool that keeps
//! configuration (a default target language) in its receiver so the
//! per-call args can be minimal.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example tool_stateful_http -p cognis-examples
//! ```

use cognis_core::error::Result;
use cognis_core::tool;
use cognis_core::tools::{BaseTool, ToolInput, ToolOutput};
use serde_json::json;
use std::collections::HashMap;

pub struct Translator {
    default_target: String,
}

impl Translator {
    /// Construct with a fallback target language used when callers omit
    /// the `target` argument.
    pub fn new(default_target: impl Into<String>) -> Self {
        Self {
            default_target: default_target.into(),
        }
    }
}

#[tool(name = "translate")]
impl Translator {
    /// Translate text to a target language. Mock implementation — a real
    /// one would call an HTTP service using credentials on `self`.
    async fn translate(
        &self,
        /// Source text.
        #[schema(length(min = 1))]
        text: String,
        /// Target language code. Defaults to the translator's configured
        /// target when omitted.
        #[schema(enum_values("en", "fr", "es", "de", "ja"))]
        target: Option<String>,
    ) -> Result<ToolOutput> {
        let target = target.unwrap_or_else(|| self.default_target.clone());
        Ok(ToolOutput::Content(json!({
            "text": text,
            "target": target,
            "translated": format!("[{target}] {text}"),
        })))
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let tool = Translator::new("fr");

    println!("tool name: {}", tool.name());
    println!("tool description: {}", tool.description());
    println!(
        "schema:\n{}",
        serde_json::to_string_pretty(&tool.args_schema().unwrap())?
    );

    // Call with target explicitly set.
    let mut args = HashMap::new();
    args.insert("text".to_string(), json!("hello, world"));
    args.insert("target".to_string(), json!("es"));
    if let ToolOutput::Content(v) = tool._run(ToolInput::Structured(args)).await? {
        println!("\nexplicit target:\n{}", serde_json::to_string_pretty(&v)?);
    }

    // Call with target omitted — falls back to the configured default.
    let mut args = HashMap::new();
    args.insert("text".to_string(), json!("hello, world"));
    if let ToolOutput::Content(v) = tool._run(ToolInput::Structured(args)).await? {
        println!(
            "\ndefault target (from receiver state):\n{}",
            serde_json::to_string_pretty(&v)?
        );
    }

    Ok(())
}
