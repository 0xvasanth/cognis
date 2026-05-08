//! Stateful Tool — Translator holds a default target language in its
//! receiver, so per-call args stay minimal.

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::tools::{SchemaBasedTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TranslateArgs {
    text: String,
    target: Option<String>,
}

struct Translator { default_target: String }

#[async_trait]
impl SchemaBasedTool for Translator {
    type Params = TranslateArgs;
    type Output = Value;
    fn name(&self) -> &str { "translate" }
    fn description(&self) -> &str { "Translate text into a target language." }
    async fn execute_typed(&self, args: TranslateArgs) -> Result<Value> {
        let target = args.target.unwrap_or_else(|| self.default_target.clone());
        Ok(json!({
            "original": args.text,
            "target": target,
            "translated": format!("[{target}] (stubbed translation)"),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let t = Translator { default_target: "fr".into() };
    println!("schema:\n{:#}", Tool::args_schema(&t).unwrap());

    let out = t.execute_typed(TranslateArgs { text: "hello".into(), target: None }).await?;
    println!("default → {out}");

    let out = t.execute_typed(TranslateArgs { text: "hello".into(), target: Some("es".into()) }).await?;
    println!("override → {out}");
    Ok(())
}
