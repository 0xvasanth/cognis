//! V2 SchemaBasedTool — derives JSON schema from a typed `Params`
//! struct. Implementer only writes `execute_typed`; the blanket impl
//! handles serde and the Tool trait surface.

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::tools::{SchemaBasedTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City to look up.
    city: String,
    /// Either "c" or "f" — defaults to "c".
    #[serde(default)]
    unit: Option<String>,
}

struct WeatherTool;

#[async_trait]
impl SchemaBasedTool for WeatherTool {
    type Params = WeatherArgs;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Fetches current weather for a city."
    }
    async fn execute_typed(&self, args: WeatherArgs) -> Result<Value> {
        let unit = args.unit.unwrap_or_else(|| "c".into());
        Ok(json!({"city": args.city, "temp": 21, "unit": unit}))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let tool = WeatherTool;
    println!("name: {}", Tool::name(&tool));
    println!("schema:\n{:#}", Tool::args_schema(&tool).unwrap());
    let out = tool
        .execute_typed(WeatherArgs {
            city: "Paris".into(),
            unit: Some("c".into()),
        })
        .await?;
    println!("output: {out}");
    Ok(())
}
