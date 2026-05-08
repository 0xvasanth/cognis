//! What you'll learn:
//!   How `SchemaBasedTool` derives the JSON schema the LLM sees from
//!   a typed `Params` struct, then watch the agent loop dispatch the
//!   tool twice with different arguments — once per city — when the
//!   user asks a multi-target question.
//!
//! Why this matters:
//!   Hand-writing tool schemas is a top source of agent bugs — typos
//!   in the JSON, drift between the schema and the function body, no
//!   compile-time checks. Deriving the schema from a Rust struct is
//!   the cleanest fix, and seeing the agent fan-out across multiple
//!   tool calls in a single user turn is what tool-using agents are
//!   *for*.
//!
//! Scenario:
//!   The user asks "what's the weather in Tokyo and Berlin?". The
//!   agent picks the `get_weather` tool, calls it twice (once per
//!   city) inside a single turn, observes both replies, and writes a
//!   summary answer.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example tools_schema
//!
//! Sample output (against ollama / llama3.1):
//!   The current weather is clear in both Tokyo with a temperature of 18°C and Berlin.
//!   (messages exchanged: 3)

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::tools::SchemaBasedTool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City name to look up (e.g. "Tokyo").
    city: String,
    /// Either "c" or "f". Defaults to "c".
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
        "Look up the current weather for a single city. Call once per city."
    }
    async fn execute_typed(&self, args: WeatherArgs) -> Result<Value> {
        // Stand-in for a real API: deterministic per-city values so the
        // demo prints something sensible without network calls.
        let unit = args.unit.unwrap_or_else(|| "c".into());
        let temp = match args.city.to_lowercase().as_str() {
            "tokyo" => 22,
            "berlin" => 14,
            "lagos" => 31,
            _ => 18,
        };
        Ok(json!({"city": args.city, "temp": temp, "unit": unit, "conditions": "clear"}))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_tool(Arc::new(WeatherTool))
        .with_system_prompt(
            "You are a weather assistant. When the user asks about \
             multiple cities, call `get_weather` once per city and \
             then summarise the results in one short sentence.",
        )
        .with_max_iterations(6)
        .build()?;

    let resp = agent
        .run(Message::human("What's the weather in Tokyo and Berlin?"))
        .await?;
    println!("{}", resp.content);
    println!("(messages exchanged: {})", resp.messages.len());
    Ok(())
}
