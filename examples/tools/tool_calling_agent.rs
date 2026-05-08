//! Agent with two tools: Calculator + a hand-written WordCount tool.
//! Demonstrates tool registration and the agent picking between them.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis::{AgentBuilder, Calculator};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};
use cognis_llm::Client;
use serde_json::json;

struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn name(&self) -> &str { "word_count" }
    fn description(&self) -> &str { "Counts words in the supplied `text`." }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }))
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let text = input.into_json()
            .get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let n = text.split_whitespace().count();
        Ok(ToolOutput::Content(json!({"words": n})))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_tool(Arc::new(Calculator::new()))
        .with_tool(Arc::new(WordCount))
        .with_system_prompt(
            "You have two tools: `calculator` for arithmetic and \
             `word_count` to count words. Pick the right one.",
        )
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("How many words in: 'rust is fast and safe'?")).await?;
    println!("{}", resp.content);
    Ok(())
}
