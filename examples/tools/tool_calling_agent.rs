//! What you'll learn:
//!   How to register more than one tool with an agent and let the LLM
//!   pick the right one — `Calculator` for arithmetic, a hand-written
//!   `WordCount` tool for text length.
//!
//! Why this matters:
//!   Single-tool agents are toys. Real agents juggle a handful of
//!   tools — a calculator, a search call, a database query — and the
//!   model decides which to call. The agent loop dispatches on the
//!   tool name in the LLM's reply; you just hand it `Arc<dyn Tool>`s.
//!
//! Scenario:
//!   Two tools sit on the agent: `calculator` for arithmetic and
//!   `word_count` for text length. The user asks "How many words in:
//!   'rust is fast and safe'?" — the model picks `word_count`, not
//!   `calculator`, and answers from the tool's reply.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example tools_calling_agent
//!
//! Sample output (against ollama / llama3.1):
//!   There are 5 words in 'rust is fast and safe'.

use std::sync::Arc;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};
use serde_json::json;

struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn name(&self) -> &str {
        "word_count"
    }
    fn description(&self) -> &str {
        "Counts words in the supplied `text`."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        }))
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let text = input
            .into_json()
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let n = text.split_whitespace().count();
        Ok(ToolOutput::Content(json!({"words": n})))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
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

    let resp = agent
        .run(Message::human(
            "How many words in: 'rust is fast and safe'?",
        ))
        .await?;
    println!("{}", resp.content);
    Ok(())
}
