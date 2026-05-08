//! V2 plugin via FnPlugin: a closure that mutates AgentBuilder before build.

use std::sync::Arc;

use cognis::prelude::*;
use cognis::{AgentBuilder, AgentPlugin, Calculator, FnPlugin};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let math_plugin = FnPlugin(|b: AgentBuilder| {
        b.with_tool(Arc::new(Calculator::new()))
            .with_system_prompt("Use the calculator for any arithmetic.")
    });

    let builder = AgentBuilder::new().with_llm(Client::from_env()?);
    let mut agent = math_plugin.install(builder).build()?;
    let r = agent.run(Message::human("What is 7 * 9?")).await?;
    println!("{}", r.content);
    Ok(())
}
