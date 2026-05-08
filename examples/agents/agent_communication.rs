//! Inter-agent messaging via the V2 MessageBus. Agents communicate
//! through an InMemoryMessageBus by default; swap in your own
//! transport by implementing MessageBus.

use std::sync::Arc;

use cognis::prelude::*;
use cognis::{
    AgentBuilder, AgentMessage, InMemoryMessageBus, MessageBus, MultiAgentOrchestrator, Sequential,
};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let bus: Arc<dyn MessageBus> = Arc::new(InMemoryMessageBus::new());

    let orch = MultiAgentOrchestrator::new(Sequential)
        .with_bus(bus.clone())
        .add(
            "researcher",
            AgentBuilder::new()
                .with_llm(Client::from_env()?)
                .with_system_prompt("Reply with 2 facts about Mars in one sentence.")
                .build()?,
        )
        .add(
            "writer",
            AgentBuilder::new()
                .with_llm(Client::from_env()?)
                .with_system_prompt("Turn the input into a single tweet.")
                .build()?,
        );

    let _ = bus
        .publish(AgentMessage {
            from: "user".into(),
            to: "researcher".into(),
            content: Message::human("kickoff"),
            metadata: serde_json::Value::Null,
            ..Default::default()
        })
        .await;

    let resp = orch.run("Tell me about Mars").await?;
    println!("final: {}", resp.content);
    println!(
        "\nbus traffic for researcher inbox: {:?}",
        bus.drain("researcher").await?.len()
    );
    println!(
        "bus traffic for writer inbox:     {:?}",
        bus.drain("writer").await?.len()
    );
    Ok(())
}
