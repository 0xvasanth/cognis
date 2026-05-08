//! Stateful conversational agent — V2 uses cognis::Buffer for memory.
//! With ConversationMode::Stateful the agent remembers across turns.

use cognis::prelude::*;
use cognis::{AgentBuilder, Buffer};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_memory(Buffer::new().with_system("You are a friendly chatbot."))
        .stateful()
        .build()?;

    for prompt in ["Hi! My name is Sam.", "What's my name?"] {
        let r = agent.run(Message::human(prompt)).await?;
        println!("user: {prompt}");
        println!("ai:   {}\n", r.content);
    }
    Ok(())
}
