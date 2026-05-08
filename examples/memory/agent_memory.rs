//! Agent + Buffer memory across multiple turns. ConversationMode::Stateful
//! persists messages in the agent's memory for the next .run() call.

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
        .with_memory(Buffer::new().with_system("You are a math tutor."))
        .stateful()
        .build()?;

    let r1 = agent
        .run(Message::human("My favorite number is 7."))
        .await?;
    println!("turn 1: {}", r1.content);
    let r2 = agent
        .run(Message::human("What's my favorite number times 3?"))
        .await?;
    println!("turn 2: {}", r2.content);
    Ok(())
}
