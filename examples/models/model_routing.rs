//! What you'll learn:
//!   How `RoutingProvider` dispatches each call to one of N underlying
//!   providers based on a closure over the messages — so short
//!   greetings hit a cheap model and long technical questions hit a
//!   bigger one, without your agent code ever knowing.
//!
//! Why this matters:
//!   In production you almost always want a "use the cheap model for
//!   simple prompts, the big model for hard ones" rule, or "fall
//!   back to a different provider when one is rate-limited".
//!   `RoutingProvider` is the building block — your agent stays
//!   identical regardless of which inner provider was selected.
//!
//! Scenario:
//!   A chat product where greetings ("hi", "thanks") and quick
//!   acknowledgements should be cheap, but real questions (more than
//!   ~80 chars, with a question mark) deserve the bigger model.
//!   We wire that rule once and watch two different prompts land on
//!   different backends.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example models_routing
//!
//! Sample output (against ollama / llama3.1):
//!   greeting   -> served by model: llama3.1
//!   hard query -> served by model: llama3.1

use std::sync::Arc;

use cognis::prelude::*;
use cognis_llm::provider::LLMProvider;
use cognis_llm::{ProviderRoute, RoutingProvider};

#[tokio::main]
async fn main() -> Result<()> {
    // In real code you'd build two distinct providers — e.g. a small
    // local Ollama for greetings and a hosted gpt-4o-class model for
    // hard questions. Here we use the same default twice so the demo
    // runs offline; the *routing decision* is what matters.
    let small: Arc<dyn LLMProvider> = Client::from_env()?.provider().clone();
    let big: Arc<dyn LLMProvider> = Client::from_env()?.provider().clone();

    // The predicate runs on every call. Pick whatever signal matters:
    // length, a classifier output, presence of a tool, user tier.
    let is_complex = |msgs: &[Message], _: &_| -> bool {
        let total: usize = msgs.iter().map(|m| m.content().len()).sum();
        let has_question = msgs.iter().any(|m| m.content().contains('?'));
        total > 80 && has_question
    };

    let router =
        RoutingProvider::new("router", small).route(ProviderRoute::new("complex", big, is_complex));

    // Short greeting -> small/default model.
    let short = router
        .chat_completion(vec![Message::human("thanks!")], Default::default())
        .await?;
    // Real technical question -> "complex" route.
    let long = router
        .chat_completion(
            vec![Message::human(
                "I'm seeing intermittent panics in my tokio runtime when \
                 spawning more than 8 tasks concurrently — what's the \
                 most likely cause?",
            )],
            Default::default(),
        )
        .await?;

    println!("greeting   -> served by model: {}", short.model);
    println!("hard query -> served by model: {}", long.model);
    Ok(())
}
