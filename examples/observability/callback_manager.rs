//! Callback Manager Example
//!
//! Shows how to set up logging and metrics callbacks, then observe them
//! fire during an LLM call.
//!
//! Run with: `cargo run -p cognis-examples --example callback_manager`

#[path = "../shared.rs"]
mod shared;
use cognis::callbacks::manager::{
    CallbackDataBuilder, CallbackHandler, CallbackManager, CallbackPhase, CallbackScope,
    ConsoleCallbackHandler, MetricsCallbackHandler,
};
use cognis_core::messages::Message;
use serde_json::json;
use std::sync::Arc;

/// Wrapper to share a handler while giving ownership to the CallbackManager.
struct SharedHandler<H: CallbackHandler>(Arc<H>);
impl<H: CallbackHandler> CallbackHandler for SharedHandler<H> {
    fn on_event(&self, data: &cognis::callbacks::manager::CallbackData) {
        self.0.on_event(data);
    }
    fn name(&self) -> &str {
        self.0.name()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Callback Manager Example ===\n");

    // --- Set up callback handlers ---
    let console = Arc::new(ConsoleCallbackHandler::new());
    let metrics = Arc::new(MetricsCallbackHandler::new());

    let manager = CallbackManager::new();
    manager.add_handler(Box::new(SharedHandler(console.clone())));
    manager.add_handler(Box::new(SharedHandler(metrics.clone())));

    // --- Use CallbackScope for automatic start/end lifecycle ---
    println!("--- Running a chain with callback scope ---\n");
    {
        let _scope = CallbackScope::new(
            &manager,
            CallbackPhase::ChainStart,
            CallbackPhase::ChainEnd,
            CallbackManager::new_run_id(),
            json!({"chain": "qa_chain"}),
        );

        // Simulate an LLM call inside the chain
        let run_id = CallbackManager::new_run_id();
        manager.emit_phase(
            CallbackPhase::LlmStart,
            &run_id,
            json!({"model": "llama3.2"}),
        );

        let model = shared::get_chat_model(vec![
            "Callbacks provide observability into LLM execution — tracking latency, errors, and usage.".into(),
        ]);
        let messages = vec![Message::human(
            "Why are callbacks important in LLM frameworks?",
        )];
        let result = model._generate(&messages, None).await?;

        if let Some(gen) = result.generations.first() {
            let text = gen.message.content().text();
            manager.emit_phase(CallbackPhase::LlmEnd, &run_id, json!({"response": text}));
            println!("LLM response: {}\n", text);
        }
    } // ChainEnd emitted automatically when scope drops

    // --- Inspect collected logs ---
    println!("--- Callback logs ---");
    for log in &console.logs() {
        println!("  {}", log);
    }

    // --- Inspect collected metrics ---
    println!("\n--- Metrics summary ---");
    println!("  Total events: {}", metrics.total_events());
    for (phase, count) in metrics.events_by_phase() {
        println!("  {}: {} event(s)", phase, count);
    }

    // --- Demonstrate scope failure path ---
    println!("\n--- Simulating a failed tool call ---");
    console.clear();
    {
        let scope = CallbackScope::new(
            &manager,
            CallbackPhase::ToolStart,
            CallbackPhase::ToolEnd,
            CallbackManager::new_run_id(),
            json!({"tool": "calculator"}),
        );
        scope.fail("division by zero".into());
    } // ToolError emitted instead of ToolEnd

    for log in &console.logs() {
        println!("  {}", log);
    }

    println!("\n=== Callback Manager Example Complete ===");
    Ok(())
}
