//! State Machine Example
//!
//! Demonstrates building and running a finite state machine for an order
//! fulfillment workflow with guards, actions, and LLM-guided transitions.
//!
//! Run with: `cargo run -p cognis-examples --example state_machine`

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognisgraph::graph::{StateMachineBuilder, StateMachineValidator};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Order Fulfillment State Machine ===\n");

    // Build an order processing state machine using the fluent builder API.
    // States: pending -> confirmed -> shipped -> delivered
    //                 \-> cancelled (guarded by cancel_requested flag)
    let mut machine = StateMachineBuilder::new("pending")
        .state("confirmed")
        .state("shipped")
        .final_state("delivered")
        .final_state("cancelled")
        .transition("pending", "confirmed", "confirm_order")
        .transition("confirmed", "shipped", "ship_order")
        .transition("shipped", "delivered", "deliver_order")
        .guarded_transition("pending", "cancelled", "cancel_order", |ctx| {
            ctx.get("cancel_requested")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .build()?;

    // Validate the machine before running it.
    StateMachineValidator::validate(&machine)
        .map_err(|errors| format!("Validation failed: {:?}", errors))?;
    println!("Machine validated successfully.");

    // --- Happy path: order flows to delivery ---
    let mut ctx = json!({"order_id": "ORD-001", "items": 3});
    let path = machine.run(&mut ctx)?;
    let names: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
    println!("Happy path:        {:?}", names);
    println!("Final state:       {}", machine.current());

    // --- Cancellation path: guard redirects to cancelled ---
    machine.reset();
    let mut cancel_ctx = json!({"order_id": "ORD-002", "cancel_requested": true});
    let cancel_path = machine.run(&mut cancel_ctx)?;
    let cancel_names: Vec<&str> = cancel_path.iter().map(|s| s.as_str()).collect();
    println!("Cancellation path: {:?}", cancel_names);
    println!("Final state:       {}", machine.current());

    // --- LLM-guided transition advice ---
    println!("\n--- LLM Transition Advice ---");
    let model = shared::get_chat_model(vec![
        "The order should transition to 'confirmed' because payment is verified and all items are in stock.".into(),
    ]);
    let messages = vec![
        Message::system("You are a state machine advisor. Given a state and context, suggest the next transition."),
        Message::human("Current state: 'pending', context: {\"order_id\": \"ORD-003\", \"items\": 3, \"payment\": \"verified\"}. What transition should fire next?"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM advice: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
