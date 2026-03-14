//! Chat Model Registry Example
//!
//! Shows how to register models with metadata and costs, then select
//! the best or cheapest model for a set of capability requirements.
//!
//! Run with: `cargo run -p cognis-examples --example chat_model_registry`

#[path = "../shared.rs"]
mod shared;
use cognis::chat_models::registry::{
    ModelCapability, ModelConfig, ModelInfo, ModelRegistry, ModelSelector,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chat Model Registry Example ===\n");

    // --- Register models from different providers ---
    let mut registry = ModelRegistry::new();

    registry.register(
        ModelInfo::new("anthropic", "claude-sonnet-4-20250514", 200_000)
            .with_streaming(true)
            .with_tools(true)
            .with_vision(true)
            .with_input_cost(0.003)
            .with_output_cost(0.015),
    );

    registry.register(
        ModelInfo::new("anthropic", "claude-haiku-3-20240307", 200_000)
            .with_streaming(true)
            .with_tools(true)
            .with_input_cost(0.00025)
            .with_output_cost(0.00125),
    );

    registry.register(
        ModelInfo::new("openai", "gpt-4o", 128_000)
            .with_streaming(true)
            .with_tools(true)
            .with_vision(true)
            .with_input_cost(0.005)
            .with_output_cost(0.015),
    );

    registry.register(
        ModelInfo::new("google", "gemini-1.5-pro", 1_000_000)
            .with_streaming(true)
            .with_tools(true)
            .with_vision(true)
            .with_input_cost(0.00125)
            .with_output_cost(0.005),
    );

    println!("Registered {} models\n", registry.len());

    // --- Select a model based on capabilities ---
    let selector = ModelSelector::new(&registry);

    let requirements = [
        ModelCapability::Vision,
        ModelCapability::ToolCalling,
        ModelCapability::LargeContext(200_000),
    ];

    let best = selector.select(&requirements);
    println!(
        "Best model for Vision + Tools + 200K context: {}",
        best.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    let cheapest = selector.select_cheapest(&requirements);
    println!(
        "Cheapest model for the same requirements: {}",
        cheapest.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    let huge_ctx = selector.select(&[ModelCapability::LargeContext(500_000)]);
    println!(
        "Model with 500K+ context: {}",
        huge_ctx.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    // --- Use the selected model to make a real call ---
    println!();
    let selected = best.unwrap();
    let config = ModelConfig::new(&selected.model_id)
        .with_temperature(0.7)
        .with_max_tokens(1024);
    println!(
        "Calling {} (temp={:?}, max_tokens={:?})",
        config.model_name, config.temperature, config.max_tokens
    );

    let model = shared::get_chat_model(vec![
        "The builder pattern constructs complex objects step by step, \
         separating construction from representation."
            .into(),
    ]);
    let messages = vec![
        Message::system("You are a helpful coding assistant."),
        Message::human("Explain the builder pattern in one sentence."),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("Response: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
