//! Model Routing Example
//!
//! Configures three model profiles with different cost/latency/quality tradeoffs,
//! then routes a request using a cost-optimized strategy with a budget constraint.

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use cognis::chat_models::routing::{
    ModelCapabilities, ModelCapability, ModelRouter, RoutingContext, RoutingModelProfile,
    RoutingRule, RoutingStrategy,
};
use cognis_core::messages::{HumanMessage, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Routing Example ===\n");

    // -- Define model profiles with cost, latency, and capability metadata --

    let gpt4 = RoutingModelProfile::new("gpt-4o")
        .with_cost(0.005, 0.015)
        .with_latency(300)
        .with_context_length(128_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING
                | ModelCapability::TOOL_CALLING
                | ModelCapability::VISION
                | ModelCapability::STRUCTURED_OUTPUT,
        ))
        .with_quality(0.95);

    let claude = RoutingModelProfile::new("claude-sonnet")
        .with_cost(0.003, 0.015)
        .with_latency(250)
        .with_context_length(200_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING
                | ModelCapability::TOOL_CALLING
                | ModelCapability::LONG_CONTEXT,
        ))
        .with_quality(0.92);

    let mini = RoutingModelProfile::new("gpt-4o-mini")
        .with_cost(0.00015, 0.0006)
        .with_latency(100)
        .with_context_length(128_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING | ModelCapability::TOOL_CALLING,
        ))
        .with_quality(0.70);

    // -- Create fake models via shared helper --

    let gpt4_model = shared::get_chat_model(vec!["I am GPT-4o.".into()]);
    let claude_model = shared::get_chat_model(vec!["I am Claude Sonnet.".into()]);
    let mini_model = shared::get_chat_model(vec!["I am GPT-4o-mini.".into()]);

    // -- Build a cost-optimized router with a budget constraint --
    // The budget rule excludes models costing more than $0.004/1K input tokens,
    // then the strategy picks the cheapest remaining option.

    let router = ModelRouter::builder()
        .add_model(gpt4, Arc::clone(&gpt4_model))
        .add_model(claude, Arc::clone(&claude_model))
        .add_model(mini, Arc::clone(&mini_model))
        .strategy(RoutingStrategy::CostOptimized)
        .rule(RoutingRule::max_input_cost(0.004))
        .build()?;

    // Preview which model the router would select.
    let ctx = RoutingContext::default();
    let selected = router.preview_selection(&ctx)?;
    println!("Budget-constrained cost-optimized selection: {selected}");

    // Route an actual request through the selected model.
    let messages = vec![Message::Human(HumanMessage::new(
        "What is the meaning of life?",
    ))];
    let (model_name, result) = router.route(&messages, None, &ctx).await?;

    println!("Routed to: {model_name}");
    if let Some(gen) = result.generations.first() {
        println!("Response:  {}", gen.text);
    }

    Ok(())
}
