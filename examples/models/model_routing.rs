//! Model Routing Example
//!
//! Demonstrates the model routing system from `cognis::chat_models::routing`.
//! Shows how to define model profiles with capabilities and costs, configure
//! routing strategies, apply routing rules, use fallback chains, and inspect
//! routing metrics.
//!
//! No API keys required -- uses fake/mock models via shared helper.

#[path = "../shared.rs"]
mod shared;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use cognis::chat_models::routing::{
    FallbackChain, ModelCapabilities, ModelCapability, ModelRouter, RoutingContext, RoutingMetrics,
    RoutingModelProfile, RoutingRule, RoutingStrategy,
};
use cognis_core::messages::{HumanMessage, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Routing Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ModelCapabilities — bitflag-style capability sets
    // -----------------------------------------------------------------------
    println!("--- 1. ModelCapabilities ---\n");

    let mut caps = ModelCapabilities::EMPTY;
    println!("Empty capabilities: {:?}", caps);

    caps.insert(ModelCapability::STREAMING);
    caps.insert(ModelCapability::TOOL_CALLING);
    println!(
        "After adding STREAMING + TOOL_CALLING: {:?} (raw bits: {:#010b})",
        caps, caps.0
    );

    println!(
        "  contains STREAMING?      {}",
        caps.contains(ModelCapability::STREAMING)
    );
    println!(
        "  contains VISION?         {}",
        caps.contains(ModelCapability::VISION)
    );

    let required = ModelCapabilities(ModelCapability::STREAMING | ModelCapability::TOOL_CALLING);
    println!(
        "  satisfies STREAMING+TOOL_CALLING? {}",
        caps.satisfies(required)
    );

    let need_vision = ModelCapabilities(ModelCapability::VISION);
    println!("  satisfies VISION?        {}", caps.satisfies(need_vision));

    // -----------------------------------------------------------------------
    // 2. RoutingModelProfile — metadata for routing decisions
    // -----------------------------------------------------------------------
    println!("\n--- 2. RoutingModelProfile ---\n");

    let gpt4_profile = RoutingModelProfile::new("gpt-4o")
        .with_cost(0.005, 0.015)
        .with_latency(300)
        .with_context_length(128_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING
                | ModelCapability::TOOL_CALLING
                | ModelCapability::VISION
                | ModelCapability::LONG_CONTEXT
                | ModelCapability::STRUCTURED_OUTPUT,
        ))
        .with_quality(0.95);

    let claude_profile = RoutingModelProfile::new("claude-sonnet")
        .with_cost(0.003, 0.015)
        .with_latency(250)
        .with_context_length(200_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING
                | ModelCapability::TOOL_CALLING
                | ModelCapability::LONG_CONTEXT,
        ))
        .with_quality(0.92);

    let mini_profile = RoutingModelProfile::new("gpt-4o-mini")
        .with_cost(0.00015, 0.0006)
        .with_latency(100)
        .with_context_length(128_000)
        .with_capabilities(ModelCapabilities(
            ModelCapability::STREAMING | ModelCapability::TOOL_CALLING,
        ))
        .with_quality(0.70);

    println!(
        "Profile: {} (quality={}, cost_in={}, latency={}ms, ctx={})",
        gpt4_profile.name,
        gpt4_profile.quality_score,
        gpt4_profile.cost_per_1k_input_tokens,
        gpt4_profile.avg_latency_ms,
        gpt4_profile.max_context_length
    );
    println!(
        "Profile: {} (quality={}, cost_in={}, latency={}ms, ctx={})",
        claude_profile.name,
        claude_profile.quality_score,
        claude_profile.cost_per_1k_input_tokens,
        claude_profile.avg_latency_ms,
        claude_profile.max_context_length
    );
    println!(
        "Profile: {} (quality={}, cost_in={}, latency={}ms, ctx={})",
        mini_profile.name,
        mini_profile.quality_score,
        mini_profile.cost_per_1k_input_tokens,
        mini_profile.avg_latency_ms,
        mini_profile.max_context_length
    );

    // -----------------------------------------------------------------------
    // 3. RoutingStrategy — different selection strategies
    // -----------------------------------------------------------------------
    println!("\n--- 3. RoutingStrategy ---\n");

    // We use fake models for all three profiles.
    let gpt4_model = shared::get_chat_model(vec!["I am GPT-4o, the high-quality model.".into()]);
    let claude_model =
        shared::get_chat_model(vec!["I am Claude Sonnet, balanced and capable.".into()]);
    let mini_model = shared::get_chat_model(vec!["I am GPT-4o-mini, fast and affordable.".into()]);

    // --- CostOptimized ---
    let cost_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::CostOptimized)
        .build()?;

    let ctx = RoutingContext::default();
    let selected = cost_router.preview_selection(&ctx)?;
    println!(
        "CostOptimized selects:    {} (cheapest input cost)",
        selected
    );

    // --- LatencyOptimized ---
    let latency_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::LatencyOptimized)
        .build()?;

    let selected = latency_router.preview_selection(&ctx)?;
    println!("LatencyOptimized selects: {} (lowest latency)", selected);

    // --- QualityOptimized ---
    let quality_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::QualityOptimized)
        .build()?;

    let selected = quality_router.preview_selection(&ctx)?;
    println!("QualityOptimized selects: {} (highest quality)", selected);

    // -----------------------------------------------------------------------
    // 4. RoutingRule — filtering candidates
    // -----------------------------------------------------------------------
    println!("\n--- 4. RoutingRule ---\n");

    // Built-in rule: require minimum context length.
    let long_ctx_rule = RoutingRule::min_context_length(150_000);
    println!("Rule: {}", long_ctx_rule.name);

    // Built-in rule: require specific capabilities.
    let vision_rule =
        RoutingRule::requires_capabilities(ModelCapabilities(ModelCapability::VISION));
    println!("Rule: {}", vision_rule.name);

    // Built-in rule: max cost per 1K input tokens.
    let budget_rule = RoutingRule::max_input_cost(0.004);
    println!("Rule: {}", budget_rule.name);

    // Built-in rule: context fits estimated tokens.
    let fits_rule = RoutingRule::context_fits();
    println!("Rule: {}", fits_rule.name);

    // Custom rule: only models with quality >= 0.9.
    let quality_rule = RoutingRule::new("min_quality(0.9)", |_ctx, profile| {
        profile.quality_score >= 0.9
    });
    println!("Rule: {}", quality_rule.name);

    // Build a router that requires VISION + quality >= 0.9 using QualityOptimized.
    let filtered_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::QualityOptimized)
        .rule(vision_rule)
        .rule(quality_rule)
        .build()?;

    let selected = filtered_router.preview_selection(&ctx)?;
    println!(
        "\nQualityOptimized + VISION + quality>=0.9 selects: {} (only gpt-4o qualifies)",
        selected
    );

    // Budget-constrained routing: cost optimized with max input cost $0.004.
    let budget_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::QualityOptimized)
        .rule(budget_rule)
        .build()?;

    let selected = budget_router.preview_selection(&ctx)?;
    println!(
        "QualityOptimized + max_cost<=0.004 selects: {} (gpt-4o excluded by cost)",
        selected
    );

    // Context-aware routing with estimated tokens.
    let ctx_with_tokens = RoutingContext {
        estimated_tokens: Some(150_000),
        required_capabilities: ModelCapabilities::EMPTY,
        tags: vec![],
    };

    let ctx_router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::CostOptimized)
        .rule(RoutingRule::context_fits())
        .build()?;

    let selected = ctx_router.preview_selection(&ctx_with_tokens)?;
    println!(
        "CostOptimized + context_fits(150k tokens) selects: {} (only claude has 200k ctx)",
        selected
    );

    // -----------------------------------------------------------------------
    // 5. ModelRouter — route a real request
    // -----------------------------------------------------------------------
    println!("\n--- 5. ModelRouter — Routing a Request ---\n");

    let router = ModelRouter::builder()
        .add_model(gpt4_profile.clone(), Arc::clone(&gpt4_model))
        .add_model(claude_profile.clone(), Arc::clone(&claude_model))
        .add_model(mini_profile.clone(), Arc::clone(&mini_model))
        .strategy(RoutingStrategy::CostOptimized)
        .build()?;

    let messages = vec![Message::Human(HumanMessage::new(
        "What is the meaning of life?",
    ))];

    let (model_name, result) = router
        .route(&messages, None, &RoutingContext::default())
        .await?;
    println!("Routed to: {}", model_name);
    if let Some(gen) = result.generations.first() {
        println!("Response: {}", gen.text);
    }

    // -----------------------------------------------------------------------
    // 6. FallbackChain — cascading through models
    // -----------------------------------------------------------------------
    println!("\n--- 6. FallbackChain ---\n");

    let shared_metrics = Arc::new(RoutingMetrics::new());
    let fallback = FallbackChain::new(Arc::clone(&shared_metrics))
        .add_model("primary", Arc::clone(&gpt4_model))
        .add_model("secondary", Arc::clone(&claude_model))
        .add_model("tertiary", Arc::clone(&mini_model));

    println!("Fallback chain length: {}", fallback.len());
    println!("Is empty: {}", fallback.is_empty());

    let (chosen_name, result) = fallback.generate(&messages, None).await?;
    println!("FallbackChain used: {}", chosen_name);
    if let Some(gen) = result.generations.first() {
        println!("Response: {}", gen.text);
    }

    // -----------------------------------------------------------------------
    // 7. RoutingMetrics — observability
    // -----------------------------------------------------------------------
    println!("\n--- 7. RoutingMetrics ---\n");

    let metrics = router.metrics();
    println!(
        "Total requests: {}",
        metrics.total_requests.load(Ordering::Relaxed)
    );
    println!(
        "Fallback activations: {}",
        metrics.fallback_activations.load(Ordering::Relaxed)
    );

    let counts = metrics.selection_counts();
    println!("Selection counts per model:");
    for (name, count) in &counts {
        println!("  {}: {}", name, count);
    }

    // List all registered profiles.
    println!("\nRegistered profiles in router:");
    for profile in router.profiles() {
        println!(
            "  {} — quality={:.2}, cost_in={:.5}, latency={}ms",
            profile.name,
            profile.quality_score,
            profile.cost_per_1k_input_tokens,
            profile.avg_latency_ms
        );
    }
    println!("Total model count: {}", router.model_count());

    // -----------------------------------------------------------------------
    // 8. Using routing with a real chat model (via shared helper)
    // -----------------------------------------------------------------------
    println!("\n--- 8. Routing with shared::get_chat_model() ---\n");

    let real_model = shared::get_chat_model(vec![
        "42 is the answer to life, the universe, and everything.".into(),
    ]);

    let real_profile = RoutingModelProfile::new("local-model")
        .with_cost(0.0, 0.0)
        .with_latency(50)
        .with_context_length(4096)
        .with_capabilities(ModelCapabilities(ModelCapability::STREAMING))
        .with_quality(0.5);

    let combined_router = ModelRouter::builder()
        .add_model(real_profile, real_model)
        .strategy(RoutingStrategy::QualityOptimized)
        .build()?;

    let (name, result) = combined_router
        .route(&messages, None, &RoutingContext::default())
        .await?;
    println!("Routed to: {}", name);
    if let Some(gen) = result.generations.first() {
        println!("Response: {}", gen.text);
    }

    println!("\nDone!");
    Ok(())
}
