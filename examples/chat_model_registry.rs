//! Chat Model Registry Example
//!
//! Demonstrates the chat model registry system for managing model metadata,
//! configurations, capabilities, and selection. This is useful for applications
//! that need to dynamically choose the right model based on requirements such
//! as cost, context window size, or feature support.
//!
//! Features shown:
//! - ModelConfig building with temperature, max_tokens, stop_sequences
//! - ModelConfig merging (overrides)
//! - ModelInfo creation for multiple providers (Anthropic, OpenAI, Google)
//! - ModelRegistry registration and lookup
//! - ModelCapability filtering (Streaming, ToolCalling, Vision, LargeContext)
//! - ModelSelector selecting best model for requirements
//! - ModelSelector selecting cheapest model
//! - TokenUsage tracking
//! - ChatRequest and ChatResponse building
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example chat_model_registry`

use rustchain::chat_models::registry::{
    ChatRequest, ChatResponse, ModelCapability, ModelConfig, ModelInfo, ModelRegistry,
    ModelSelector, TokenUsage,
};
use rustchain_core::messages::Message;

fn main() {
    println!("=== Chat Model Registry Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ModelConfig Building
    // -----------------------------------------------------------------------
    println!("--- 1. ModelConfig Building ---");
    println!("Build generation configs with temperature, max_tokens, stop sequences, and extras.\n");

    let config = ModelConfig::new("claude-sonnet-4-20250514")
        .with_temperature(0.7)
        .with_max_tokens(4096)
        .with_top_p(0.95)
        .with_stop_sequence("END")
        .with_stop_sequence("STOP")
        .with_timeout_ms(30_000)
        .with_extra("system_prompt", serde_json::json!("You are a helpful assistant."));

    println!("Config for '{}':", config.model_name);
    println!("  temperature:    {:?}", config.temperature);
    println!("  max_tokens:     {:?}", config.max_tokens);
    println!("  top_p:          {:?}", config.top_p);
    println!("  stop_sequences: {:?}", config.stop_sequences);
    println!("  timeout_ms:     {:?}", config.timeout_ms);
    println!("  extras:         {} key(s)", config.extra.len());
    println!("\nSerialized config (JSON):");
    println!("  {}", serde_json::to_string_pretty(&config.to_json()).unwrap());

    // -----------------------------------------------------------------------
    // 2. ModelConfig Merging (Overrides)
    // -----------------------------------------------------------------------
    println!("\n--- 2. ModelConfig Merging ---");
    println!("Merge a partial override config into a base config.\n");

    let mut base_config = ModelConfig::new("claude-sonnet-4-20250514")
        .with_temperature(0.5)
        .with_max_tokens(2048)
        .with_timeout_ms(10_000);

    println!("Base config:");
    println!("  model:       {}", base_config.model_name);
    println!("  temperature: {:?}", base_config.temperature);
    println!("  max_tokens:  {:?}", base_config.max_tokens);
    println!("  top_p:       {:?}", base_config.top_p);
    println!("  timeout_ms:  {:?}", base_config.timeout_ms);

    let override_config = ModelConfig::new("")
        .with_temperature(0.9)
        .with_top_p(0.8)
        .with_stop_sequences(vec!["###".into()]);

    println!("\nOverride config:");
    println!("  temperature:    {:?}", override_config.temperature);
    println!("  top_p:          {:?}", override_config.top_p);
    println!("  stop_sequences: {:?}", override_config.stop_sequences);

    base_config.merge(&override_config);

    println!("\nAfter merge:");
    println!("  model:          {} (kept from base, override was empty)", base_config.model_name);
    println!("  temperature:    {:?} (overridden)", base_config.temperature);
    println!("  max_tokens:     {:?} (kept from base)", base_config.max_tokens);
    println!("  top_p:          {:?} (new from override)", base_config.top_p);
    println!("  stop_sequences: {:?} (replaced by override)", base_config.stop_sequences);
    println!("  timeout_ms:     {:?} (kept from base)", base_config.timeout_ms);

    // -----------------------------------------------------------------------
    // 3. ModelInfo Creation for Multiple Providers
    // -----------------------------------------------------------------------
    println!("\n--- 3. ModelInfo for Multiple Providers ---");
    println!("Create model metadata for Anthropic, OpenAI, and Google models.\n");

    let claude_sonnet = ModelInfo::new("anthropic", "claude-sonnet-4-20250514", 200_000)
        .with_streaming(true)
        .with_tools(true)
        .with_vision(true)
        .with_input_cost(0.003)
        .with_output_cost(0.015);

    let claude_haiku = ModelInfo::new("anthropic", "claude-haiku-3-20240307", 200_000)
        .with_streaming(true)
        .with_tools(true)
        .with_input_cost(0.00025)
        .with_output_cost(0.00125);

    let gpt4o = ModelInfo::new("openai", "gpt-4o", 128_000)
        .with_streaming(true)
        .with_tools(true)
        .with_vision(true)
        .with_input_cost(0.005)
        .with_output_cost(0.015);

    let gpt4o_mini = ModelInfo::new("openai", "gpt-4o-mini", 128_000)
        .with_streaming(true)
        .with_tools(true)
        .with_vision(true)
        .with_input_cost(0.00015)
        .with_output_cost(0.0006);

    let gemini_pro = ModelInfo::new("google", "gemini-1.5-pro", 1_000_000)
        .with_streaming(true)
        .with_tools(true)
        .with_vision(true)
        .with_input_cost(0.00125)
        .with_output_cost(0.005);

    let gemini_flash = ModelInfo::new("google", "gemini-1.5-flash", 1_000_000)
        .with_streaming(true)
        .with_tools(true)
        .with_input_cost(0.000075)
        .with_output_cost(0.0003);

    let models = [
        &claude_sonnet,
        &claude_haiku,
        &gpt4o,
        &gpt4o_mini,
        &gemini_pro,
        &gemini_flash,
    ];

    println!(
        "{:<12} {:<30} {:>10} {:>8} {:>6} {:>6}  {:>10}  {:>10}",
        "Provider", "Model ID", "Context", "Stream", "Tools", "Vision", "In Cost", "Out Cost"
    );
    println!("{}", "-".repeat(106));
    for m in &models {
        println!(
            "{:<12} {:<30} {:>10} {:>8} {:>6} {:>6}  {:>10.6}  {:>10.6}",
            m.provider,
            m.model_id,
            m.context_window,
            m.supports_streaming,
            m.supports_tools,
            m.supports_vision,
            m.cost_per_input_token.unwrap_or(0.0),
            m.cost_per_output_token.unwrap_or(0.0),
        );
    }

    // -----------------------------------------------------------------------
    // 4. ModelRegistry Registration and Lookup
    // -----------------------------------------------------------------------
    println!("\n--- 4. ModelRegistry Registration and Lookup ---");
    println!("Register models and look them up by ID or provider.\n");

    let mut registry = ModelRegistry::new();
    registry.register(claude_sonnet);
    registry.register(claude_haiku);
    registry.register(gpt4o);
    registry.register(gpt4o_mini);
    registry.register(gemini_pro);
    registry.register(gemini_flash);

    println!("Registered {} models in the registry", registry.len());

    // Lookup by ID
    if let Some(info) = registry.get("claude-sonnet-4-20250514") {
        println!(
            "\nLookup 'claude-sonnet-4-20250514': provider={}, context={}",
            info.provider, info.context_window
        );
    }

    if let Some(info) = registry.get("gpt-4o-mini") {
        println!(
            "Lookup 'gpt-4o-mini': provider={}, context={}",
            info.provider, info.context_window
        );
    }

    println!(
        "Lookup 'nonexistent': {:?}",
        registry.get("nonexistent").map(|i| &i.model_id)
    );

    // Filter by provider
    let anthropic_models = registry.by_provider("anthropic");
    println!("\nAnthropic models ({}):", anthropic_models.len());
    for m in &anthropic_models {
        println!("  - {}", m.model_id);
    }

    let openai_models = registry.by_provider("openai");
    println!("OpenAI models ({}):", openai_models.len());
    for m in &openai_models {
        println!("  - {}", m.model_id);
    }

    let google_models = registry.by_provider("google");
    println!("Google models ({}):", google_models.len());
    for m in &google_models {
        println!("  - {}", m.model_id);
    }

    // -----------------------------------------------------------------------
    // 5. ModelCapability Filtering
    // -----------------------------------------------------------------------
    println!("\n--- 5. ModelCapability Filtering ---");
    println!("Filter models by capability: Streaming, ToolCalling, Vision, LargeContext.\n");

    let streaming_models = registry.with_capability(&ModelCapability::Streaming);
    println!("Streaming models ({}):", streaming_models.len());
    for m in &streaming_models {
        println!("  - {} ({})", m.model_id, m.provider);
    }

    let tool_models = registry.with_capability(&ModelCapability::ToolCalling);
    println!("\nToolCalling models ({}):", tool_models.len());
    for m in &tool_models {
        println!("  - {} ({})", m.model_id, m.provider);
    }

    let vision_models = registry.with_capability(&ModelCapability::Vision);
    println!("\nVision models ({}):", vision_models.len());
    for m in &vision_models {
        println!("  - {} ({})", m.model_id, m.provider);
    }

    let large_ctx = registry.with_capability(&ModelCapability::LargeContext(500_000));
    println!("\nLargeContext (>= 500K tokens) models ({}):", large_ctx.len());
    for m in &large_ctx {
        println!("  - {} ({}, context={})", m.model_id, m.provider, m.context_window);
    }

    let json_models = registry.with_capability(&ModelCapability::Json);
    println!("\nJSON mode models ({}):", json_models.len());
    for m in &json_models {
        println!("  - {} ({})", m.model_id, m.provider);
    }

    // -----------------------------------------------------------------------
    // 6. ModelSelector — Best Model for Requirements
    // -----------------------------------------------------------------------
    println!("\n--- 6. ModelSelector — Best Model ---");
    println!("Select the first model matching a set of capability requirements.\n");

    let selector = ModelSelector::new(&registry);

    // No requirements — any model
    let any = selector.select(&[]);
    println!(
        "Any model (no requirements): {}",
        any.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    // Vision + ToolCalling + Large context
    let result = selector.select(&[
        ModelCapability::Vision,
        ModelCapability::ToolCalling,
        ModelCapability::LargeContext(200_000),
    ]);
    println!(
        "Vision + Tools + LargeContext(200K): {}",
        result.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    // Vision + LargeContext(500K) — only Google models qualify
    let result = selector.select(&[
        ModelCapability::Vision,
        ModelCapability::LargeContext(500_000),
    ]);
    println!(
        "Vision + LargeContext(500K): {}",
        result.map(|m| m.model_id.as_str()).unwrap_or("none")
    );

    // Impossible requirement
    let result = selector.select(&[ModelCapability::LargeContext(10_000_000)]);
    println!(
        "LargeContext(10M): {}",
        result.map(|m| m.model_id.as_str()).unwrap_or("none (no model qualifies)")
    );

    // -----------------------------------------------------------------------
    // 7. ModelSelector — Cheapest Model
    // -----------------------------------------------------------------------
    println!("\n--- 7. ModelSelector — Cheapest Model ---");
    println!("Select the cheapest model that meets capability requirements.\n");

    let cheapest_any = selector.select_cheapest(&[]);
    if let Some(m) = cheapest_any {
        println!(
            "Cheapest overall: {} (total cost/token: {:.6})",
            m.model_id,
            m.cost_per_input_token.unwrap_or(0.0) + m.cost_per_output_token.unwrap_or(0.0)
        );
    }

    let cheapest_vision = selector.select_cheapest(&[ModelCapability::Vision]);
    if let Some(m) = cheapest_vision {
        println!(
            "Cheapest with Vision: {} (total cost/token: {:.6})",
            m.model_id,
            m.cost_per_input_token.unwrap_or(0.0) + m.cost_per_output_token.unwrap_or(0.0)
        );
    }

    let cheapest_tools_vision = selector.select_cheapest(&[
        ModelCapability::ToolCalling,
        ModelCapability::Vision,
        ModelCapability::Streaming,
    ]);
    if let Some(m) = cheapest_tools_vision {
        println!(
            "Cheapest with Tools + Vision + Streaming: {} (total cost/token: {:.6})",
            m.model_id,
            m.cost_per_input_token.unwrap_or(0.0) + m.cost_per_output_token.unwrap_or(0.0)
        );
    }

    let cheapest_huge = selector.select_cheapest(&[ModelCapability::LargeContext(500_000)]);
    if let Some(m) = cheapest_huge {
        println!(
            "Cheapest with LargeContext(500K): {} (total cost/token: {:.6})",
            m.model_id,
            m.cost_per_input_token.unwrap_or(0.0) + m.cost_per_output_token.unwrap_or(0.0)
        );
    }

    // -----------------------------------------------------------------------
    // 8. TokenUsage Tracking
    // -----------------------------------------------------------------------
    println!("\n--- 8. TokenUsage Tracking ---");
    println!("Track prompt and completion token counts.\n");

    let usage1 = TokenUsage::new(1500, 350, 1850);
    let usage2 = TokenUsage::new(2200, 800, 3000);

    println!("Request 1: {:?}", usage1);
    println!("Request 2: {:?}", usage2);

    // Aggregate usage across requests
    let total_prompt = usage1.prompt_tokens + usage2.prompt_tokens;
    let total_completion = usage1.completion_tokens + usage2.completion_tokens;
    let total = usage1.total_tokens + usage2.total_tokens;
    println!(
        "Aggregated: prompt={}, completion={}, total={}",
        total_prompt, total_completion, total
    );

    // Serialization
    println!("\nTokenUsage as JSON:");
    println!("  {}", serde_json::to_string(&usage1.to_json()).unwrap());

    // Equality check
    let usage_copy = TokenUsage::new(1500, 350, 1850);
    println!(
        "\nusage1 == usage_copy: {} (structural equality)",
        usage1 == usage_copy
    );
    println!("usage1 == usage2: {}", usage1 == usage2);

    // -----------------------------------------------------------------------
    // 9. ChatRequest and ChatResponse Building
    // -----------------------------------------------------------------------
    println!("\n--- 9. ChatRequest and ChatResponse ---");
    println!("Build structured chat requests with messages, config, and metadata.\n");

    let request_config = ModelConfig::new("claude-sonnet-4-20250514")
        .with_temperature(0.7)
        .with_max_tokens(1024);

    let request = ChatRequest::new()
        .with_config(request_config)
        .add_message(Message::system("You are a helpful coding assistant."))
        .add_message(Message::human("Explain the builder pattern in Rust."))
        .with_metadata("trace_id", serde_json::json!("req-abc-123"))
        .with_metadata("user_id", serde_json::json!("user-42"));

    println!("ChatRequest:");
    println!("  model:         {}", request.config.model_name);
    println!("  messages:      {}", request.message_count());
    println!("  temperature:   {:?}", request.config.temperature);
    println!("  max_tokens:    {:?}", request.config.max_tokens);
    println!("  metadata keys: {:?}", request.metadata.keys().collect::<Vec<_>>());

    // Simulate a response
    let usage = TokenUsage::new(45, 250, 295);
    let response = ChatResponse::new(
        "The builder pattern in Rust is a creational design pattern that lets you construct \
         complex objects step by step. It is especially useful when an object has many optional \
         fields or configuration parameters...",
        "claude-sonnet-4-20250514",
        usage,
    )
    .with_finish_reason("stop")
    .with_metadata("latency_ms", serde_json::json!(1234))
    .with_metadata("region", serde_json::json!("us-east-1"));

    println!("\nChatResponse:");
    println!("  model:         {}", response.model);
    println!("  content:       {}...", &response.content[..60]);
    println!("  finish_reason: {:?}", response.finish_reason);
    println!("  usage:         {:?}", response.usage);
    println!("  metadata:      {:?}", response.metadata);

    println!("\nResponse as JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&response.to_json()).unwrap()
    );

    println!("\n=== Chat Model Registry Example Complete ===");
}
