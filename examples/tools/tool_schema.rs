//! Tool Schema Example
//!
//! Defines a weather tool with a JSON schema, validates inputs, and
//! asks an LLM to select the tool with appropriate parameters.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example tool_schema`

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognis_core::tools::schema::{
    PropertySchema, SchemaRegistry, SchemaValidator, ToolSchemaGenerator,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool Schema Example ===\n");

    // 1. Define a weather tool using the declarative builder
    let weather_tool = ToolSchemaGenerator::new("get_weather", "Get current weather for a city")
        .add_string_param("city", "City name", true)
        .add_enum_param(
            "units",
            "Temperature units",
            vec!["celsius".into(), "fahrenheit".into()],
            false,
        )
        .add_boolean_param("include_forecast", "Include 5-day forecast", false)
        .add_integer_param("forecast_days", "Number of forecast days (1-7)", false)
        .add_array_param(
            "fields",
            "Specific weather fields to return",
            PropertySchema::string(),
            false,
        )
        .build();

    println!("Tool schema (OpenAI format):");
    println!(
        "{}\n",
        serde_json::to_string_pretty(&weather_tool.to_json())?
    );

    // 2. Validate inputs against the schema
    let valid_input = json!({
        "city": "Tokyo",
        "units": "celsius",
        "include_forecast": true,
        "forecast_days": 3
    });
    match weather_tool.validate_input(&valid_input) {
        Ok(()) => println!("Valid input:          PASS"),
        Err(errs) => println!("Valid input:          FAIL {:?}", errs),
    }

    let missing_required = json!({ "units": "celsius" });
    match weather_tool.validate_input(&missing_required) {
        Ok(()) => println!("Missing 'city':       PASS (unexpected)"),
        Err(errs) => println!("Missing 'city':       FAIL — {:?}", errs),
    }

    let bad_enum = json!({ "city": "Tokyo", "units": "kelvin" });
    match weather_tool.validate_input(&bad_enum) {
        Ok(()) => println!("Invalid enum value:   PASS (unexpected)"),
        Err(errs) => println!("Invalid enum value:   FAIL — {:?}", errs),
    }

    // 3. Register the tool and export the registry
    let mut registry = SchemaRegistry::new();
    registry.register(weather_tool);
    println!("\nRegistry contains {} tool(s)", registry.len());

    // 4. Ask an LLM to pick the right tool
    println!("\n--- LLM Tool Selection ---");
    let model = shared::get_chat_model(vec![
        "I would use `get_weather` with: {\"city\": \"Tokyo\", \"units\": \"celsius\", \"include_forecast\": true, \"forecast_days\": 3}".into(),
    ]);

    let tool_list = serde_json::to_string_pretty(&registry.to_json())?;
    let messages = vec![
        Message::system(&format!(
            "You have these tools:\n{}\n\nSay which tool and parameters you'd use.",
            tool_list
        )),
        Message::human("What's the weather in Tokyo for the next 3 days in Celsius?"),
    ];

    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM response: {}", gen.message.content().text());
    }

    // 5. Validate the call through the registry
    let call_args = json!({ "city": "Tokyo", "units": "celsius", "include_forecast": true, "forecast_days": 3 });
    match registry.validate_call("get_weather", &call_args) {
        Ok(()) => println!("\nRegistry validation: PASS"),
        Err(e) => println!("\nRegistry validation: FAIL — {}", e),
    }

    println!("\n=== Done ===");
    Ok(())
}
