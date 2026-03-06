//! Structured Data Extraction Example
//!
//! Demonstrates using StructuredOutputChain to extract typed structured data
//! from unstructured text. Shows the full pipeline: input text -> chat model ->
//! JSON parser -> validated typed output.
//!
//! No API keys required -- uses GenericFakeChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example structured_extraction

use std::sync::Arc;

use serde_json::json;

use rustchain::chains::structured_output::StructuredOutputChain;
use rustchain_core::language_models::GenericFakeChatModel;
use rustchain_core::messages::AIMessage;
use rustchain_core::runnables::Runnable;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Structured Data Extraction Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Define the JSON schema for person information
    // -------------------------------------------------------------------------
    let person_schema = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's full name"
            },
            "age": {
                "type": "integer",
                "description": "The person's age in years"
            },
            "occupation": {
                "type": "string",
                "description": "The person's job or profession"
            }
        },
        "required": ["name", "age", "occupation"]
    });

    println!("Step 1: Defined person schema:");
    println!("  Required fields: name (string), age (integer), occupation (string)\n");

    // -------------------------------------------------------------------------
    // Step 2: Create a GenericFakeChatModel with predefined JSON responses
    // -------------------------------------------------------------------------
    // Each response is a valid JSON object matching the schema.
    // In production, a real LLM would generate these from the input text.

    let model = Arc::new(GenericFakeChatModel::from_messages(vec![
        AIMessage::new(
            r#"{"name": "Alice Johnson", "age": 32, "occupation": "Software Engineer"}"#,
        ),
        AIMessage::new(r#"{"name": "Dr. Robert Chen", "age": 45, "occupation": "Neurosurgeon"}"#),
        AIMessage::new(r#"{"name": "Maria Garcia", "age": 28, "occupation": "Data Scientist"}"#),
    ]));

    println!("Step 2: Created GenericFakeChatModel with 3 predefined responses\n");

    // -------------------------------------------------------------------------
    // Step 3: Build the StructuredOutputChain
    // -------------------------------------------------------------------------
    let chain = StructuredOutputChain::builder()
        .model(model)
        .schema(person_schema.clone())
        .prompt("Extract the person's information from this text: {text}")
        .build();

    println!("Step 3: Built StructuredOutputChain");
    println!("  Chain name: {}", chain.name());
    if let Some(instructions) = chain.format_instructions() {
        println!(
            "  Format instructions preview: {}...",
            &instructions[..instructions.len().min(80)]
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 4: Extract structured data from text inputs
    // -------------------------------------------------------------------------
    println!("Step 4: Extracting structured data\n");

    let texts = vec![
        "Alice Johnson is a 32-year-old software engineer who works at a tech startup in San Francisco.",
        "Dr. Robert Chen, aged 45, is a renowned neurosurgeon at Johns Hopkins Hospital.",
        "Maria Garcia recently turned 28 and has been working as a data scientist for three years.",
    ];

    for (i, text) in texts.iter().enumerate() {
        println!("  Input {}: \"{}\"", i + 1, text);

        let input = json!({ "text": text });
        let result = chain.invoke(input, None).await?;

        // The result is a validated JSON object matching the schema.
        let name = result["name"].as_str().unwrap_or("unknown");
        let age = result["age"].as_i64().unwrap_or(0);
        let occupation = result["occupation"].as_str().unwrap_or("unknown");

        println!("  Extracted:");
        println!("    Name:       {name}");
        println!("    Age:        {age}");
        println!("    Occupation: {occupation}");
        println!("    Raw JSON:   {result}");
        println!();
    }

    // -------------------------------------------------------------------------
    // Step 5: Demonstrate extraction with output key wrapping
    // -------------------------------------------------------------------------
    println!("--- Step 5: Extraction with output key ---\n");

    let wrapped_model = Arc::new(GenericFakeChatModel::from_messages(vec![AIMessage::new(
        r#"{"name": "Eve Park", "age": 38, "occupation": "Architect"}"#,
    )]));

    let wrapped_chain = StructuredOutputChain::builder()
        .model(wrapped_model)
        .schema(person_schema.clone())
        .prompt("Extract: {text}")
        .output_key("person")
        .build();

    let result = wrapped_chain
        .invoke(json!({"text": "Eve Park is a 38-year-old architect"}), None)
        .await?;

    println!("  With output_key=\"person\":");
    println!(
        "    result[\"person\"][\"name\"] = {}",
        result["person"]["name"]
    );
    println!(
        "    result[\"person\"][\"age\"] = {}",
        result["person"]["age"]
    );
    println!(
        "    result[\"person\"][\"occupation\"] = {}",
        result["person"]["occupation"]
    );
    println!();

    // -------------------------------------------------------------------------
    // Step 6: Demonstrate extraction with a more complex schema
    // -------------------------------------------------------------------------
    println!("--- Step 6: Complex schema extraction ---\n");

    let event_schema = json!({
        "type": "object",
        "properties": {
            "event_name": { "type": "string" },
            "date": { "type": "string" },
            "location": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "country": { "type": "string" }
                }
            },
            "attendees": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["event_name", "date"]
    });

    let event_model = Arc::new(GenericFakeChatModel::from_messages(vec![AIMessage::new(
        r#"{"event_name": "RustConf 2025", "date": "2025-09-15", "location": {"city": "Portland", "country": "USA"}, "attendees": ["Alice", "Bob", "Carol"]}"#,
    )]));

    let event_chain = StructuredOutputChain::builder()
        .model(event_model)
        .schema(event_schema)
        .prompt("Extract event details from: {text}")
        .build();

    let event_result = event_chain
        .invoke(
            json!({"text": "RustConf 2025 will be held on September 15, 2025 in Portland, USA. Alice, Bob, and Carol plan to attend."}),
            None,
        )
        .await?;

    println!("  Event: {}", event_result["event_name"]);
    println!("  Date: {}", event_result["date"]);
    println!(
        "  Location: {}, {}",
        event_result["location"]["city"], event_result["location"]["country"]
    );
    println!(
        "  Attendees: {:?}",
        event_result["attendees"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default()
    );

    println!("\nDone!");
    Ok(())
}
