//! Structured Data Extraction Example
//!
//! Demonstrates StructuredOutputChain for extracting typed structured data
//! from unstructured text: input text -> chat model -> JSON parser -> typed output.

#[path = "../shared.rs"]
mod shared;

use serde_json::json;

use cognis::chains::structured_output::StructuredOutputChain;
use cognis_core::runnables::Runnable;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let person_schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "The person's full name" },
            "age": { "type": "integer", "description": "The person's age in years" },
            "occupation": { "type": "string", "description": "The person's job or profession" }
        },
        "required": ["name", "age", "occupation"]
    });

    // 1. Basic extraction
    let model = shared::get_chat_model(vec![
        r#"{"name": "Alice Johnson", "age": 32, "occupation": "Software Engineer"}"#.into(),
        r#"{"name": "Dr. Robert Chen", "age": 45, "occupation": "Neurosurgeon"}"#.into(),
        r#"{"name": "Maria Garcia", "age": 28, "occupation": "Data Scientist"}"#.into(),
    ]);

    let chain = StructuredOutputChain::builder()
        .model(model)
        .schema(person_schema.clone())
        .prompt("Extract the person's information from this text: {text}")
        .build();

    let texts = [
        "Alice Johnson is a 32-year-old software engineer at a SF startup.",
        "Dr. Robert Chen, aged 45, is a neurosurgeon at Johns Hopkins.",
        "Maria Garcia, 28, has been a data scientist for three years.",
    ];

    for text in &texts {
        let result = chain.invoke(json!({ "text": text }), None).await?;
        println!(
            "{} (age {}, {})",
            result["name"].as_str().unwrap_or("?"),
            result["age"],
            result["occupation"].as_str().unwrap_or("?"),
        );
    }

    // 2. Extraction with output key wrapping
    let model2 = shared::get_chat_model(vec![
        r#"{"name": "Eve Park", "age": 38, "occupation": "Architect"}"#.into(),
    ]);
    let wrapped_chain = StructuredOutputChain::builder()
        .model(model2)
        .schema(person_schema.clone())
        .prompt("Extract: {text}")
        .output_key("person")
        .build();

    let result = wrapped_chain
        .invoke(json!({"text": "Eve Park is a 38-year-old architect"}), None)
        .await?;
    println!(
        "Wrapped: {} (age {}, {})",
        result["person"]["name"], result["person"]["age"], result["person"]["occupation"]
    );

    // 3. Complex schema extraction (event)
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
            "attendees": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["event_name", "date"]
    });

    let event_model = shared::get_chat_model(vec![
        r#"{"event_name": "RustConf 2025", "date": "2025-09-15", "location": {"city": "Portland", "country": "USA"}, "attendees": ["Alice", "Bob", "Carol"]}"#.into(),
    ]);

    let event_chain = StructuredOutputChain::builder()
        .model(event_model)
        .schema(event_schema)
        .prompt("Extract event details from: {text}")
        .build();

    let result = event_chain
        .invoke(json!({"text": "RustConf 2025 on Sept 15 in Portland, USA. Alice, Bob, Carol attending."}), None)
        .await?;

    let attendees = result["attendees"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();
    println!(
        "Event: {} on {} in {}, {} — attendees: {:?}",
        result["event_name"],
        result["date"],
        result["location"]["city"],
        result["location"]["country"],
        attendees,
    );

    Ok(())
}
