//! Extraction Chain Example
//!
//! Demonstrates ExtractionChain for extracting structured entities from
//! unstructured text using a schema with typed fields.

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::chat_model::BaseChatModel;
use std::sync::Arc;

use cognis::chains::extraction::{
    ExtractionChain, ExtractionSchema, FieldType, SchemaFieldBuilder,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the extraction schema
    let schema = ExtractionSchema::builder()
        .name("Person")
        .description("A person mentioned in the text")
        .field(
            SchemaFieldBuilder::new("name", FieldType::String)
                .description("The person's full name")
                .required(true)
                .build(),
        )
        .field(
            SchemaFieldBuilder::new("age", FieldType::Integer)
                .description("The person's age in years")
                .required(false)
                .build(),
        )
        .field(
            SchemaFieldBuilder::new("occupation", FieldType::String)
                .description("The person's job title or profession")
                .required(true)
                .build(),
        )
        .field(
            SchemaFieldBuilder::new("location", FieldType::String)
                .description("Where the person is based")
                .required(false)
                .build(),
        )
        .build();

    // Create a model with predefined JSON responses
    let model: Arc<dyn BaseChatModel> = shared::get_chat_model(vec![
        r#"[{"name": "Alice Chen", "age": 34, "occupation": "Machine Learning Engineer", "location": "San Francisco"}]"#.into(),
        r#"[{"name": "Dr. James Wilson", "age": 52, "occupation": "Chief Medical Officer", "location": "Boston"}, {"name": "Sarah Park", "age": 29, "occupation": "Research Scientist", "location": "Boston"}]"#.into(),
        r#"[{"name": "Marcus Johnson", "occupation": "Software Architect"}, {"name": "Elena Rodriguez", "age": 41, "occupation": "VP of Engineering", "location": "Austin"}]"#.into(),
    ]);

    let chain = ExtractionChain::builder().llm(model).schema(schema).build();

    // Extract entities from sample texts
    let texts = [
        "Alice Chen is a 34-year-old machine learning engineer based in San Francisco.",
        "Dr. James Wilson, 52, serves as CMO at Boston General. He works with Sarah Park, a 29-year-old research scientist.",
        "The team is led by Marcus Johnson, a software architect, and Elena Rodriguez, 41, VP of Engineering from Austin.",
    ];

    for (i, text) in texts.iter().enumerate() {
        let result = chain.extract(text).await?;
        println!("Input {}: \"{}\"", i + 1, text);
        for entity in &result.entities {
            let name = entity["name"].as_str().unwrap_or("?");
            let occupation = entity["occupation"].as_str().unwrap_or("?");
            let age = entity.get("age").and_then(|v| v.as_i64());
            let location = entity.get("location").and_then(|v| v.as_str());
            println!(
                "  -> {} ({}{}{})",
                name,
                occupation,
                age.map(|a| format!(", age {}", a)).unwrap_or_default(),
                location.map(|l| format!(", {}", l)).unwrap_or_default(),
            );
        }
    }

    Ok(())
}
