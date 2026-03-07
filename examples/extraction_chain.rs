//! Extraction Chain Example
//!
//! Demonstrates the ExtractionChain for extracting structured entities from
//! unstructured text:
//! - Defining an ExtractionSchema with typed fields
//! - Using FakeListChatModel to return structured JSON responses
//! - Extracting entities from sample text
//! - Displaying the extracted results
//!
//! No API keys required -- uses FakeListChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example extraction_chain

use std::sync::Arc;

use rustchain::chains::extraction::{
    ExtractionChain, ExtractionSchema, FieldType, SchemaFieldBuilder,
};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::FakeListChatModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Extraction Chain Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Define an ExtractionSchema
    // -------------------------------------------------------------------------
    // The schema describes the structure of the entities we want to extract.
    // Each field has a name, type, description, and required flag.

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

    println!("Step 1: Defined ExtractionSchema");
    println!("  Schema: {}", schema.to_prompt_instruction());
    println!();

    // -------------------------------------------------------------------------
    // Step 2: Create a FakeListChatModel with predefined JSON responses
    // -------------------------------------------------------------------------
    // Each response simulates what a real LLM would return: a JSON array of
    // extracted entities matching the schema.

    let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
        r#"[{"name": "Alice Chen", "age": 34, "occupation": "Machine Learning Engineer", "location": "San Francisco"}]"#.into(),
        r#"[{"name": "Dr. James Wilson", "age": 52, "occupation": "Chief Medical Officer", "location": "Boston"}, {"name": "Sarah Park", "age": 29, "occupation": "Research Scientist", "location": "Boston"}]"#.into(),
        r#"[{"name": "Marcus Johnson", "occupation": "Software Architect"}, {"name": "Elena Rodriguez", "age": 41, "occupation": "VP of Engineering", "location": "Austin"}]"#.into(),
    ]));

    println!("Step 2: Created FakeListChatModel with 3 predefined responses\n");

    // -------------------------------------------------------------------------
    // Step 3: Build the ExtractionChain
    // -------------------------------------------------------------------------
    let chain = ExtractionChain::builder().llm(model).schema(schema).build();

    println!("Step 3: Built ExtractionChain\n");

    // -------------------------------------------------------------------------
    // Step 4: Extract entities from sample texts
    // -------------------------------------------------------------------------
    println!("Step 4: Extracting entities from text\n");

    let texts = [
        "Alice Chen is a 34-year-old machine learning engineer based in San Francisco. She specializes in natural language processing and has published several papers on transformer architectures.",
        "Dr. James Wilson, 52, serves as Chief Medical Officer at Boston General Hospital. He works closely with Sarah Park, a 29-year-old research scientist who joined the team last year to study novel therapeutics.",
        "The engineering team is led by Marcus Johnson, a seasoned software architect, and Elena Rodriguez, 41, who serves as VP of Engineering from the Austin office.",
    ];

    for (i, text) in texts.iter().enumerate() {
        println!("  Input {}: \"{}\"", i + 1, text);

        let result = chain.extract(text).await?;

        println!("  Extracted {} entity/entities:", result.entities.len());
        for (j, entity) in result.entities.iter().enumerate() {
            let name = entity["name"].as_str().unwrap_or("?");
            let age = entity
                .get("age")
                .and_then(|v| v.as_i64())
                .map(|a| format!("{}", a))
                .unwrap_or_else(|| "unknown".to_string());
            let occupation = entity["occupation"].as_str().unwrap_or("?");
            let location = entity
                .get("location")
                .and_then(|v| v.as_str())
                .unwrap_or("not specified");

            println!("    Entity {}:", j + 1);
            println!("      Name:       {}", name);
            println!("      Age:        {}", age);
            println!("      Occupation: {}", occupation);
            println!("      Location:   {}", location);
        }
        println!("  Raw JSON: {}", serde_json::to_string(&result.entities)?);
        println!();
    }

    println!("Done!");
    Ok(())
}
