//! LCEL Chain Composition Example
//!
//! Demonstrates how to compose a chain using the LCEL (LangChain Expression Language)
//! pattern: ChatPromptTemplate -> FakeListChatModel -> StrOutputParser.
//!
//! No API keys required -- uses fake/mock models.

use std::sync::Arc;

use serde_json::json;

use rustchain_core::chain;
use rustchain_core::language_models::{ChatModelRunnable, FakeListChatModel};
use rustchain_core::output_parsers::StrOutputParser;
use rustchain_core::prompts::ChatPromptTemplate;
use rustchain_core::runnables::Runnable;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LCEL Chain Composition Example ===\n");

    // Step 1: Create a ChatPromptTemplate with a system message and a user variable.
    //
    // The template uses {topic} as a variable that will be filled at invoke time.
    let prompt = ChatPromptTemplate::from_messages(vec![
        (
            "system",
            "You are a helpful assistant that explains topics concisely.",
        ),
        ("human", "Explain {topic} in one sentence."),
    ])?;

    println!(
        "Created ChatPromptTemplate with input variables: {:?}",
        prompt.input_variables
    );

    // Step 2: Create a FakeListChatModel that returns predefined responses.
    //
    // In a real application, this would be an OpenAI, Anthropic, or other provider model.
    let model = FakeListChatModel::new(vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.".into(),
        "Python is a high-level, interpreted language known for readability and versatility."
            .into(),
        "TypeScript adds static typing to JavaScript for better tooling and error detection."
            .into(),
    ]);

    // Step 3: Create a StrOutputParser to extract text from the model's response.
    let parser = StrOutputParser;

    // Step 4: Compose the chain using the chain! macro.
    //
    // This creates a RunnableSequence: prompt -> model -> parser.
    // Data flows through each step: JSON input -> formatted messages -> AI response -> string.
    let model_runnable = ChatModelRunnable::new(Arc::new(model));
    let chain = chain!(prompt, model_runnable, parser)?;

    println!("Built chain: {}\n", chain.name());

    // Step 5: Invoke the chain with different topics.
    let topics = vec!["Rust", "Python", "TypeScript"];

    for topic in topics {
        let input = json!({ "topic": topic });
        let result = chain.invoke(input, None).await?;

        // The result is a JSON string value.
        let text = match result.as_str() {
            Some(s) => s.to_string(),
            None => result.to_string(),
        };
        println!("Topic: {topic}");
        println!("Response: {text}\n");
    }

    // Step 6: Demonstrate batch invocation.
    println!("--- Batch Invocation ---\n");

    let batch_inputs = vec![json!({ "topic": "Rust" }), json!({ "topic": "Python" })];
    let batch_results = chain.batch(batch_inputs, None).await?;

    for (i, result) in batch_results.iter().enumerate() {
        let text = match result.as_str() {
            Some(s) => s.to_string(),
            None => result.to_string(),
        };
        println!("Batch result {}: {text}", i + 1);
    }

    println!("\nDone!");
    Ok(())
}
