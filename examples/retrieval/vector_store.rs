//! Vector Store Example
//!
//! Demonstrates creating an in-memory vector store, adding documents with
//! embeddings, and running similarity search with metadata filtering.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example vector_store`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognis::vectorstores::memory::{
    InMemoryVectorStore, SearchQuery, SimilarityMetric, VectorEntry,
};
use cognis_core::messages::Message;
use serde_json::Value;

/// Helper to build a VectorEntry with a topic metadata field.
fn doc(id: &str, embedding: Vec<f64>, text: &str, topic: &str) -> VectorEntry {
    let mut meta = HashMap::new();
    meta.insert("topic".to_string(), Value::String(topic.into()));
    VectorEntry::new(id, embedding, text, meta)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vector Store Example ===\n");

    // 1. Build a vector store and add documents
    let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);

    store.add_batch(vec![
        doc(
            "rust",
            vec![1.0, 0.0, 0.0],
            "Rust programming language",
            "language",
        ),
        doc(
            "python",
            vec![0.8, 0.6, 0.0],
            "Python programming language",
            "language",
        ),
        doc(
            "ml",
            vec![0.3, 0.9, 0.2],
            "Machine learning with neural networks",
            "ai",
        ),
        doc(
            "llm",
            vec![0.4, 0.8, 0.4],
            "Large language models and transformers",
            "ai",
        ),
        doc(
            "web",
            vec![0.0, 0.1, 1.0],
            "Web development with JavaScript",
            "web",
        ),
    ]);

    println!(
        "Store: {} entries, {:?} dimensions\n",
        store.len(),
        store.dimensions()
    );

    // 2. Similarity search — find documents closest to a query embedding
    let query = SearchQuery::builder(vec![1.0, 0.0, 0.0]).top_k(3).build();
    let results = store.search(&query);

    println!("Top 3 results for query [1.0, 0.0, 0.0]:");
    for r in &results {
        println!(
            "  {:<8} score={:.4}  \"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // 3. Filtered search — restrict results by metadata
    let mut filter = HashMap::new();
    filter.insert("topic".to_string(), Value::String("ai".into()));

    let filtered = SearchQuery::builder(vec![0.5, 0.8, 0.3])
        .top_k(5)
        .metadata_filter(filter)
        .build();
    let results = store.search(&filtered);

    println!("\nFiltered results (topic=ai):");
    for r in &results {
        println!(
            "  {:<8} score={:.4}  \"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // 4. Threshold search — only return high-confidence matches
    let threshold = SearchQuery::builder(vec![1.0, 0.0, 0.0])
        .top_k(10)
        .min_score(0.9)
        .build();
    let results = store.search(&threshold);

    println!("\nResults with min_score=0.9:");
    for r in &results {
        println!(
            "  {:<8} score={:.4}  \"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // 5. Use search results as context for an LLM question
    let context_query = SearchQuery::builder(vec![0.4, 0.8, 0.3]).top_k(2).build();
    let context_results = store.search(&context_query);
    let context: String = context_results
        .iter()
        .map(|r| format!("- {}", r.entry.document))
        .collect::<Vec<_>>()
        .join("\n");

    let model = shared::get_chat_model(vec![
        "The topics most related to AI are machine learning with neural networks \
         and large language models with transformers."
            .into(),
    ]);
    let messages = vec![
        Message::system("Answer using only the provided context."),
        Message::human(&format!(
            "Context:\n{context}\n\nQuestion: What topics are related to AI?"
        )),
    ];

    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("\nRetrieved context:\n{context}");
        println!("\nLLM answer: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
