//! Document Store Example
//!
//! Shows how to create a document store, add documents with metadata,
//! search/filter them, and use the results as context for an LLM query.
//!
//! Run with: `cargo run -p cognis-examples --example document_store`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognis::stores::docstore::{DocStore, DocStoreQuery, IndexedDocStore, MetadataCondition};
use cognis_core::documents::Document;
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use serde_json::json;

fn main() {
    println!("=== Document Store Example ===\n");

    // --- 1. Build a document store with some articles ---
    let mut store = IndexedDocStore::new();

    let articles = vec![
        (
            "Rust provides memory safety without garbage collection.",
            "systems",
            8.0,
        ),
        (
            "Python is popular for data science and web development.",
            "scripting",
            3.0,
        ),
        (
            "Machine learning algorithms learn patterns from data.",
            "AI",
            5.0,
        ),
        (
            "Deep learning uses neural networks with many layers.",
            "AI",
            7.0,
        ),
        (
            "Go is designed for simplicity and concurrency.",
            "systems",
            4.0,
        ),
        (
            "SQL is essential for data engineering and analytics.",
            "databases",
            3.5,
        ),
    ];

    for (content, category, level) in &articles {
        let metadata = HashMap::from([
            ("category".to_string(), json!(category)),
            ("level".to_string(), json!(level)),
        ]);
        store
            .add(Document::new(*content).with_metadata(metadata))
            .unwrap();
    }
    println!("Added {} documents to the store.", store.count());

    // --- 2. Text search (uses inverted index) ---
    let query = DocStoreQuery::new().with_text("learning");
    let results = store.search(&query).unwrap();
    println!("\nSearch for 'learning' ({} results):", results.len());
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // --- 3. Metadata filtering ---
    let query = DocStoreQuery::new().with_metadata("category", json!("AI"));
    let results = store.search(&query).unwrap();
    println!("\nAI articles ({} results):", results.len());
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // Combined: text search + metadata filter (level > 5.0)
    let mut query = DocStoreQuery::new().with_text("learning");
    query
        .metadata_filters
        .push(MetadataCondition::GreaterThan("level".to_string(), 5.0));
    let results = store.search(&query).unwrap();
    println!(
        "\nAdvanced learning articles, level > 5.0 ({} results):",
        results.len()
    );
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // --- 4. Pagination ---
    let page1 = store
        .search(&DocStoreQuery::new().with_limit(3).with_offset(0))
        .unwrap();
    let page2 = store
        .search(&DocStoreQuery::new().with_limit(3).with_offset(3))
        .unwrap();
    println!(
        "\nPagination: page 1 has {} docs, page 2 has {} docs",
        page1.len(),
        page2.len()
    );

    // --- 5. Use search results as LLM context ---
    println!("\n--- LLM Q&A with Retrieved Documents ---\n");

    let model = shared::get_chat_model(vec![
        "Based on the documents, machine learning learns patterns from data \
         while deep learning specifically uses neural networks with many layers. \
         Both are sub-fields of AI."
            .into(),
    ]);

    let query = DocStoreQuery::new().with_metadata("category", json!("AI"));
    let relevant_docs = store.search(&query).unwrap();
    let context: String = relevant_docs
        .iter()
        .map(|d| d.page_content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let messages = vec![
        Message::System(SystemMessage::new(
            "Answer the user's question based only on the provided context.",
        )),
        Message::Human(HumanMessage::new(&format!(
            "Context:\n{context}\n\nQuestion: What do these documents say about AI and learning?"
        ))),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(async { model.invoke_messages(&messages, None).await }) {
        Ok(response) => println!("LLM answer: {}", response.base.content.text()),
        Err(e) => println!("LLM error: {}", e),
    }

    println!("\n=== Done ===");
}
