//! Vector Store Similarity Search Example
//!
//! Demonstrates InMemoryVectorStore: adding documents with metadata,
//! similarity search with scores, and retrieval by ID.
//!
//! Run with: cargo run -p cognis-examples --example vector_store_search

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::vectorstores::base::VectorStore;
use cognis_core::vectorstores::in_memory::InMemoryVectorStore;
use serde_json::json;

fn doc_with_meta(id: &str, content: &str, category: &str) -> Document {
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), json!(category));
    Document {
        page_content: content.to_string(),
        metadata: meta,
        id: Some(id.to_string()),
        doc_type: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(128));
    let store = Arc::new(InMemoryVectorStore::new(embedding));

    // Add documents with metadata.
    let documents = vec![
        doc_with_meta("doc_rust", "Rust is a systems programming language focused on safety, speed, and concurrency.", "language"),
        doc_with_meta("doc_python", "Python is a high-level, interpreted language known for readability and versatility.", "language"),
        doc_with_meta("doc_langchain", "LangChain is a framework for developing applications powered by large language models.", "framework"),
        doc_with_meta("doc_cargo", "Cargo is Rust's build system and package manager for dependencies and compilation.", "tool"),
        doc_with_meta("doc_vectordb", "Vector databases store high-dimensional embeddings for efficient similarity search.", "concept"),
    ];

    let ids = store.add_documents(documents, None).await?;
    println!("Stored {} documents\n", ids.len());

    // Similarity search with scores.
    let queries = [
        "systems programming and memory safety",
        "building and managing software packages",
        "large language model applications",
    ];

    for query in &queries {
        println!("Query: \"{query}\"");
        let results = store.similarity_search_with_score(query, 3).await?;
        for (i, (doc, score)) in results.iter().enumerate() {
            let preview = &doc.page_content[..doc.page_content.len().min(60)];
            println!(
                "  {}. [{:.4}] {}... ({})",
                i + 1,
                score,
                preview,
                doc.id.as_deref().unwrap_or("?")
            );
        }
        println!();
    }

    // Retrieve by ID.
    let fetched = store
        .get_by_ids(&["doc_rust".to_string(), "doc_langchain".to_string()])
        .await?;
    println!("Fetched {} document(s) by ID:", fetched.len());
    for doc in &fetched {
        println!(
            "  [{}] {}",
            doc.id.as_deref().unwrap_or("?"),
            &doc.page_content[..doc.page_content.len().min(60)]
        );
    }

    Ok(())
}
