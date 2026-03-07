//! Vector Store Similarity Search Example
//!
//! Demonstrates how to use InMemoryVectorStore for document storage and
//! similarity search:
//! - Creating an InMemoryVectorStore with DeterministicFakeEmbedding
//! - Adding documents with metadata
//! - Performing similarity search
//! - Displaying results with similarity scores
//!
//! No API keys required -- uses DeterministicFakeEmbedding.
//!
//! Run with: cargo run -p rustchain-examples --example vector_store_search

use std::collections::HashMap;
use std::sync::Arc;

use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
use rustchain_core::vectorstores::base::VectorStore;
use rustchain_core::vectorstores::in_memory::InMemoryVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vector Store Similarity Search Example ===\n");

    // Step 1: Create a fake embedding model.
    //
    // DeterministicFakeEmbedding produces hash-based vectors that are consistent
    // for the same input text. The dimension parameter controls vector size.
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(128));
    println!("Step 1: Created DeterministicFakeEmbedding (dim=128)\n");

    // Step 2: Create an InMemoryVectorStore.
    let store = Arc::new(InMemoryVectorStore::new(embedding));
    println!("Step 2: Created InMemoryVectorStore\n");

    // Step 3: Prepare documents with metadata.
    //
    // Each document has page content and optional metadata that can be used
    // for filtering or display.
    let documents = vec![
        {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::json!("language"));
            meta.insert("year".to_string(), serde_json::json!(2010));
            Document {
                page_content: "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents segfaults and guarantees thread safety.".to_string(),
                metadata: meta,
                id: Some("doc_rust".to_string()),
            }
        },
        {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::json!("language"));
            meta.insert("year".to_string(), serde_json::json!(1991));
            Document {
                page_content: "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms.".to_string(),
                metadata: meta,
                id: Some("doc_python".to_string()),
            }
        },
        {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::json!("framework"));
            meta.insert("year".to_string(), serde_json::json!(2022));
            Document {
                page_content: "LangChain is a framework for developing applications powered by large language models. It provides tools for prompt management, chains, and agents.".to_string(),
                metadata: meta,
                id: Some("doc_langchain".to_string()),
            }
        },
        {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::json!("tool"));
            meta.insert("year".to_string(), serde_json::json!(2015));
            Document {
                page_content: "Cargo is Rust's build system and package manager. It downloads dependencies, compiles packages, and runs tests.".to_string(),
                metadata: meta,
                id: Some("doc_cargo".to_string()),
            }
        },
        {
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), serde_json::json!("concept"));
            meta.insert("year".to_string(), serde_json::json!(2020));
            Document {
                page_content: "Vector databases store high-dimensional embeddings for efficient similarity search. They are essential for retrieval-augmented generation (RAG) pipelines.".to_string(),
                metadata: meta,
                id: Some("doc_vectordb".to_string()),
            }
        },
    ];

    println!("Step 3: Prepared {} documents with metadata:", documents.len());
    for doc in &documents {
        let preview = if doc.page_content.len() > 60 {
            format!("{}...", &doc.page_content[..60])
        } else {
            doc.page_content.clone()
        };
        println!(
            "  [{}] {} (category: {})",
            doc.id.as_deref().unwrap_or("?"),
            preview,
            doc.metadata
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
        );
    }
    println!();

    // Step 4: Add documents to the vector store.
    let ids = store.add_documents(documents, None).await?;
    println!("Step 4: Added {} documents to vector store", ids.len());
    println!("  IDs: {:?}\n", ids);

    // Step 5: Perform similarity searches.
    println!("Step 5: Performing similarity searches\n");

    let queries = [
        "systems programming and memory safety",
        "building and managing software packages",
        "large language model applications",
    ];

    for query in &queries {
        println!("  Query: \"{query}\"");

        // Search with scores to see similarity values.
        let results = store.similarity_search_with_score(query, 3).await?;

        println!("  Top {} results:", results.len());
        for (i, (doc, score)) in results.iter().enumerate() {
            let preview = if doc.page_content.len() > 55 {
                format!("{}...", &doc.page_content[..55])
            } else {
                doc.page_content.clone()
            };
            println!(
                "    {}. [score: {:.4}] {} (id: {})",
                i + 1,
                score,
                preview,
                doc.id.as_deref().unwrap_or("?")
            );
        }
        println!();
    }

    // Step 6: Basic similarity search (without scores).
    println!("Step 6: Basic similarity search (no scores)\n");

    let basic_results = store.similarity_search("programming language", 2).await?;
    println!("  Query: \"programming language\" (k=2)");
    for (i, doc) in basic_results.iter().enumerate() {
        println!("    {}. {}", i + 1, doc.page_content);
    }
    println!();

    // Step 7: Retrieve documents by ID.
    println!("Step 7: Retrieve by ID\n");

    let fetched = store
        .get_by_ids(&["doc_rust".to_string(), "doc_langchain".to_string()])
        .await?;
    println!("  Fetched {} document(s) by ID:", fetched.len());
    for doc in &fetched {
        let preview = if doc.page_content.len() > 60 {
            format!("{}...", &doc.page_content[..60])
        } else {
            doc.page_content.clone()
        };
        println!("    [{}] {}", doc.id.as_deref().unwrap_or("?"), preview);
    }

    println!("\nDone!");
    Ok(())
}
