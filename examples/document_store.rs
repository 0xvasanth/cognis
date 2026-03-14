//! Document Store Example
//!
//! Demonstrates the document store module for managing collections of documents
//! with text search, metadata filtering, pagination, and inverted-index search.
//!
//! Features shown:
//! - Creating an InMemoryDocStore and adding documents
//! - Searching documents by text content
//! - Filtering by metadata using MetadataCondition
//! - Using DocStoreQuery with limit/offset for pagination
//! - Using DocStoreIndex for inverted-index text search
//! - Using IndexedDocStore for combined store + index
//! - Computing DocStoreStats for store analytics
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example document_store`

mod shared;
use std::collections::HashMap;

use cognis::stores::docstore::{
    DocStore, DocStoreIndex, DocStoreQuery, DocStoreStats, InMemoryDocStore, IndexedDocStore,
    MetadataCondition,
};
use cognis_core::documents::Document;
use serde_json::json;

fn main() {
    println!("=== Document Store Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating an InMemoryDocStore and Adding Documents
    // -----------------------------------------------------------------------
    println!("--- 1. Creating an InMemoryDocStore ---");
    println!("The InMemoryDocStore provides a simple HashMap-backed document store.\n");

    let mut store = InMemoryDocStore::new();

    // Add individual documents
    let id1 = store
        .add(Document::new(
            "Rust is a systems programming language focused on safety and performance.",
        ))
        .unwrap();
    println!("Added document with auto-generated ID: {}", id1);

    // Add a document with a custom ID
    let id2 = store
        .add(
            Document::new(
                "Python is a versatile language popular for data science and web development.",
            )
            .with_id("python-intro"),
        )
        .unwrap();
    println!("Added document with custom ID: {}", id2);

    // Add documents with metadata
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), json!("systems"));
    meta.insert("difficulty".to_string(), json!(8.5));
    meta.insert("tags".to_string(), json!("low-level, compiled"));

    let id3 = store
        .add(
            Document::new("C++ is a powerful systems language with manual memory management.")
                .with_metadata(meta),
        )
        .unwrap();
    println!("Added document with metadata, ID: {}", id3);

    // Batch add documents
    let docs = vec![
        Document::new("JavaScript powers the modern web with event-driven programming."),
        Document::new("Go is a statically typed language designed for simplicity and concurrency."),
        Document::new(
            "Haskell is a purely functional programming language with strong static typing.",
        ),
    ];
    let batch_ids = store.add_batch(docs).unwrap();
    println!("Batch added {} documents", batch_ids.len());
    println!("Total documents in store: {}\n", store.count());

    // -----------------------------------------------------------------------
    // 2. Searching Documents by Text
    // -----------------------------------------------------------------------
    println!("--- 2. Text Search ---");
    println!("Search for documents containing specific text (case-insensitive).\n");

    let query = DocStoreQuery::new().with_text("programming");
    let results = store.search(&query).unwrap();
    println!(
        "Search for 'programming' found {} documents:",
        results.len()
    );
    for doc in &results {
        println!(
            "  [{}] {}",
            doc.id.as_deref().unwrap_or("?"),
            doc.page_content
        );
    }

    println!();

    let query = DocStoreQuery::new().with_text("SYSTEMS");
    let results = store.search(&query).unwrap();
    println!(
        "Search for 'SYSTEMS' (case-insensitive) found {} documents:",
        results.len()
    );
    for doc in &results {
        println!(
            "  [{}] {}...",
            doc.id.as_deref().unwrap_or("?"),
            &doc.page_content[..50]
        );
    }

    println!();

    let query = DocStoreQuery::new().with_text("quantum computing");
    let results = store.search(&query).unwrap();
    println!(
        "Search for 'quantum computing' found {} documents (no match expected)\n",
        results.len()
    );

    // -----------------------------------------------------------------------
    // 3. Filtering by Metadata Using MetadataCondition
    // -----------------------------------------------------------------------
    println!("--- 3. Metadata Filtering ---");
    println!("Use MetadataCondition to filter documents by their metadata fields.\n");

    // Add more documents with metadata for filtering
    let mut store2 = InMemoryDocStore::new();

    let articles = vec![
        (
            "Intro to Rust ownership",
            vec![("category", json!("tutorial")), ("level", json!(3.0))],
        ),
        (
            "Advanced Rust lifetimes",
            vec![("category", json!("tutorial")), ("level", json!(8.0))],
        ),
        (
            "Rust in production at scale",
            vec![("category", json!("case-study")), ("level", json!(5.0))],
        ),
        (
            "Building web APIs with Rust",
            vec![("category", json!("tutorial")), ("level", json!(4.5))],
        ),
        (
            "Comparing Rust and C++",
            vec![("category", json!("comparison")), ("level", json!(6.0))],
        ),
    ];

    for (content, meta_pairs) in &articles {
        let metadata: HashMap<String, serde_json::Value> = meta_pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        store2
            .add(Document::new(*content).with_metadata(metadata))
            .unwrap();
    }

    // Equals filter
    let query = DocStoreQuery::new().with_metadata("category", json!("tutorial"));
    let results = store2.search(&query).unwrap();
    println!(
        "Tutorials (category == 'tutorial'): {} found",
        results.len()
    );
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // GreaterThan filter using raw MetadataCondition
    let mut query = DocStoreQuery::new();
    query
        .metadata_filters
        .push(MetadataCondition::GreaterThan("level".to_string(), 5.0));
    let results = store2.search(&query).unwrap();
    println!("\nAdvanced articles (level > 5.0): {} found", results.len());
    for doc in &results {
        let level = doc
            .metadata
            .get("level")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        println!("  - {} (level: {})", doc.page_content, level);
    }

    // Contains filter
    let mut query = DocStoreQuery::new();
    query.metadata_filters.push(MetadataCondition::Contains(
        "category".to_string(),
        "study".to_string(),
    ));
    let results = store2.search(&query).unwrap();
    println!(
        "\nCase studies (category contains 'study'): {} found",
        results.len()
    );
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // Combined: text search + metadata filter
    let mut query = DocStoreQuery::new().with_text("rust");
    query
        .metadata_filters
        .push(MetadataCondition::LessThan("level".to_string(), 5.0));
    let results = store2.search(&query).unwrap();
    println!(
        "\nBeginner Rust articles (text='rust', level < 5.0): {} found",
        results.len()
    );
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // Exists / NotExists
    let mut query = DocStoreQuery::new();
    query
        .metadata_filters
        .push(MetadataCondition::Exists("level".to_string()));
    let results = store2.search(&query).unwrap();
    println!("\nDocuments with 'level' metadata: {} found", results.len());

    let mut query = DocStoreQuery::new();
    query
        .metadata_filters
        .push(MetadataCondition::NotExists("author".to_string()));
    let results = store2.search(&query).unwrap();
    println!(
        "Documents without 'author' metadata: {} found\n",
        results.len()
    );

    // -----------------------------------------------------------------------
    // 4. DocStoreQuery with Limit and Offset for Pagination
    // -----------------------------------------------------------------------
    println!("--- 4. Pagination with Limit and Offset ---");
    println!("Use limit and offset on DocStoreQuery to paginate results.\n");

    let mut paginated_store = InMemoryDocStore::new();
    for i in 1..=10 {
        paginated_store
            .add(Document::new(format!("Document number {}", i)).with_id(format!("doc-{:02}", i)))
            .unwrap();
    }

    // Page 1: first 3 results
    let query = DocStoreQuery::new().with_limit(3).with_offset(0);
    let page1 = paginated_store.search(&query).unwrap();
    println!("Page 1 (limit=3, offset=0): {} results", page1.len());
    for doc in &page1 {
        println!("  - {}", doc.page_content);
    }

    // Page 2: next 3 results
    let query = DocStoreQuery::new().with_limit(3).with_offset(3);
    let page2 = paginated_store.search(&query).unwrap();
    println!("\nPage 2 (limit=3, offset=3): {} results", page2.len());
    for doc in &page2 {
        println!("  - {}", doc.page_content);
    }

    // Page 4: beyond the end
    let query = DocStoreQuery::new().with_limit(3).with_offset(9);
    let page4 = paginated_store.search(&query).unwrap();
    println!(
        "\nPage 4 (limit=3, offset=9): {} results (partial last page)\n",
        page4.len()
    );

    // -----------------------------------------------------------------------
    // 5. DocStoreIndex for Inverted-Index Text Search
    // -----------------------------------------------------------------------
    println!("--- 5. Inverted-Index Text Search (DocStoreIndex) ---");
    println!("DocStoreIndex builds a TF-based inverted index for fast text search.\n");

    let mut index = DocStoreIndex::new();

    index.index_document(
        "d1",
        "Rust provides memory safety without garbage collection",
    );
    index.index_document(
        "d2",
        "Garbage collection in Java manages memory automatically",
    );
    index.index_document("d3", "Rust and C++ both offer fine-grained memory control");
    index.index_document("d4", "Python uses garbage collection for memory management");

    println!(
        "Indexed {} documents with {} distinct terms",
        index.document_count(),
        index.term_count()
    );

    // Search by relevance
    let results = index.search("memory safety");
    println!("\nSearch for 'memory safety' (ranked by TF score):");
    for (doc_id, score) in &results {
        println!("  {} (score: {:.4})", doc_id, score);
    }

    let results = index.search("garbage collection");
    println!("\nSearch for 'garbage collection':");
    for (doc_id, score) in &results {
        println!("  {} (score: {:.4})", doc_id, score);
    }

    // Remove a document from the index
    index.remove("d4");
    println!(
        "\nRemoved 'd4' from index. Documents remaining: {}",
        index.document_count()
    );

    let results = index.search("garbage");
    println!("Search for 'garbage' after removal:");
    for (doc_id, score) in &results {
        println!("  {} (score: {:.4})", doc_id, score);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. IndexedDocStore — Combined Store + Index
    // -----------------------------------------------------------------------
    println!("--- 6. IndexedDocStore (Store + Index) ---");
    println!("IndexedDocStore combines InMemoryDocStore with DocStoreIndex.\n");

    let mut indexed_store = IndexedDocStore::new();

    indexed_store
        .add(
            Document::new("Machine learning algorithms learn patterns from data")
                .with_metadata(HashMap::from([("field".to_string(), json!("AI"))])),
        )
        .unwrap();
    indexed_store
        .add(
            Document::new("Deep learning uses neural networks with many layers")
                .with_metadata(HashMap::from([("field".to_string(), json!("AI"))])),
        )
        .unwrap();
    indexed_store
        .add(
            Document::new("Database indexing improves query performance significantly")
                .with_metadata(HashMap::from([("field".to_string(), json!("databases"))])),
        )
        .unwrap();
    indexed_store
        .add(
            Document::new("Learning SQL is essential for data engineering")
                .with_metadata(HashMap::from([("field".to_string(), json!("databases"))])),
        )
        .unwrap();

    println!("IndexedDocStore has {} documents", indexed_store.count());

    // Text search uses the inverted index under the hood
    let query = DocStoreQuery::new().with_text("learning");
    let results = indexed_store.search(&query).unwrap();
    println!("\nSearch for 'learning' (uses inverted index):");
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // Combined text + metadata filter
    let mut query = DocStoreQuery::new().with_text("learning");
    query
        .metadata_filters
        .push(MetadataCondition::Equals("field".to_string(), json!("AI")));
    let results = indexed_store.search(&query).unwrap();
    println!("\nSearch for 'learning' in AI field only:");
    for doc in &results {
        println!("  - {}", doc.page_content);
    }

    // Update a document and verify re-indexing
    let query = DocStoreQuery::new().with_text("database");
    let db_docs = indexed_store.search(&query).unwrap();
    if let Some(db_doc) = db_docs.first() {
        let doc_id = db_doc.id.clone().unwrap();
        indexed_store
            .update(
                &doc_id,
                Document::new("Search engine indexing enables fast full-text retrieval")
                    .with_metadata(HashMap::from([("field".to_string(), json!("search"))])),
            )
            .unwrap();
        println!("\nUpdated document '{}' with new content", doc_id);

        // Old text should no longer match
        let query = DocStoreQuery::new().with_text("database");
        let results = indexed_store.search(&query).unwrap();
        println!(
            "Search for 'database' after update: {} results (old content removed from index)",
            results.len()
        );

        // New text should match
        let query = DocStoreQuery::new().with_text("search engine");
        let results = indexed_store.search(&query).unwrap();
        println!(
            "Search for 'search engine': {} result (new content indexed)",
            results.len()
        );
    }

    // Without text query, falls back to the underlying store search
    let query = DocStoreQuery::new();
    let results = indexed_store.search(&query).unwrap();
    println!(
        "\nSearch with no text filter: {} documents (returns all)\n",
        results.len()
    );

    // -----------------------------------------------------------------------
    // 7. DocStoreStats — Store Analytics
    // -----------------------------------------------------------------------
    println!("--- 7. DocStoreStats ---");
    println!("Compute summary statistics about an InMemoryDocStore.\n");

    let mut stats_store = InMemoryDocStore::new();

    let sample_docs = vec![
        ("Short note.", vec![("type", json!("note")), ("priority", json!(1))]),
        ("A medium-length document with some additional content.", vec![("type", json!("article")), ("priority", json!(3))]),
        ("This is a longer document that contains significantly more text to demonstrate how document length statistics are computed across the store.", vec![("type", json!("article")), ("author", json!("Alice"))]),
    ];

    for (content, meta_pairs) in &sample_docs {
        let metadata: HashMap<String, serde_json::Value> = meta_pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        stats_store
            .add(Document::new(*content).with_metadata(metadata))
            .unwrap();
    }

    let stats = DocStoreStats::from_store(&stats_store);
    println!("Store statistics:");
    println!("  Total documents:      {}", stats.total_documents);
    println!("  Total characters:     {}", stats.total_chars);
    println!("  Avg document length:  {:.1} chars", stats.avg_doc_length);
    println!("  Metadata keys:        {:?}", stats.metadata_keys);

    // Stats as JSON
    let stats_json = stats.to_json();
    println!("\nStats as JSON:");
    println!("  {}", serde_json::to_string_pretty(&stats_json).unwrap());

    // -----------------------------------------------------------------------
    // 8. Real LLM Demo — Q&A using stored documents
    // -----------------------------------------------------------------------
    println!("\n--- 8. Real LLM Demo (Q&A with Stored Documents) ---\n");

    let model = shared::get_chat_model(vec![
        "Based on the stored documents, machine learning algorithms learn patterns from data, and deep learning uses neural networks with many layers. These are sub-fields of AI.".into(),
    ]);

    // Retrieve relevant documents from the indexed store
    let query = DocStoreQuery::new().with_text("learning");
    let relevant_docs = indexed_store.search(&query).unwrap();
    let context: String = relevant_docs
        .iter()
        .map(|doc| doc.page_content.clone())
        .collect::<Vec<_>>()
        .join("\n");

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "Answer the user's question based only on the provided context.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            &format!("Context:\n{}\n\nQuestion: What do these documents tell us about learning?", context),
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let ai_msg = rt.block_on(async {
        use cognis_core::language_models::chat_model::BaseChatModel;
        model.invoke_messages(&messages, None).await
    });

    match ai_msg {
        Ok(response) => println!("  LLM answer: {}", response.content),
        Err(e) => println!("  LLM error: {}", e),
    }

    println!("\n=== Document Store Example Complete ===");
}
