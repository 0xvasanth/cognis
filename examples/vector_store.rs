//! Vector Store Example
//!
//! Demonstrates the in-memory vector store for similarity search, including
//! embedding creation, multiple similarity metrics, configurable search queries,
//! Maximal Marginal Relevance (MMR) search for diversity, and store statistics.
//!
//! Features shown:
//! - VectorEntry creation with embeddings and metadata
//! - InMemoryVectorStore add and search operations
//! - All SimilarityMetric types (Cosine, Euclidean, DotProduct, Manhattan)
//! - SearchQuery with top_k, min_score, and metadata filtering
//! - MaxMarginalRelevance search for balancing relevance and diversity
//! - VectorStoreStats computation and inspection
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example vector_store`

mod shared;
use std::collections::HashMap;

use cognis::vectorstores::memory::{
    InMemoryVectorStore, MaxMarginalRelevance, SearchQuery, SimilarityMetric, VectorEntry,
    VectorStoreStats,
};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vector Store Example ===\n");

    // -----------------------------------------------------------------------
    // 1. VectorEntry Creation
    // -----------------------------------------------------------------------
    println!("--- 1. VectorEntry Creation ---");
    println!("Each entry pairs an embedding vector with document text and metadata.\n");

    // Simple entry without metadata
    let entry_a = VectorEntry::new(
        "doc-1",
        vec![1.0, 0.0, 0.0],
        "Rust is a systems programming language",
        HashMap::new(),
    );
    println!("Entry '{}': \"{}\"", entry_a.id, entry_a.document);
    println!("  dimensions: {}", entry_a.dimensions());

    // Entry with metadata
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), Value::String("wiki".into()));
    metadata.insert("category".to_string(), Value::String("programming".into()));

    let entry_b = VectorEntry::new(
        "doc-2",
        vec![0.9, 0.1, 0.0],
        "Rust focuses on safety and performance",
        metadata,
    );
    println!("Entry '{}': \"{}\"", entry_b.id, entry_b.document);
    println!("  metadata: {:?}", entry_b.metadata);

    // Serialize entry to JSON
    let json = entry_a.to_json();
    println!("\nEntry as JSON:");
    println!("  {}", serde_json::to_string_pretty(&json).unwrap());

    // -----------------------------------------------------------------------
    // 2. InMemoryVectorStore — Add and Search
    // -----------------------------------------------------------------------
    println!("\n--- 2. InMemoryVectorStore — Add and Search ---");
    println!("Create a store, add entries, and perform similarity search.\n");

    let mut store = InMemoryVectorStore::new(SimilarityMetric::Cosine);

    // Create a set of document entries with fake embeddings
    let entries = vec![
        VectorEntry::new("rust", vec![1.0, 0.0, 0.0], "Rust programming language", {
            let mut m = HashMap::new();
            m.insert("topic".to_string(), Value::String("language".into()));
            m
        }),
        VectorEntry::new(
            "python",
            vec![0.8, 0.6, 0.0],
            "Python programming language",
            {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), Value::String("language".into()));
                m
            },
        ),
        VectorEntry::new(
            "ml",
            vec![0.3, 0.9, 0.2],
            "Machine learning with neural networks",
            {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), Value::String("ai".into()));
                m
            },
        ),
        VectorEntry::new(
            "llm",
            vec![0.4, 0.8, 0.4],
            "Large language models and transformers",
            {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), Value::String("ai".into()));
                m
            },
        ),
        VectorEntry::new(
            "web",
            vec![0.0, 0.1, 1.0],
            "Web development with JavaScript",
            {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), Value::String("web".into()));
                m
            },
        ),
    ];

    store.add_batch(entries);
    println!("Added {} entries to the store", store.len());
    println!("Dimensions: {:?}", store.dimensions());

    // Basic search
    let query = SearchQuery::builder(vec![1.0, 0.0, 0.0]).top_k(3).build();
    let results = store.search(&query);
    println!("\nSearch for [1.0, 0.0, 0.0] (top 3):");
    for r in &results {
        println!(
            "  rank={} id={:<8} score={:.4} doc=\"{}\"",
            r.rank, r.entry.id, r.score, r.entry.document
        );
    }

    // -----------------------------------------------------------------------
    // 3. All SimilarityMetric Types
    // -----------------------------------------------------------------------
    println!("\n--- 3. Similarity Metrics ---");
    println!("Compare the same vectors using all four metrics.\n");

    let vec_a = vec![1.0, 0.5, 0.0];
    let vec_b = vec![0.8, 0.6, 0.1];

    let metrics = [
        SimilarityMetric::Cosine,
        SimilarityMetric::Euclidean,
        SimilarityMetric::DotProduct,
        SimilarityMetric::Manhattan,
    ];

    for metric in &metrics {
        let score = metric.compute(&vec_a, &vec_b);
        println!(
            "  {:<12} score: {:>8.4}  (higher = more similar)",
            metric.name(),
            score
        );
    }

    println!("\nNote: Euclidean and Manhattan scores are negated distances,");
    println!("so higher values (closer to 0) indicate greater similarity.");

    // Search using each metric on the same data
    println!("\nTop result for each metric (query = [1.0, 0.0, 0.0]):");
    for metric in &metrics {
        let mut metric_store = InMemoryVectorStore::new(*metric);
        metric_store.add(VectorEntry::new(
            "close",
            vec![0.95, 0.05, 0.0],
            "close",
            HashMap::new(),
        ));
        metric_store.add(VectorEntry::new(
            "mid",
            vec![0.5, 0.5, 0.0],
            "mid",
            HashMap::new(),
        ));
        metric_store.add(VectorEntry::new(
            "far",
            vec![0.0, 0.0, 1.0],
            "far",
            HashMap::new(),
        ));

        let q = SearchQuery::builder(vec![1.0, 0.0, 0.0]).top_k(1).build();
        let res = metric_store.search(&q);
        if let Some(top) = res.first() {
            println!(
                "  {:<12} -> id={:<6} score={:.4}",
                metric.name(),
                top.entry.id,
                top.score
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. SearchQuery with top_k, min_score, and Metadata Filtering
    // -----------------------------------------------------------------------
    println!("\n--- 4. SearchQuery Options ---");
    println!("Combine top_k, min_score, and metadata filters.\n");

    // Search with min_score threshold
    let high_threshold = SearchQuery::builder(vec![1.0, 0.0, 0.0])
        .top_k(10)
        .min_score(0.9)
        .build();
    let results = store.search(&high_threshold);
    println!("Search with min_score=0.9:");
    for r in &results {
        println!(
            "  id={:<8} score={:.4} doc=\"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // Search with metadata filter
    let mut filter = HashMap::new();
    filter.insert("topic".to_string(), Value::String("ai".into()));
    let filtered_query = SearchQuery::builder(vec![0.5, 0.8, 0.3])
        .top_k(5)
        .metadata_filter(filter)
        .build();
    let results = store.search(&filtered_query);
    println!("\nSearch with metadata filter (topic=ai):");
    for r in &results {
        println!(
            "  id={:<8} score={:.4} doc=\"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // Combined: metadata filter + min_score
    let mut strict_filter = HashMap::new();
    strict_filter.insert("topic".to_string(), Value::String("language".into()));
    let combined_query = SearchQuery::builder(vec![1.0, 0.0, 0.0])
        .top_k(10)
        .min_score(0.8)
        .metadata_filter(strict_filter)
        .build();
    let results = store.search(&combined_query);
    println!("\nSearch with topic=language AND min_score=0.8:");
    for r in &results {
        println!(
            "  id={:<8} score={:.4} doc=\"{}\"",
            r.entry.id, r.score, r.entry.document
        );
    }

    // -----------------------------------------------------------------------
    // 5. MaxMarginalRelevance Search
    // -----------------------------------------------------------------------
    println!("\n--- 5. MaxMarginalRelevance (MMR) Search ---");
    println!("Balance relevance and diversity using the lambda parameter.\n");

    // Add some closely-clustered entries for a better MMR demonstration
    let mut mmr_store = InMemoryVectorStore::new(SimilarityMetric::Cosine);
    mmr_store.add(VectorEntry::new(
        "a1",
        vec![1.0, 0.0, 0.0],
        "Topic A - variant 1",
        HashMap::new(),
    ));
    mmr_store.add(VectorEntry::new(
        "a2",
        vec![0.98, 0.02, 0.0],
        "Topic A - variant 2",
        HashMap::new(),
    ));
    mmr_store.add(VectorEntry::new(
        "a3",
        vec![0.95, 0.05, 0.0],
        "Topic A - variant 3",
        HashMap::new(),
    ));
    mmr_store.add(VectorEntry::new(
        "b1",
        vec![0.0, 1.0, 0.0],
        "Topic B - variant 1",
        HashMap::new(),
    ));
    mmr_store.add(VectorEntry::new(
        "c1",
        vec![0.0, 0.0, 1.0],
        "Topic C - variant 1",
        HashMap::new(),
    ));

    let query_vec = vec![1.0, 0.0, 0.0];

    // Pure relevance (lambda=1.0) — acts like standard similarity search
    let relevant = MaxMarginalRelevance::search(&mmr_store, &query_vec, 4, 1.0);
    println!("MMR with lambda=1.0 (pure relevance):");
    for r in &relevant {
        println!(
            "  rank={} id={:<4} score={:.4} doc=\"{}\"",
            r.rank, r.entry.id, r.score, r.entry.document
        );
    }

    // Balanced (lambda=0.5) — mix of relevance and diversity
    let balanced = MaxMarginalRelevance::search(&mmr_store, &query_vec, 4, 0.5);
    println!("\nMMR with lambda=0.5 (balanced):");
    for r in &balanced {
        println!(
            "  rank={} id={:<4} score={:.4} doc=\"{}\"",
            r.rank, r.entry.id, r.score, r.entry.document
        );
    }

    // Pure diversity (lambda=0.0) — maximally diverse results
    let diverse = MaxMarginalRelevance::search(&mmr_store, &query_vec, 4, 0.0);
    println!("\nMMR with lambda=0.0 (pure diversity):");
    for r in &diverse {
        println!(
            "  rank={} id={:<4} score={:.4} doc=\"{}\"",
            r.rank, r.entry.id, r.score, r.entry.document
        );
    }

    // -----------------------------------------------------------------------
    // 6. VectorStoreStats
    // -----------------------------------------------------------------------
    println!("\n--- 6. VectorStoreStats ---");
    println!("Compute and inspect statistics about the vector store.\n");

    let stats = VectorStoreStats::from_store(&store);
    println!("Store statistics:");
    println!("  total entries:        {}", stats.total_entries);
    println!("  dimensions:           {:?}", stats.dimensions);
    println!("  metric:               {}", stats.metric_name);
    println!("  avg vector magnitude: {:.4}", stats.avg_vector_magnitude);

    // Stats as JSON
    let stats_json = stats.to_json();
    println!("\nStats as JSON:");
    println!("  {}", serde_json::to_string_pretty(&stats_json).unwrap());

    // Stats for empty store
    let empty_store = InMemoryVectorStore::new(SimilarityMetric::Euclidean);
    let empty_stats = VectorStoreStats::from_store(&empty_store);
    println!("\nEmpty store stats:");
    println!("  total entries:        {}", empty_stats.total_entries);
    println!("  dimensions:           {:?}", empty_stats.dimensions);
    println!(
        "  avg vector magnitude: {:.4}",
        empty_stats.avg_vector_magnitude
    );

    // -----------------------------------------------------------------------
    // 7. Real LLM Demo — Q&A with vector store results
    // -----------------------------------------------------------------------
    println!("\n--- 7. Real LLM Demo ---");
    println!("Search the vector store, then ask an LLM to answer based on results.\n");

    let qa_query = SearchQuery::builder(vec![0.4, 0.8, 0.3]).top_k(2).build();
    let qa_results = store.search(&qa_query);

    let context: String = qa_results
        .iter()
        .map(|r| format!("- {}", r.entry.document))
        .collect::<Vec<_>>()
        .join("\n");

    let model = shared::get_chat_model(vec![
        "Based on the retrieved documents, the topics most related to your query are machine learning with neural networks and large language models with transformers. Both are subfields of artificial intelligence.".into(),
    ]);
    let messages = vec![
        Message::system("Answer the user's question using only the provided context documents."),
        Message::human(&format!("Context:\n{}\n\nQuestion: What topics are most related to AI in this collection?", context)),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("Retrieved context:\n{}\n", context);
        println!("LLM answer: {}", gen.message.content().text());
    }

    println!("\n=== Vector Store Example Complete ===");
    Ok(())
}
