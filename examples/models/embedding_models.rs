//! Embedding Models Example
//!
//! Demonstrates creating embeddings, comparing them with distance metrics,
//! and using a model registry -- all without API keys.
//!
//! Run with: `cargo run -p cognis-examples --example embedding_models`

#[path = "../shared.rs"]
mod shared;

use cognis::embeddings::models::{
    EmbeddingDistance, EmbeddingModel, EmbeddingRegistry, FakeEmbeddingModel, NormalizedEmbedding,
};
use cognis_core::language_models::chat_model::BaseChatModel;

fn main() {
    println!("=== Embedding Models Example ===\n");

    // --- Create a fake embedding model (deterministic, no API keys) ---
    let model = FakeEmbeddingModel::new(8);

    // Embed a batch of texts
    let texts = vec![
        "machine learning",
        "deep learning",
        "cooking recipes",
        "baking bread",
    ];
    let embeddings = model.embed_batch(&texts).unwrap();
    println!(
        "Embedded {} texts into {}-dim vectors",
        texts.len(),
        model.dimensions()
    );

    // --- Compare embeddings with distance metrics ---
    let query = &embeddings[0]; // "machine learning"
    println!("\nSimilarity to \"machine learning\":");
    for (i, text) in texts.iter().enumerate().skip(1) {
        let cosine = EmbeddingDistance::cosine_similarity(query, &embeddings[i]);
        let euclidean = EmbeddingDistance::euclidean_distance(query, &embeddings[i]);
        println!("  \"{text}\": cosine={cosine:.4}, euclidean={euclidean:.4}");
    }

    // Top-K search: find the 2 most similar texts to the query
    let candidates: Vec<Vec<f64>> = embeddings[1..].to_vec();
    let top2 = EmbeddingDistance::most_similar(query, &candidates, 2);
    println!("\nTop 2 most similar to \"machine learning\":");
    for (idx, score) in &top2 {
        println!("  \"{}\" (score: {score:.4})", texts[idx + 1]);
    }

    // --- Normalized embeddings (dot product == cosine similarity) ---
    let norm_a = NormalizedEmbedding::from_vec(vec![3.0, 4.0]);
    let norm_b = NormalizedEmbedding::from_vec(vec![1.0, 2.0]);
    let dot = norm_a.dot_product(&norm_b);
    let cosine = EmbeddingDistance::cosine_similarity(&[3.0, 4.0], &[1.0, 2.0]);
    println!("\nNormalized dot product: {dot:.6}");
    println!("Raw cosine similarity:  {cosine:.6}");
    println!("Match: {}", (dot - cosine).abs() < 1e-10);

    // --- Model registry ---
    let mut registry = EmbeddingRegistry::new();
    registry.register("small", Box::new(FakeEmbeddingModel::new(64)));
    registry.register("large", Box::new(FakeEmbeddingModel::new(1024)));

    let small = registry.get("small").unwrap();
    let vec = small.embed_text("hello").unwrap();
    println!(
        "\nRegistry lookup \"small\": {} dims, embedded to {} values",
        small.dimensions(),
        vec.len()
    );

    // --- Chat model + embeddings together ---
    let chat_model = shared::get_chat_model(vec![
        "Text embeddings map words into vector space where similar meanings are nearby.".into(),
    ]);

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a helpful AI teacher.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            "What are text embeddings in one sentence?",
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(async { chat_model.invoke_messages(&messages, None).await }) {
        Ok(response) => println!("\nLLM: {}", response.base.content.text()),
        Err(e) => println!("\nLLM error: {e}"),
    }

    println!("\n=== Done ===");
}
