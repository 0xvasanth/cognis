//! Embedding Models Example
//!
//! Demonstrates the embedding model abstractions provided by `rustchain::embeddings::models`,
//! including deterministic fake embeddings for testing, configuration, distance metrics,
//! normalized embeddings, and a model registry.
//!
//! Features shown:
//! - FakeEmbeddingModel creation and single-text embedding
//! - Batch embedding of multiple texts
//! - EmbeddingModelConfig builder pattern
//! - EmbeddingDistance: cosine similarity, euclidean distance, dot product, manhattan distance
//! - most_similar for finding top-K closest embeddings
//! - NormalizedEmbedding creation and dot-product-as-cosine property
//! - EmbeddingRegistry for registering and looking up models
//! - EmbeddingResult and EmbeddingUsage metadata types
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example embedding_models`

use rustchain::embeddings::models::{
    EmbeddingDistance, EmbeddingModel, EmbeddingModelConfig, EmbeddingRegistry, EmbeddingResult,
    EmbeddingUsage, FakeEmbeddingModel, NormalizedEmbedding,
};

fn main() {
    println!("=== Embedding Models Example ===\n");

    // -----------------------------------------------------------------------
    // 1. FakeEmbeddingModel — Creation and Single-Text Embedding
    // -----------------------------------------------------------------------
    println!("--- 1. FakeEmbeddingModel ---");
    println!("A deterministic embedding model useful for testing (no API keys needed).\n");

    let model = FakeEmbeddingModel::new(8);
    println!("Model name: {}", model.model_name());
    println!("Dimensions: {}", model.dimensions());

    let embedding = model.embed_text("hello world").unwrap();
    println!("Embedding for \"hello world\": {:?}", embedding);

    // Determinism: same input always produces the same output.
    let embedding2 = model.embed_text("hello world").unwrap();
    assert_eq!(embedding, embedding2);
    println!("Deterministic: re-embedding produces identical vector");

    // Different inputs produce different embeddings.
    let other = model.embed_text("goodbye world").unwrap();
    assert_ne!(embedding, other);
    println!("Different text produces a different vector");

    // -----------------------------------------------------------------------
    // 2. Batch Embedding
    // -----------------------------------------------------------------------
    println!("\n--- 2. Batch Embedding ---");
    println!("Embed multiple texts in one call.\n");

    let texts = vec!["rust", "python", "javascript", "go"];
    let batch_results = model.embed_batch(&texts).unwrap();
    println!(
        "Batch-embedded {} texts, each with {} dimensions",
        batch_results.len(),
        model.dimensions()
    );
    for (text, vec) in texts.iter().zip(batch_results.iter()) {
        println!("  \"{}\": [{:.4}, {:.4}, ...]", text, vec[0], vec[1]);
    }

    // Batch results match individual embeddings.
    let single_rust = model.embed_text("rust").unwrap();
    assert_eq!(batch_results[0], single_rust);
    println!("Batch result matches single embed_text call");

    // -----------------------------------------------------------------------
    // 3. EmbeddingModelConfig Builder
    // -----------------------------------------------------------------------
    println!("\n--- 3. EmbeddingModelConfig ---");
    println!("Configure embedding model parameters with a builder pattern.\n");

    let config = EmbeddingModelConfig::new("text-embedding-3-small", 1536)
        .with_batch_size(50)
        .with_normalize(true)
        .with_timeout_ms(30000);

    println!("Config:");
    println!("  model_name:  {}", config.model_name);
    println!("  dimensions:  {}", config.dimensions);
    println!("  batch_size:  {}", config.batch_size);
    println!("  normalize:   {}", config.normalize);
    println!("  timeout_ms:  {:?}", config.timeout_ms);

    // Defaults
    let defaults = EmbeddingModelConfig::new("default-model", 768);
    println!("\nDefaults:");
    println!("  batch_size:  {} (default)", defaults.batch_size);
    println!("  normalize:   {} (default)", defaults.normalize);
    println!("  timeout_ms:  {:?} (default)", defaults.timeout_ms);

    // -----------------------------------------------------------------------
    // 4. EmbeddingDistance — Distance and Similarity Metrics
    // -----------------------------------------------------------------------
    println!("\n--- 4. EmbeddingDistance ---");
    println!("Compute distances and similarities between embedding vectors.\n");

    let vec_a = vec![1.0, 0.0, 0.0];
    let vec_b = vec![0.0, 1.0, 0.0];
    let vec_c = vec![0.9, 0.1, 0.0];

    // Cosine similarity
    let cos_ab = EmbeddingDistance::cosine_similarity(&vec_a, &vec_b);
    let cos_ac = EmbeddingDistance::cosine_similarity(&vec_a, &vec_c);
    let cos_aa = EmbeddingDistance::cosine_similarity(&vec_a, &vec_a);
    println!("Cosine similarity:");
    println!("  a vs a (identical):   {:.6}", cos_aa);
    println!("  a vs b (orthogonal):  {:.6}", cos_ab);
    println!("  a vs c (close to a):  {:.6}", cos_ac);

    // Euclidean distance
    let euc_ab = EmbeddingDistance::euclidean_distance(&vec_a, &vec_b);
    let euc_ac = EmbeddingDistance::euclidean_distance(&vec_a, &vec_c);
    println!("\nEuclidean distance:");
    println!("  a vs b: {:.6}", euc_ab);
    println!("  a vs c: {:.6}", euc_ac);

    // Dot product
    let dot_ab = EmbeddingDistance::dot_product(&vec_a, &vec_b);
    let dot_ac = EmbeddingDistance::dot_product(&vec_a, &vec_c);
    println!("\nDot product:");
    println!("  a . b: {:.6}", dot_ab);
    println!("  a . c: {:.6}", dot_ac);

    // Manhattan distance
    let man_ab = EmbeddingDistance::manhattan_distance(&vec_a, &vec_b);
    let man_ac = EmbeddingDistance::manhattan_distance(&vec_a, &vec_c);
    println!("\nManhattan distance:");
    println!("  a vs b: {:.6}", man_ab);
    println!("  a vs c: {:.6}", man_ac);

    // -----------------------------------------------------------------------
    // 5. most_similar — Top-K Search
    // -----------------------------------------------------------------------
    println!("\n--- 5. most_similar (Top-K) ---");
    println!("Find the most similar vectors to a query from a candidate set.\n");

    let query = vec![1.0, 0.0, 0.0];
    let candidates = vec![
        vec![0.0, 1.0, 0.0],   // orthogonal
        vec![0.99, 0.01, 0.0], // very close
        vec![0.5, 0.5, 0.0],   // moderate
        vec![-1.0, 0.0, 0.0],  // opposite
        vec![0.9, 0.1, 0.0],   // close
    ];
    let labels = ["orthogonal", "very close", "moderate", "opposite", "close"];

    let top3 = EmbeddingDistance::most_similar(&query, &candidates, 3);
    println!("Query: {:?}", query);
    println!("Top 3 most similar:");
    for (idx, similarity) in &top3 {
        println!(
            "  #{}: \"{}\" (index {}) — cosine similarity: {:.6}",
            top3.iter().position(|(i, _)| i == idx).unwrap() + 1,
            labels[*idx],
            idx,
            similarity
        );
    }

    // -----------------------------------------------------------------------
    // 6. NormalizedEmbedding
    // -----------------------------------------------------------------------
    println!("\n--- 6. NormalizedEmbedding ---");
    println!("L2-normalized wrapper where dot product equals cosine similarity.\n");

    let norm_a = NormalizedEmbedding::from_vec(vec![3.0, 4.0]);
    println!("Input:      [3.0, 4.0]");
    println!("Normalized: {:?}", norm_a.as_slice());
    println!("Dimensions: {}", norm_a.dimensions());

    // Verify unit length
    let magnitude: f64 = norm_a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("Magnitude:  {:.10} (unit length)", magnitude);

    // Dot product of normalized vectors equals cosine similarity
    let norm_b = NormalizedEmbedding::from_vec(vec![1.0, 2.0]);
    let dp = norm_a.dot_product(&norm_b);
    let cosine = EmbeddingDistance::cosine_similarity(&[3.0, 4.0], &[1.0, 2.0]);
    println!("\nDot product of normalized vectors: {:.10}", dp);
    println!("Cosine similarity of raw vectors:  {:.10}", cosine);
    println!("Equal (within epsilon): {}", (dp - cosine).abs() < 1e-10);

    // Zero vector stays zero
    let zero = NormalizedEmbedding::from_vec(vec![0.0, 0.0, 0.0]);
    println!("\nZero vector normalized: {:?}", zero.as_slice());

    // -----------------------------------------------------------------------
    // 7. EmbeddingRegistry
    // -----------------------------------------------------------------------
    println!("\n--- 7. EmbeddingRegistry ---");
    println!("Register and look up embedding models by name.\n");

    let mut registry = EmbeddingRegistry::new();
    println!(
        "Empty registry: len={}, is_empty={}",
        registry.len(),
        registry.is_empty()
    );

    // Register models
    registry.register("small", Box::new(FakeEmbeddingModel::new(64)));
    registry.register("medium", Box::new(FakeEmbeddingModel::new(256)));
    registry.register("large", Box::new(FakeEmbeddingModel::new(1024)));
    println!("\nRegistered 3 models");
    println!("Registry len: {}", registry.len());

    let mut names = registry.model_names();
    names.sort();
    println!("Model names: {:?}", names);

    // Look up and use a model
    let medium = registry.get("medium").unwrap();
    println!(
        "\nLooked up 'medium': {} ({} dims)",
        medium.model_name(),
        medium.dimensions()
    );
    let vec = medium.embed_text("test").unwrap();
    println!(
        "Embedded \"test\" via registry: vector of {} dimensions",
        vec.len()
    );

    // Missing model returns None
    let missing = registry.get("nonexistent");
    println!(
        "Looking up 'nonexistent': {:?}",
        missing.map(|m| m.model_name())
    );

    // -----------------------------------------------------------------------
    // 8. EmbeddingResult and EmbeddingUsage
    // -----------------------------------------------------------------------
    println!("\n--- 8. EmbeddingResult and EmbeddingUsage ---");
    println!("Structured result type with metadata and optional token usage.\n");

    // Without usage
    let result = EmbeddingResult::new(
        vec![0.1, 0.2, 0.3],
        "sample text",
        "fake-embedding-model",
        None,
    );
    println!("EmbeddingResult:");
    println!("  text:            \"{}\"", result.text);
    println!("  model:           {}", result.model);
    println!("  dimension_count: {}", result.dimension_count());
    println!("  usage:           {:?}", result.usage);

    // With usage
    let usage = EmbeddingUsage::new(12, 12);
    println!("\nEmbeddingUsage:");
    println!("  prompt_tokens: {}", usage.prompt_tokens);
    println!("  total_tokens:  {}", usage.total_tokens);
    println!("  JSON: {}", usage.to_json());

    let result_with_usage = EmbeddingResult::new(
        vec![0.4, 0.5, 0.6],
        "another text",
        "text-embedding-3-small",
        Some(EmbeddingUsage::new(8, 8)),
    );
    println!("\nEmbeddingResult with usage:");
    println!(
        "  JSON: {}",
        serde_json::to_string_pretty(&result_with_usage.to_json()).unwrap()
    );

    println!("\n=== Embedding Models Example Complete ===");
}
