//! Embedding Similarity Example
//!
//! Demonstrates embedding similarity utilities from `cognis::embeddings::similarity`.
//! Shows how to compute similarities with different distance metrics, build pairwise
//! matrices, perform k-nearest-neighbor search, cluster embeddings, normalize vectors,
//! reduce dimensionality, and filter by similarity thresholds.
//!
//! No API keys required -- uses synthetic embedding vectors.

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognis::embeddings::distance::DistanceMetric;
use cognis::embeddings::similarity::{
    ClusterAssignment, DimensionalityReducer, EmbeddingNormalizer, EmbeddingSimilarity,
    KNearestNeighbors, PairwiseSimilarityMatrix, SimilarityThreshold,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Embedding Similarity Example ===\n");

    // Sample embeddings representing short "document" vectors.
    // In practice these would come from an embedding model.
    let doc_rust = vec![0.9, 0.1, 0.0, 0.8]; // "Rust programming"
    let doc_python = vec![0.1, 0.9, 0.0, 0.7]; // "Python programming"
    let doc_ml = vec![0.0, 0.8, 0.9, 0.2]; // "Machine learning"
    let doc_systems = vec![0.8, 0.0, 0.1, 0.9]; // "Systems programming"
    let doc_data = vec![0.2, 0.7, 0.8, 0.3]; // "Data science"

    let all_docs = vec![
        doc_rust.clone(),
        doc_python.clone(),
        doc_ml.clone(),
        doc_systems.clone(),
        doc_data.clone(),
    ];
    let labels = ["Rust", "Python", "ML", "Systems", "Data Science"];

    // -----------------------------------------------------------------------
    // 1. EmbeddingSimilarity — compute similarity with different metrics
    // -----------------------------------------------------------------------
    println!("--- 1. EmbeddingSimilarity with Different DistanceMetrics ---\n");

    let metrics_to_try = [
        ("Cosine", DistanceMetric::Cosine),
        ("Euclidean", DistanceMetric::Euclidean),
        ("DotProduct", DistanceMetric::DotProduct),
        ("Manhattan", DistanceMetric::Manhattan),
    ];

    for (name, metric) in &metrics_to_try {
        let calc = EmbeddingSimilarity::new(*metric);
        let sim = calc.similarity(&doc_rust, &doc_python);
        let dist = calc.distance(&doc_rust, &doc_python);
        println!(
            "  {:<12} Rust vs Python: similarity={:.4}, distance={:.4}",
            name, sim, dist
        );
    }

    println!();

    // score_all: rank all docs against a query.
    let cosine_calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
    println!("Ranking all docs against 'Rust' (cosine):");
    let results = cosine_calc.score_all(&doc_rust, &all_docs);
    for r in &results {
        println!(
            "  rank={}, index={} ({}), score={:.4}",
            r.rank, r.index, labels[r.index], r.score
        );
    }

    // -----------------------------------------------------------------------
    // 2. PairwiseSimilarityMatrix — NxM or symmetric similarity
    // -----------------------------------------------------------------------
    println!("\n--- 2. PairwiseSimilarityMatrix ---\n");

    // Symmetric matrix for all documents.
    let sym_matrix =
        PairwiseSimilarityMatrix::compute_symmetric(&all_docs, DistanceMetric::Cosine);

    println!(
        "Symmetric pairwise matrix ({}x{}):",
        sym_matrix.rows, sym_matrix.cols
    );
    print!("{:>14}", "");
    for label in &labels {
        print!("{:>10}", label);
    }
    println!();
    for (i, label) in labels.iter().enumerate() {
        print!("{:>14}", label);
        for j in 0..labels.len() {
            print!("{:>10.4}", sym_matrix.get(i, j));
        }
        println!();
    }

    // Most similar per row.
    println!("\nMost similar document for each:");
    let most_similar = sym_matrix.most_similar_per_row();
    for (i, (best_j, score)) in most_similar.iter().enumerate() {
        // Skip self-similarity (the most similar is always itself with score ~1.0).
        // In a real scenario you might filter self-matches.
        println!(
            "  {} -> {} (score={:.4})",
            labels[i], labels[*best_j], score
        );
    }

    // Asymmetric: compare a subset (queries) against another subset (corpus).
    let queries = vec![doc_rust.clone(), doc_ml.clone()];
    let corpus = vec![doc_python.clone(), doc_systems.clone(), doc_data.clone()];
    let asym_matrix =
        PairwiseSimilarityMatrix::compute(&queries, &corpus, DistanceMetric::Cosine);

    println!(
        "\nAsymmetric matrix ({}x{}) — queries vs corpus:",
        asym_matrix.rows, asym_matrix.cols
    );
    let query_labels = ["Rust", "ML"];
    let corpus_labels = ["Python", "Systems", "Data Science"];
    print!("{:>14}", "");
    for label in &corpus_labels {
        print!("{:>14}", label);
    }
    println!();
    for (i, ql) in query_labels.iter().enumerate() {
        print!("{:>14}", ql);
        for j in 0..corpus_labels.len() {
            print!("{:>14.4}", asym_matrix.get(i, j));
        }
        println!();
    }

    // most_similar_per_col: best query for each corpus doc.
    let best_per_col = asym_matrix.most_similar_per_col();
    println!("\nBest query match for each corpus doc:");
    for (j, (best_i, score)) in best_per_col.iter().enumerate() {
        println!(
            "  {} -> {} (score={:.4})",
            corpus_labels[j], query_labels[*best_i], score
        );
    }

    // -----------------------------------------------------------------------
    // 3. KNearestNeighbors — top-k search
    // -----------------------------------------------------------------------
    println!("\n--- 3. KNearestNeighbors ---\n");

    let knn = KNearestNeighbors::new(all_docs.clone(), DistanceMetric::Cosine);
    println!("KNN index size: {}", knn.len());

    let query = vec![0.85, 0.15, 0.05, 0.75]; // Similar to "Rust"
    let top3 = knn.search(&query, 3);
    println!("Top 3 nearest neighbors to a Rust-like query:");
    for r in &top3 {
        println!(
            "  rank={}, {} (index={}), score={:.4}",
            r.rank, labels[r.index], r.index, r.score
        );
    }

    // Search with metadata.
    let metadata: Vec<HashMap<String, String>> = labels
        .iter()
        .map(|&label| {
            let mut m = HashMap::new();
            m.insert("topic".to_string(), label.to_string());
            m
        })
        .collect();

    let top2_meta = knn.search_with_metadata(&query, 2, &metadata);
    println!("\nTop 2 with metadata:");
    for r in &top2_meta {
        println!(
            "  rank={}, score={:.4}, metadata={:?}",
            r.rank, r.score, r.metadata
        );
    }

    // -----------------------------------------------------------------------
    // 4. ClusterAssignment — centroid-based clustering
    // -----------------------------------------------------------------------
    println!("\n--- 4. ClusterAssignment ---\n");

    // Use 2 initial centroids: one near "programming", one near "data/ML".
    let centroids = vec![
        vec![0.85, 0.05, 0.05, 0.85], // programming cluster
        vec![0.1, 0.75, 0.85, 0.25],   // data/ML cluster
    ];

    let clustering =
        ClusterAssignment::assign(&all_docs, &centroids, DistanceMetric::Cosine);
    println!("Clusters (k={}):", clustering.k);
    for (i, &cluster) in clustering.assignments.iter().enumerate() {
        println!("  {} -> cluster {}", labels[i], cluster);
    }

    let sizes = clustering.cluster_sizes();
    println!("\nCluster sizes: {:?}", sizes);

    println!("Members of cluster 0 (programming):");
    for idx in clustering.members(0) {
        println!("  - {}", labels[idx]);
    }
    println!("Members of cluster 1 (data/ML):");
    for idx in clustering.members(1) {
        println!("  - {}", labels[idx]);
    }

    println!("\nRecomputed centroids:");
    for (i, centroid) in clustering.centroids.iter().enumerate() {
        println!("  cluster {}: {:?}", i, centroid);
    }

    // -----------------------------------------------------------------------
    // 5. EmbeddingNormalizer — L2, mean-center, unit-variance
    // -----------------------------------------------------------------------
    println!("\n--- 5. EmbeddingNormalizer ---\n");

    // L2 normalize a single vector.
    let raw = vec![3.0, 4.0, 0.0];
    let normalized = EmbeddingNormalizer::l2_normalize(&raw);
    let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("L2 normalize {:?} -> {:?} (magnitude={:.4})", raw, normalized, magnitude);

    // L2 normalize a batch.
    let batch = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 3.0, 4.0],
    ];
    let norm_batch = EmbeddingNormalizer::l2_normalize_batch(&batch);
    println!("\nL2 normalize batch:");
    for (orig, normed) in batch.iter().zip(norm_batch.iter()) {
        println!("  {:?} -> {:?}", orig, normed);
    }

    // Mean-center.
    let vectors = vec![
        vec![2.0, 4.0],
        vec![4.0, 6.0],
        vec![6.0, 8.0],
    ];
    let centered = EmbeddingNormalizer::mean_center(&vectors);
    println!("\nMean-centered:");
    for (orig, cent) in vectors.iter().zip(centered.iter()) {
        println!("  {:?} -> {:?}", orig, cent);
    }

    // Unit variance.
    let scaled = EmbeddingNormalizer::unit_variance(&vectors);
    println!("\nUnit-variance scaled:");
    for (orig, sc) in vectors.iter().zip(scaled.iter()) {
        println!("  {:?} -> {:?}", orig, sc);
    }

    // Full normalization: mean-center then L2 normalize.
    let full = EmbeddingNormalizer::normalize_full(&vectors);
    println!("\nFull normalization (mean-center + L2):");
    for (orig, f) in vectors.iter().zip(full.iter()) {
        let mag: f32 = f.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  {:?} -> {:?} (mag={:.4})", orig, f, mag);
    }

    // -----------------------------------------------------------------------
    // 6. DimensionalityReducer — random projection
    // -----------------------------------------------------------------------
    println!("\n--- 6. DimensionalityReducer ---\n");

    let reducer = DimensionalityReducer::new(4, 2, 42);
    println!(
        "Reducer: {}D -> {}D (random projection)",
        reducer.input_dim, reducer.output_dim
    );

    let projected = reducer.project(&doc_rust);
    println!("Rust embedding ({:?}) -> {:?}", doc_rust, projected);

    let projected_batch = reducer.project_batch(&all_docs);
    println!("\nProjected all docs to 2D:");
    for (i, (orig, proj)) in all_docs.iter().zip(projected_batch.iter()).enumerate() {
        println!("  {} {:?} -> {:?}", labels[i], orig, proj);
    }

    // Verify distances are approximately preserved.
    let orig_sim = cosine_calc.similarity(&doc_rust, &doc_python);
    let proj_calc = EmbeddingSimilarity::new(DistanceMetric::Cosine);
    let proj_sim = proj_calc.similarity(&projected_batch[0], &projected_batch[1]);
    println!(
        "\nRust vs Python similarity: original={:.4}, projected={:.4} (approximate preservation)",
        orig_sim, proj_sim
    );

    // -----------------------------------------------------------------------
    // 7. SimilarityThreshold — filter by minimum score
    // -----------------------------------------------------------------------
    println!("\n--- 7. SimilarityThreshold ---\n");

    let threshold = SimilarityThreshold::new(0.7);

    // Use the convenience search method.
    let query = doc_rust.clone();
    let filtered = threshold.search(&query, &all_docs, DistanceMetric::Cosine);
    println!(
        "Documents with cosine similarity >= 0.7 to 'Rust' ({} of {}):",
        filtered.len(),
        all_docs.len()
    );
    for r in &filtered {
        println!(
            "  rank={}, {} (index={}), score={:.4}",
            r.rank, labels[r.index], r.index, r.score
        );
    }

    // Manual filter + rerank flow.
    let all_results = cosine_calc.score_all(&doc_ml, &all_docs);
    let strict_threshold = SimilarityThreshold::new(0.85);
    let strict_filtered = strict_threshold.filter_and_rerank(all_results);
    println!(
        "\nDocuments with cosine similarity >= 0.85 to 'ML' ({} results):",
        strict_filtered.len()
    );
    for r in &strict_filtered {
        println!(
            "  rank={}, {} (index={}), score={:.4}",
            r.rank, labels[r.index], r.index, r.score
        );
    }

    // -----------------------------------------------------------------------
    // 8. Context: using with a chat model (via shared helper)
    // -----------------------------------------------------------------------
    println!("\n--- 8. Context with shared::get_chat_model() ---\n");

    let _model = shared::get_chat_model(vec![
        "Embeddings are dense vector representations of text.".into(),
    ]);
    println!(
        "A chat model could generate text to embed, then use these similarity \
         utilities to find related documents, cluster them, or reduce dimensions \
         for visualization."
    );

    println!("\nDone!");
    Ok(())
}
