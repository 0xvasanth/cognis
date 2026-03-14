//! Embedding Similarity Example
//!
//! Embed texts as vectors, compute pairwise similarity, and find nearest neighbors.
//! Uses synthetic embeddings (no API keys required).

#[path = "../shared.rs"]
mod shared;

use cognis::embeddings::distance::DistanceMetric;
use cognis::embeddings::similarity::{
    EmbeddingSimilarity, KNearestNeighbors, PairwiseSimilarityMatrix,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic embeddings representing different topics.
    let embeddings = vec![
        vec![0.9, 0.1, 0.0, 0.8], // Rust programming
        vec![0.1, 0.9, 0.0, 0.7], // Python programming
        vec![0.0, 0.8, 0.9, 0.2], // Machine learning
        vec![0.8, 0.0, 0.1, 0.9], // Systems programming
        vec![0.2, 0.7, 0.8, 0.3], // Data science
    ];
    let labels = ["Rust", "Python", "ML", "Systems", "Data Science"];

    // --- Compare two embeddings ---
    let sim = EmbeddingSimilarity::new(DistanceMetric::Cosine);
    println!("Cosine similarity:");
    println!(
        "  Rust vs Systems:  {:.4}",
        sim.similarity(&embeddings[0], &embeddings[3])
    );
    println!(
        "  Rust vs Python:   {:.4}",
        sim.similarity(&embeddings[0], &embeddings[1])
    );
    println!(
        "  Python vs ML:     {:.4}",
        sim.similarity(&embeddings[1], &embeddings[2])
    );

    // --- Pairwise similarity matrix ---
    let matrix = PairwiseSimilarityMatrix::compute_symmetric(&embeddings, DistanceMetric::Cosine);

    println!("\nPairwise similarity matrix:");
    print!("{:>14}", "");
    for l in &labels {
        print!("{:>10}", l);
    }
    println!();
    for (i, label) in labels.iter().enumerate() {
        print!("{:>14}", label);
        for j in 0..labels.len() {
            print!("{:>10.4}", matrix.get(i, j));
        }
        println!();
    }

    // Most similar pair per document (excluding self).
    println!("\nNearest neighbor per document:");
    let most_similar = matrix.most_similar_per_row();
    for (i, (j, score)) in most_similar.iter().enumerate() {
        if labels[i] != labels[*j] {
            println!("  {} -> {} ({:.4})", labels[i], labels[*j], score);
        }
    }

    // --- K-nearest-neighbor search ---
    let knn = KNearestNeighbors::new(embeddings.clone(), DistanceMetric::Cosine);
    let query = vec![0.85, 0.15, 0.05, 0.75]; // Rust-like query

    println!("\nTop 3 neighbors for a Rust-like query:");
    for r in knn.search(&query, 3) {
        println!("  {} (score={:.4})", labels[r.index], r.score);
    }

    // --- Show that a chat model can generate text for embedding ---
    let _model = shared::get_chat_model(vec![
        "Embeddings turn text into vectors for similarity search.".into(),
    ]);

    Ok(())
}
