//! CrossEncoder — score (query, doc) pairs in bulk. FnCrossEncoder
//! takes a per-doc closure and runs the batch concurrently.

use cognis_rag::{CrossEncoder, Document, FnCrossEncoder};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    let encoder = FnCrossEncoder {
        f: |q: &str, d: &Document| {
            let qw: std::collections::HashSet<_> = q.split_whitespace().collect();
            let dw: std::collections::HashSet<_> = d.content.split_whitespace().collect();
            qw.intersection(&dw).count() as f32
        },
    };
    let docs = vec![
        Document::new("tokio is an async runtime for rust"),
        Document::new("go has goroutines"),
        Document::new("rust runtime tokio"),
    ];
    let scores = encoder.score("rust async runtime", &docs).await?;
    for (s, d) in scores.iter().zip(docs.iter()) {
        println!("{s:.2}  {}", d.content);
    }
    Ok(())
}
