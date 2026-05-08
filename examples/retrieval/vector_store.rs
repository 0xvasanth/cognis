//! InMemoryVectorStore round-trip with FakeEmbeddings.

use std::sync::Arc;

use cognis_rag::{Embeddings, FakeEmbeddings, InMemoryVectorStore, VectorStore};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(16));
    let mut store = InMemoryVectorStore::new(emb);
    let ids = store
        .add_texts(
            vec![
                "Rust is a systems language.".into(),
                "Tokio is an async runtime.".into(),
                "Cargo is the package manager.".into(),
            ],
            None,
        )
        .await?;
    println!("inserted ids: {ids:?}");

    let hits = store
        .similarity_search("which crate manages packages?", 2)
        .await?;
    for h in hits {
        println!("{:.3}  {}  ({})", h.score, h.text, h.id);
    }
    Ok(())
}
