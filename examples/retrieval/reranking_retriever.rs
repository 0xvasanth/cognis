//! Two-stage retrieval: vector recall + CrossEncoderReranker.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{
    CrossEncoder, CrossEncoderReranker, Document, Embeddings, FakeEmbeddings, FnCrossEncoder,
    InMemoryVectorStore, VectorRetriever, VectorStore,
};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(16));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    {
        let mut s = store.write().await;
        s.add_texts(
            vec![
                "Rust async runtime tokio".into(),
                "Cargo manages crates".into(),
                "Tokio futures are zero-cost".into(),
                "Python asyncio is comparable".into(),
            ],
            None,
        )
        .await?;
    }
    let recall: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(VectorRetriever::new(store, 4));
    let encoder: Arc<dyn CrossEncoder> = Arc::new(FnCrossEncoder {
        f: |q: &str, d: &Document| {
            q.split_whitespace()
                .filter(|w| d.content.contains(w))
                .count() as f32
        },
    });
    let reranker = CrossEncoderReranker::new(recall, encoder, 2);
    let docs = reranker
        .invoke("rust async tokio".into(), Default::default())
        .await?;
    for d in docs {
        println!("- {}", d.content);
    }
    Ok(())
}
