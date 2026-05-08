//! CachingRetriever — wrap any retriever to dedupe identical queries.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{
    CachingRetriever, Document, Embeddings, FakeEmbeddings, InMemoryVectorStore, VectorRetriever,
    VectorStore,
};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(16));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    {
        let mut s = store.write().await;
        s.add_texts(vec!["alpha".into(), "beta".into(), "gamma".into()], None).await?;
    }
    let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(VectorRetriever::new(store, 2));
    let cached = CachingRetriever::new(inner);
    let a = cached.invoke("alpha".into(), Default::default()).await?;
    let b = cached.invoke("alpha".into(), Default::default()).await?;
    println!("first call : {} hits", a.len());
    println!("cached call: {} hits", b.len());
    Ok(())
}
