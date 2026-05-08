//! IndexingPipeline.run_incremental — track per-doc fingerprints via
//! a RecordManager so only changed docs get re-embedded.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_rag::loaders::{DocumentLoader, DocumentStream};
use cognis_rag::{
    CharacterSplitter, Document, Embeddings, FakeEmbeddings, InMemoryRecordManager,
    InMemoryVectorStore, IndexingPipeline,
};
use futures::stream;
use tokio::sync::RwLock;

struct VecLoader(Arc<Mutex<Vec<Document>>>);

#[async_trait]
impl DocumentLoader for VecLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let v = self.0.lock().unwrap().clone();
        Ok(Box::pin(stream::iter(v.into_iter().map(Ok))))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    let manager = InMemoryRecordManager::default();
    let docs = Arc::new(Mutex::new(vec![
        Document::new("doc one v1").with_id("a"),
        Document::new("doc two v1").with_id("b"),
    ]));
    let pipeline = IndexingPipeline::new(
        VecLoader(docs.clone()),
        CharacterSplitter::new().with_chunk_size(200),
        store.clone(),
    );

    let r1 = pipeline
        .run_incremental(&manager, "g1", |d| d.id.clone())
        .await?;
    println!(
        "first run : added={} changed={} unchanged={} deleted={}",
        r1.added, r1.changed, r1.unchanged, r1.deleted
    );

    *docs.lock().unwrap() = vec![
        Document::new("doc one v1").with_id("a"),
        Document::new("doc two v2 changed").with_id("b"),
        Document::new("doc three new").with_id("c"),
    ];
    let r2 = pipeline
        .run_incremental(&manager, "g1", |d| d.id.clone())
        .await?;
    println!(
        "second run: added={} changed={} unchanged={} deleted={}",
        r2.added, r2.changed, r2.unchanged, r2.deleted
    );
    Ok(())
}
