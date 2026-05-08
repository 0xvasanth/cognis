//! What you'll learn:
//!   How `IndexingPipeline::run_incremental` uses a `RecordManager`
//!   to track per-document fingerprints, so a second pass only
//!   re-embeds the docs that actually changed since last time.
//!
//! Why this matters:
//!   Re-embedding an entire corpus on every reindex is wasteful and
//!   slow. Your docs change daily — but most pages don't change in
//!   any given day. The record-manager pattern is how production RAG
//!   systems keep their indexes fresh on a schedule without paying
//!   for embeddings on unchanged content.
//!
//! Scenario:
//!   You're maintaining a docs site with three published pages.
//!   Round 1: index all three from scratch. Then someone edits one
//!   page. Round 2: re-run the indexer — it should report exactly
//!   one `changed`, two `unchanged`, and zero new embeddings paid
//!   for the unchanged pages.
//!
//! Run with:
//!   cargo run -p cognis-examples --example retrieval_indexing_rag
//!
//! Sample output (against ollama / llama3.1):
//!   === round 1 (initial index) ===
//!   added=3 changed=0 unchanged=0 deleted=0
//!
//!   === round 2 (after editing 'tools' page) ===
//!   added=0 changed=1 unchanged=2 deleted=0
//!   (only the edited page re-embedded; you didn't pay for the other 2 pages)

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

/// Loader that reads from a shared `Vec<Document>` we can edit
/// between reindex calls — a stand-in for "files on disk" or
/// "rows in a CMS".
struct DocsSource(Arc<Mutex<Vec<Document>>>);

#[async_trait]
impl DocumentLoader for DocsSource {
    async fn load(&self) -> Result<DocumentStream> {
        let snapshot = self.0.lock().unwrap().clone();
        Ok(Box::pin(stream::iter(snapshot.into_iter().map(Ok))))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(16));
    let store = Arc::new(RwLock::new(InMemoryVectorStore::new(emb)));
    let manager = InMemoryRecordManager::default();

    // Round 1: three docs from scratch.
    let docs = Arc::new(Mutex::new(vec![
        Document::new("Getting started with Cognis: install, build, run.")
            .with_id("getting-started"),
        Document::new("Cognis tools: Calculator, ShellTool, JsonQueryTool.").with_id("tools"),
        Document::new("Cognis memory: Buffer, Window, SummaryBufferMemory.").with_id("memory"),
    ]));

    let pipeline = IndexingPipeline::new(
        DocsSource(docs.clone()),
        CharacterSplitter::new().with_chunk_size(200),
        store.clone(),
    );

    let r1 = pipeline
        .run_incremental(&manager, "docs", |d| d.id.clone())
        .await?;
    println!("=== round 1 (initial index) ===");
    println!(
        "added={} changed={} unchanged={} deleted={}",
        r1.added, r1.changed, r1.unchanged, r1.deleted
    );

    // Edit one page, leave the other two untouched. This is exactly
    // what your nightly indexer would see after a normal docs day.
    {
        let mut d = docs.lock().unwrap();
        d[1] = Document::new(
            "Cognis tools: Calculator, ShellTool, JsonQueryTool. \
             Updated 2026-05 with PythonReplTool example.",
        )
        .with_id("tools");
    }

    let r2 = pipeline
        .run_incremental(&manager, "docs", |d| d.id.clone())
        .await?;
    println!("\n=== round 2 (after editing 'tools' page) ===");
    println!(
        "added={} changed={} unchanged={} deleted={}",
        r2.added, r2.changed, r2.unchanged, r2.deleted
    );
    println!(
        "(only the edited page re-embedded; you didn't pay for the other {} pages)",
        r2.unchanged
    );
    Ok(())
}
