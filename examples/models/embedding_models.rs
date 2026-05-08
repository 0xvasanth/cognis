//! V2 Embeddings: FakeEmbeddings (always available), OllamaEmbeddings,
//! OpenAIEmbeddings, GoogleEmbeddings, VoyageEmbeddings (all behind
//! their feature flags). All implement the same Embeddings trait.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_rag::{Embeddings, FakeEmbeddings};

#[tokio::main]
async fn main() -> Result<()> {
    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
    let v = emb.embed_query("hello world".into()).await?;
    println!(
        "{} produced {}-dim embedding: {:?}",
        emb.model(),
        v.len(),
        &v[..4]
    );

    let batch = emb
        .embed_documents(vec!["a".into(), "b".into(), "c".into()])
        .await?;
    println!("batch of 3 → {} vectors", batch.len());
    Ok(())
}
