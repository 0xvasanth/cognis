//! Distance metrics over embedding vectors.

use cognis_rag::{Distance, Embeddings, FakeEmbeddings};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    let emb = FakeEmbeddings::new(16);
    let a = emb.embed_query("rust programming".into()).await?;
    let b = emb.embed_query("rust language".into()).await?;
    let c = emb.embed_query("apple pie recipe".into()).await?;

    for metric in [Distance::Cosine, Distance::Euclidean, Distance::Dot] {
        println!("{metric:?}:");
        println!("  a~b: {:.4}", metric.similarity(&a, &b));
        println!("  a~c: {:.4}", metric.similarity(&a, &c));
    }
    Ok(())
}
