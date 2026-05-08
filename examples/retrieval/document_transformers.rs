//! Document transformers — Dedup, Enrichment, MetadataTransformer.
//! All implement Runnable<Vec<Document>, Vec<Document>> so they
//! compose into RAG pipelines.

use cognis::prelude::*;
use cognis_rag::{Dedup, Document, Enrichment, MetadataTransformer};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let raw = vec![
        Document::new("Rust is fast."),
        Document::new("Tokio is async."),
        Document::new(" Rust is fast. "), // whitespace dup
        Document::new("Cargo manages crates."),
    ];

    // 1. Dedup by trimmed content.
    let after_dedup = Dedup::new().invoke(raw.clone(), Default::default()).await?;
    println!("--- after Dedup ---");
    for d in &after_dedup {
        println!("  - {}", d.content);
    }

    // 2. Enrich: uppercase content + tag with seen=true.
    let enriched = Enrichment::new(|d: &mut Document| {
        d.content = d.content.to_uppercase();
        d.metadata.insert("seen".into(), json!(true));
        Ok(())
    })
    .invoke(after_dedup, Default::default())
    .await?;
    println!("--- after Enrichment ---");
    for d in &enriched {
        println!("  - {} (meta: {:?})", d.content, d.metadata);
    }

    // 3. Stamp provenance via MetadataTransformer.
    let final_docs = MetadataTransformer::new()
        .set("source", "demo-pipeline")
        .set("stage", "preprocess")
        .invoke(enriched, Default::default())
        .await?;
    println!("--- after MetadataTransformer ---");
    for d in &final_docs {
        println!(
            "  - {} (source={} stage={})",
            d.content,
            d.metadata.get("source").unwrap(),
            d.metadata.get("stage").unwrap()
        );
    }
    Ok(())
}
