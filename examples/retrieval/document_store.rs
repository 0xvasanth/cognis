//! InMemoryDocstore — a minimal id-keyed `Document` store.

use cognis_rag::{Docstore, Document, InMemoryDocstore};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    let store = InMemoryDocstore::new();
    store
        .put(vec![
            ("a".into(), Document::new("Rust is fast.")),
            ("b".into(), Document::new("Tokio is async.")),
            ("c".into(), Document::new("Cargo is the build tool.")),
        ])
        .await?;

    let got = store.get(&["b".into(), "c".into()]).await?;
    for d in got {
        println!("- {}", d.content);
    }
    store.delete(&["a".into()]).await?;
    let after = store.get(&["a".into(), "b".into(), "c".into()]).await?;
    println!("after delete: {} docs", after.len());
    Ok(())
}
