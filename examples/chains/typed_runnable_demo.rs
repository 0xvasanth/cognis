//! V2's `Runnable<I, O>` is generic over input/output types — no
//! separate "TypedRunnable" needed. Composition stays type-safe end
//! to end through `.pipe()`.

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::runnable_ext::RunnableExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct UserQuery {
    text: String,
    user_id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Embedded {
    text: String,
    user_id: u64,
    embedding_dims: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Ranked {
    text: String,
    user_id: u64,
    score: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Typed Runnable Composition ===\n");

    let embed = lambda(|q: UserQuery| async move {
        Ok::<_, cognis_core::CognisError>(Embedded {
            text: q.text,
            user_id: q.user_id,
            embedding_dims: 768,
        })
    });
    let rank = lambda(|e: Embedded| async move {
        let score = (e.embedding_dims as f32) / 1000.0;
        Ok::<_, cognis_core::CognisError>(Ranked {
            text: e.text,
            user_id: e.user_id,
            score,
        })
    });

    // Compile-time type check: embed: UserQuery -> Embedded, rank: Embedded -> Ranked.
    let pipeline = embed.pipe(rank);
    let out = pipeline
        .invoke(
            UserQuery { text: "hello".into(), user_id: 42 },
            Default::default(),
        )
        .await?;
    println!("ranked: {out:?}");
    Ok(())
}
