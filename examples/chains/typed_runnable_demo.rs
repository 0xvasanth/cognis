//! What you'll learn:
//!   How implementing `Runnable<I, O>` directly — instead of stitching
//!   `lambda`s — gives you a named, reusable pipeline stage where the
//!   types `UserQuery -> RankedResult` are enforced at compile time.
//!
//! Why this matters:
//!   Python LangChain forces every stage to round-trip through `dict`
//!   or `Any`. In Cognis, a search pipeline that takes a `UserQuery`
//!   and produces a `RankedResult` cannot accidentally lose a
//!   `user_id` field — the compiler catches it. This is the central
//!   design choice of the framework, and the shape every reusable
//!   pipeline stage will take in your code.
//!
//! Scenario:
//!   A type-safe search pipeline. We build a `SearchStage` that
//!   embeds the user's query text, then ranks documents against it.
//!   Two separate `Runnable`s composed with `.pipe()` — every
//!   intermediate value carries `user_id` so per-user re-ranking is
//!   trivial later.
//!
//! Run with:
//!   cargo run -p cognis-examples --example chains_typed_runnable
//!
//! Sample output (against ollama / llama3.1):
//!   RankedResult {
//!       user_id: 42,
//!       matched_doc: "Enable dark mode",
//!       score: 11733.0,
//!   }

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::runnable_ext::RunnableExt;

#[derive(Debug, Clone)]
struct UserQuery {
    text: String,
    user_id: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Embedded {
    text: String,
    user_id: u64,
    /// Toy 8-dim embedding — in real code, an `Embeddings` trait call.
    vector: [f32; 8],
}

#[derive(Debug)]
#[allow(dead_code)]
struct RankedResult {
    user_id: u64,
    matched_doc: String,
    score: f32,
}

/// First stage: embed the user's query. Implementing `Runnable`
/// directly (rather than using `lambda`) is the shape you reach for
/// when a stage has its own state — an HTTP client, a tokenizer, a
/// model handle.
struct EmbedStage;

#[async_trait]
impl Runnable<UserQuery, Embedded> for EmbedStage {
    async fn invoke(&self, q: UserQuery, _: RunnableConfig) -> Result<Embedded> {
        // Stand-in for a real embedding call: hash chars into 8 buckets.
        let mut vector = [0.0_f32; 8];
        for (i, c) in q.text.chars().enumerate() {
            vector[i % 8] += c as u32 as f32;
        }
        Ok(Embedded { text: q.text, user_id: q.user_id, vector })
    }
}

/// Second stage: rank an embedded query against an in-memory corpus.
struct RankStage {
    corpus: Vec<(String, [f32; 8])>,
}

#[async_trait]
impl Runnable<Embedded, RankedResult> for RankStage {
    async fn invoke(&self, e: Embedded, _: RunnableConfig) -> Result<RankedResult> {
        let (matched_doc, score) = self
            .corpus
            .iter()
            .map(|(doc, vec)| {
                let dot: f32 = e.vector.iter().zip(vec).map(|(a, b)| a * b).sum();
                (doc.clone(), dot)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        Ok(RankedResult { user_id: e.user_id, matched_doc, score })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let corpus = vec![
        ("How to reset password".into(), [10.0, 5.0, 3.0, 8.0, 2.0, 1.0, 4.0, 6.0]),
        ("Refund my last invoice".into(), [3.0, 8.0, 9.0, 2.0, 5.0, 7.0, 1.0, 4.0]),
        ("Enable dark mode".into(), [1.0, 2.0, 4.0, 6.0, 9.0, 3.0, 7.0, 5.0]),
    ];

    // Compile-time check: EmbedStage produces Embedded, RankStage
    // consumes Embedded — wiring them backwards would not type-check.
    let pipeline = EmbedStage.pipe(RankStage { corpus });

    let result = pipeline
        .invoke(UserQuery { text: "reset my password please".into(), user_id: 42 }, Default::default())
        .await?;
    println!("{result:#?}");
    Ok(())
}
