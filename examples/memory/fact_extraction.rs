//! What you'll learn:
//!   How to use [`FactExtractor`] to distil raw agent output into typed
//!   atomic facts, and how to combine it with [`DedupVectorStore`] so
//!   repeated observations never bloat the memory index.
//!   Also shows [`LlmExtractor`] — the generic building block underneath
//!   `FactExtractor` — for any custom structured-output type.
//!
//! Why this matters:
//!   Agents working on the same codebase or project will surface the
//!   same facts across many turns ("use PostgreSQL", "prefer async",
//!   "monorepo layout"). Without deduplication that noise accumulates
//!   in the vector index, degrading retrieval quality over time. The two
//!   primitives here are the write-time pair every persistent-memory
//!   system needs: extract first, then store without duplicates.
//!
//! Scenario:
//!   A backend team runs an AI agent for two planning sessions on the
//!   same service.  Both sessions surface overlapping architecture facts.
//!   We extract facts from each session, store them in a
//!   `DedupVectorStore`, and show that the second session adds only the
//!   genuinely new information — the rest is silently skipped.
//!
//! Run with (Ollama — no API key needed):
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=qwen2.5:3b \
//!     cargo run -p cognis-examples --example memory_fact_extraction
//!
//! Run with (Anthropic):
//!   COGNIS_PROVIDER=anthropic COGNIS_ANTHROPIC_API_KEY=sk-ant-… \
//!     cargo run -p cognis-examples --example memory_fact_extraction
//!
//! Run with (OpenAI):
//!   COGNIS_PROVIDER=openai COGNIS_OPENAI_API_KEY=sk-… \
//!     cargo run -p cognis-examples --example memory_fact_extraction
//!
//! Note: smaller Ollama models (1b, 3b) sometimes emit prose instead of
//! JSON.  `FactExtractor` handles that gracefully — it returns an empty
//! Vec and logs a warning rather than propagating an error.  For reliable
//! JSON output use qwen2.5:3b, llama3.1:8b, or any hosted provider.
//!
//! Sample output (against ollama / qwen2.5:3b):
//!   Backend: ollama
//!
//!   ── Session 1: initial planning ──────────────────────────────────────
//!   Extracted 6 fact(s):
//!     [Decision]    (0.90)  The team decided to keep all services in a single Rust workspace (monorepo).
//!     [Observation] (0.80)  The main reason was that the overhead of cross-repo dependency management …
//!     [Rule]        (0.90)  Engineers must always run `cargo fmt --all` and `cargo clippy` before pushing.
//!     [Preference]  (0.80)  The team prefers async-first code using tokio.
//!     [Observation] (0.70)  We use PostgreSQL for persistent storage and Redis for caching hot data.
//!     [Observation] (0.60)  The billing service will be a new crate inside the existing workspace.
//!   Stored 6 fact(s) in the memory index.
//!
//!   ── In-session dedup demo ─────────────────────────────────────────────
//!   Re-adding the same 6 facts verbatim …
//!   Skipped 6/6 — already in the index.
//!
//!   ── Session 2: follow-up ─────────────────────────────────────────────
//!   Extracted 3 fact(s):
//!     [Context]     (0.50)  PostgreSQL will store all billing records in the billing service.
//!     [Context]     (0.50)  gRPC is being used for inter-service communication.
//!     [Rule]        (0.70)  The team follows cargo fmt and clippy before every push.
//!   Added 3 new fact(s), skipped 0 duplicate(s).   ← new phrasing → new fingerprints
//!   Memory index: 6 → 9 unique document(s).
//!
//!   ── Generic LlmExtractor<TechStack> ──────────────────────────────────
//!   runtime:   tokio
//!   languages: Rust
//!   databases: PostgreSQL, Redis
//!
//!   ── Dedup persistence across restarts ───────────────────────────────
//!   Fingerprints to persist: 9
//!   After restore: 9/9 facts correctly skipped as duplicates.

use std::sync::Arc;

use cognis::agent::{Fact, FactExtractionInput, FactExtractor, LlmExtractor};
use cognis::prelude::*;
use cognis::{DedupVectorStore, FakeEmbeddings, InMemoryVectorStore};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

// ── two planning-session outputs ───────────────────────────────────────────

const SESSION_1: &str = "\
After evaluating several approaches the team decided to keep all services in a \
single Rust workspace (monorepo) rather than splitting into separate repositories. \
The main reason was that the overhead of cross-repo dependency management \
outweighed any isolation benefit at current scale.

Engineers must always run `cargo fmt --all` and `cargo clippy` before pushing. \
The team prefers async-first code using tokio. We use PostgreSQL for persistent \
storage and Redis for caching hot data. The billing service will be a new crate \
inside the existing workspace.";

// The second session overlaps on most facts (fmt rule, postgres, async preference)
// but introduces one genuinely new fact: gRPC for inter-service comms.
const SESSION_2: &str = "\
Follow-up on the billing service design. The billing service is a crate inside \
the existing Rust workspace. We confirmed that PostgreSQL will store all billing \
records. The team is adopting gRPC for inter-service communication because it \
gives us typed contracts across teams. Code quality: run cargo fmt and clippy \
before every push — this is non-negotiable.";

// ── helpers ────────────────────────────────────────────────────────────────

fn print_facts(facts: &[Fact]) {
    if facts.is_empty() {
        println!("  (no facts extracted — model may not have returned JSON)");
        return;
    }
    for fact in facts {
        println!(
            "  [{:12?}] ({:.2})  {}",
            fact.kind, fact.importance, fact.content
        );
    }
}

// ── main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let client = Arc::new(Client::from_env()?);

    // Show which backend we resolved from the environment.
    println!("Backend: {}\n", client.provider().name());

    // --------------------------------------------------------------------------
    // 1. FactExtractor — two sessions, overlapping facts
    // --------------------------------------------------------------------------

    let extractor = FactExtractor::new(Arc::clone(&client));

    // Session 1 ─────────────────────────────────────────────────────────────
    println!("── Session 1: initial planning ──────────────────────────────────────");
    let session1_facts: Vec<Fact> = extractor
        .invoke(
            FactExtractionInput::new(SESSION_1)
                .with_hint("project: billing-service")
                .with_hint("team: backend")
                .with_max_facts(7),
            Default::default(),
        )
        .await?;

    println!("Extracted {} fact(s):", session1_facts.len());
    print_facts(&session1_facts);

    // Store session-1 facts in the dedup store.
    // FakeEmbeddings is used here because this demo is about deduplication,
    // not semantic search. In production wire up OllamaEmbeddings or
    // OpenAIEmbeddings so similarity_search returns meaningful results.
    let embedder = Arc::new(FakeEmbeddings::new(16));
    let mut memory: DedupVectorStore<InMemoryVectorStore> =
        DedupVectorStore::new(InMemoryVectorStore::new(embedder));

    for fact in &session1_facts {
        memory.add_texts(vec![fact.content.clone()], None).await?;
    }
    println!("\nStored {} fact(s) in the memory index.", memory.len());

    // In-session dedup: re-adding the same facts verbatim must all be skipped.
    println!("\n── In-session dedup demo ─────────────────────────────────────────────");
    println!(
        "Re-adding the same {} facts verbatim …",
        session1_facts.len()
    );
    let mut in_session_skipped = 0usize;
    for fact in &session1_facts {
        let ids = memory.add_texts(vec![fact.content.clone()], None).await?;
        if ids[0].starts_with("dedup:skipped:") {
            in_session_skipped += 1;
        }
    }
    println!(
        "Skipped {in_session_skipped}/{} — already in the index.\n",
        session1_facts.len()
    );

    // Session 2 ─────────────────────────────────────────────────────────────
    println!("── Session 2: follow-up ─────────────────────────────────────────────");
    let session2_facts: Vec<Fact> = extractor
        .invoke(
            FactExtractionInput::new(SESSION_2)
                .with_hint("project: billing-service")
                .with_hint("team: backend")
                .with_max_facts(7),
            Default::default(),
        )
        .await?;

    println!("Extracted {} fact(s):", session2_facts.len());
    print_facts(&session2_facts);
    println!();

    let before = memory.len();
    let mut stored = 0usize;
    let mut skipped = 0usize;

    for fact in &session2_facts {
        let ids = memory.add_texts(vec![fact.content.clone()], None).await?;
        if ids[0].starts_with("dedup:skipped:") {
            skipped += 1;
        } else {
            stored += 1;
        }
    }
    let after = memory.len();

    println!("Added {stored} new fact(s), skipped {skipped} duplicate(s).");
    println!("Memory index: {} → {} unique document(s).\n", before, after);

    // --------------------------------------------------------------------------
    // 2. LlmExtractor<T> — generic extraction with a custom output type
    //    Shows that any DeserializeOwned + JsonSchema type works as O.
    // --------------------------------------------------------------------------

    #[derive(Debug, Deserialize, JsonSchema)]
    struct TechStack {
        /// Primary programming language(s) mentioned.
        languages: Vec<String>,
        /// Database systems mentioned.
        databases: Vec<String>,
        /// Async runtime or web framework mentioned, or "unknown".
        runtime: String,
    }

    println!("── Generic LlmExtractor<TechStack> ─────────────────────────────────");
    let stack_extractor = LlmExtractor::<TechStack>::builder(Arc::clone(&client))
        .system_prompt(
            "Extract the technology stack from the provided text. \
             If a field has no value use an empty list or \"unknown\".",
        )
        .build();

    match stack_extractor
        .invoke(SESSION_1.to_string(), Default::default())
        .await
    {
        Ok(stack) => {
            println!("runtime:   {}", stack.runtime);
            println!("languages: {}", stack.languages.join(", "));
            println!("databases: {}", stack.databases.join(", "));
        }
        Err(e) => {
            // Smaller models sometimes don't produce valid JSON even with
            // format instructions.  Surface the error rather than panicking.
            println!("(parse failed — model did not return JSON: {e})");
            println!("Tip: try a larger model such as qwen2.5:7b or llama3.1:8b");
        }
    }

    // --------------------------------------------------------------------------
    // 3. Fingerprint persistence — show how to restore seen-set across restarts
    // --------------------------------------------------------------------------

    println!("\n── Dedup persistence across restarts ───────────────────────────────");

    // Collect fingerprints from the in-memory store — in production you'd
    // persist these to a DB or file and reload them on startup.
    let persisted: Vec<String> = memory.seen_fingerprints().map(str::to_owned).collect();
    println!("Fingerprints to persist: {}", persisted.len());

    // Simulate restart: new store pre-seeded with saved fingerprints.
    let fresh_embedder = Arc::new(FakeEmbeddings::new(16));
    let mut restored: DedupVectorStore<InMemoryVectorStore> =
        DedupVectorStore::with_seen(InMemoryVectorStore::new(fresh_embedder), persisted);

    // Re-inserting any already-seen fact must be skipped, even in a fresh process.
    let all_facts: Vec<&Fact> = session1_facts.iter().chain(&session2_facts).collect();
    let mut still_skipped = 0usize;
    for fact in &all_facts {
        let ids = restored.add_texts(vec![fact.content.clone()], None).await?;
        if ids[0].starts_with("dedup:skipped:") {
            still_skipped += 1;
        }
    }
    println!(
        "After restore: {still_skipped}/{} facts correctly skipped as duplicates.",
        all_facts.len()
    );

    Ok(())
}
