//! What you'll learn:
//!   The shape of a regression eval: a list of (question,
//!   expected_answer) pairs, run each through your agent, score the
//!   results, and emit a pass/fail summary. The thing you'd run in
//!   CI before every release.
//!
//! Why this matters:
//!   Before you trust an agent in production, you need a regression
//!   harness — and Cognis intentionally doesn't ship a heavy eval
//!   framework. The `Runnable` interface is enough: this is the
//!   shape every team ends up writing themselves anyway, here as a
//!   single self-contained file so you can copy it.
//!
//! Scenario:
//!   We're shipping a tiny FAQ bot. Five known questions have known
//!   correct answers ("capital of France?" -> "Paris"). The eval
//!   sends each through the agent, checks whether the answer
//!   contains the expected substring, and prints a final pass count.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example obs_evaluation
//!
//! Sample output (against ollama / llama3.1):
//!   0. [PASS] What's the capital of France? -> Paris.
//!   1. [PASS] What's 2 + 2? -> 4.
//!   2. [PASS] Which planet is known as the Red Planet? -> Mars is commonly referred to as the Red Planet.
//!   3. [PASS] What language compiles to native code with the borrow checker? -> Rust.
//!   4. [PASS] Who wrote 'Pride and Prejudice'? -> Jane Austen.
//!
//!   result: 5/5 passed

use cognis::prelude::*;

/// A single test case. In real evals you'd load these from a YAML
/// file or a database; the shape is the same.
struct Case {
    q: &'static str,
    expect_substr: &'static str,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("You are a terse FAQ bot. Answer in one short sentence.")
        .build()?;

    // Known good answers — substring check is forgiving but enough
    // to catch outright regressions.
    let cases = [
        Case { q: "What's the capital of France?",         expect_substr: "Paris" },
        Case { q: "What's 2 + 2?",                         expect_substr: "4" },
        Case { q: "Which planet is known as the Red Planet?", expect_substr: "Mars" },
        Case { q: "What language compiles to native code with the borrow checker?", expect_substr: "Rust" },
        Case { q: "Who wrote 'Pride and Prejudice'?",      expect_substr: "Austen" },
    ];

    let mut pass = 0;
    for (i, case) in cases.iter().enumerate() {
        let resp = agent.run(Message::human(case.q)).await?;
        let body = resp.content.to_lowercase();
        let ok = body.contains(&case.expect_substr.to_lowercase());
        pass += ok as usize;
        let mark = if ok { "PASS" } else { "FAIL" };
        println!("{i}. [{mark}] {} -> {}", case.q, resp.content.trim());
    }
    println!("\nresult: {pass}/{} passed", cases.len());
    if pass != cases.len() {
        // In CI you'd `std::process::exit(1)` here.
        eprintln!("(some cases regressed — fail the build)");
    }
    Ok(())
}
