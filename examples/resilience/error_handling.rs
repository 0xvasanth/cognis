//! What you'll learn:
//!   How a `Runnable` can fail with a `CognisError`, how that error
//!   propagates through `.pipe()`, and how the caller matches on it
//!   to show a friendly message instead of crashing.
//!
//! Why this matters:
//!   User input is messy — someone types "forty-two" when your form
//!   expected "42". Every Cognis primitive uses the same
//!   `CognisError` enum, so error handling is a single match arm
//!   regardless of whether you bubbled out of a parser, an LLM call,
//!   or a tool. No stringly-typed errors, no surprise panics.
//!
//! Scenario:
//!   A signup flow asks the user for their age. The chain validates
//!   the input as a positive `u32`, then proceeds to a fake "create
//!   account" stage. We send three inputs through: a valid number, a
//!   word ("forty-two"), and a negative — and show what the caller
//!   would surface to the UI for each.
//!
//! Run with:
//!   cargo run -p cognis-examples --example resilience_error_handling
//!
//! Sample output (against ollama / llama3.1):
//!   input 29         -> ok: created account (age=29)
//!   input forty-two  -> show user: couldn't parse age from "forty-two" — please enter a number
//!   input -3         -> show user: age must be positive (got "-3")

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::runnable_ext::RunnableExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Stage 1: validate. Returns a typed `Validation` error so the
    // caller can match on it.
    let parse_age = lambda(|raw: String| async move {
        match raw.trim().parse::<i32>() {
            Ok(n) if n > 0 => Ok::<_, CognisError>(n as u32),
            Ok(_) => Err(CognisError::Internal(format!(
                "age must be positive (got {raw:?})"
            ))),
            Err(_) => Err(CognisError::Internal(format!(
                "couldn't parse age from {raw:?} — please enter a number"
            ))),
        }
    });

    // Stage 2: pretend to create an account.
    let create_account =
        lambda(
            |age: u32| async move { Ok::<_, CognisError>(format!("created account (age={age})")) },
        );

    let signup = parse_age.pipe(create_account);

    for input in ["29", "forty-two", "-3"] {
        match signup.invoke(input.to_string(), Default::default()).await {
            Ok(msg) => println!("input {input:<10} -> ok: {msg}"),
            Err(CognisError::Internal(reason)) => {
                // The shape your UI middleware would consume.
                println!("input {input:<10} -> show user: {reason}");
            }
            Err(other) => {
                println!("input {input:<10} -> unexpected error: {other}");
            }
        }
    }
    Ok(())
}
