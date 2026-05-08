//! What you'll learn:
//!   How `.pipe()` composes typed `Runnable`s into a multi-stage
//!   pipeline that flows from raw input through transformation steps
//!   to a final tagged output, with the LLM as one of the stages.
//!
//! Why this matters:
//!   `.pipe()` is Cognis's answer to LCEL's `|` — the staple way to
//!   build LLM pipelines without inventing a "Chain" base class.
//!   Real pipelines mix pure-Rust transforms with LLM calls; the
//!   `lambda` adapter lets you slot a `Client::from_env()` call into
//!   any stage and keep types flowing end-to-end.
//!
//! Scenario:
//!   Triage incoming support tickets. The pipeline takes a ticket
//!   string, asks the LLM to classify it into one of a few buckets,
//!   then tags the result with a routing decision. This is the shape
//!   you'd use behind a Zendesk webhook or a queue worker.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example chains_pipe_operator
//!
//! Sample output (against ollama / llama3.1):
//!   [    finance-team] billing -> Hi, my last invoice charged me twice.
//!   [engineering-oncall] bug -> App crashes when I tap the export button.
//!   [ product-backlog] feature_request -> Can you add dark mode?

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::runnable_ext::RunnableExt;

#[derive(Debug)]
struct Triaged {
    ticket: String,
    category: String,
    queue: &'static str,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;

    // Stage 1: trim and normalise the inbound ticket text. Pure Rust —
    // anything you'd do in a webhook before reaching the LLM.
    let normalize =
        lambda(|raw: String| async move { Ok::<_, CognisError>(raw.trim().replace('\n', " ")) });

    // Stage 2: ask the LLM for a single-word category. This is where
    // most real pipelines call out — classification, summarisation,
    // extraction. The lambda owns the `Client` so the chain stays
    // typed `String -> String`.
    let categorize = lambda(move |ticket: String| {
        let client = client.clone();
        async move {
            let prompt = format!(
                "Classify this support ticket into ONE of: \
                 billing, bug, feature_request, password_reset.\n\
                 Reply with just the category, nothing else.\n\n\
                 Ticket: {ticket}"
            );
            let resp = client.invoke(vec![Message::human(prompt)]).await?;
            // Pair the original ticket with the model's category.
            Ok::<_, CognisError>((ticket, resp.content().trim().to_lowercase()))
        }
    });

    // Stage 3: route to the right queue. Pure Rust again — exactly the
    // post-processing you'd do before publishing to your task queue.
    let route = lambda(|(ticket, category): (String, String)| async move {
        let queue = match category.as_str() {
            "billing" => "finance-team",
            "bug" => "engineering-oncall",
            "feature_request" => "product-backlog",
            "password_reset" => "auth-bot",
            _ => "human-triage",
        };
        Ok::<_, CognisError>(Triaged {
            ticket,
            category,
            queue,
        })
    });

    let pipeline = normalize.pipe(categorize).pipe(route);

    for raw in [
        "  Hi, my last invoice charged me twice.\n",
        "App crashes when I tap the export button.",
        "Can you add dark mode?",
    ] {
        let out = pipeline.invoke(raw.to_string(), Default::default()).await?;
        println!("[{:>16}] {} -> {}", out.queue, out.category, out.ticket);
    }
    Ok(())
}
