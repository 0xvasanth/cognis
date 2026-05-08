//! What you'll learn:
//!   How to build a polling loop with a max-attempts cap as a single-
//!   node graph: each tick checks whether the external job is done,
//!   either routes back to itself or to `Goto::end()`.
//!
//! Why this matters:
//!   "Wait for an external job to finish" is one of the most common
//!   real workflows — webhook delivery, slow data pipeline, third-
//!   party API. Modeling it as a graph node gives you free
//!   checkpointing, observability, and the ability to swap in a
//!   smarter backoff strategy without rewriting the loop.
//!
//! Scenario:
//!   The agent kicked off a long-running export and got back a job ID.
//!   We poll the (stubbed) status endpoint up to 5 times. We run the
//!   graph twice: once with a stub that finishes on attempt 3 (success
//!   path), once with a stub that never finishes (timeout path,
//!   `gave_up = true`).
//!
//! Run with:
//!   cargo run -p cognis-examples --example graphs_state_machine
//!
//! Sample output (against ollama / llama3.1):
//!   --- success path: completes on attempt 3 ---
//!   [poll] attempt 1/5
//!   [poll] attempt 2/5
//!   [poll] attempt 3/5
//!   [poll] job complete on attempt 3
//!   final: State { attempts: 3, finished: true, gave_up: false }
//!
//!   --- timeout path: never finishes ---
//!   [poll] attempt 1/5
//!   [poll] attempt 2/5
//!   [poll] attempt 3/5
//!   [poll] attempt 4/5
//!   [poll] attempt 5/5
//!   [poll] giving up after 5 attempts
//!   final: State { attempts: 5, finished: false, gave_up: true }

use std::sync::Arc;

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State {
    attempts: u32,
    finished: bool,
    gave_up: bool,
}
#[derive(Default, Clone)]
struct Update {
    attempts: u32,
    finished: Option<bool>,
    gave_up: Option<bool>,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        self.attempts += u.attempts;
        if let Some(f) = u.finished {
            self.finished = f;
        }
        if let Some(g) = u.gave_up {
            self.gave_up = g;
        }
    }
}

const MAX_ATTEMPTS: u32 = 5;

/// Status-check stub: takes the attempt number, returns whether the
/// (pretend) job is done. Real code would be an HTTP call.
type StatusCheck = Arc<dyn Fn(u32) -> bool + Send + Sync>;

async fn run_once(label: &str, status_check: StatusCheck) -> Result<State> {
    let poll = node_fn::<State, _, _>("poll", move |s, _| {
        let already = s.attempts;
        let check = status_check.clone();
        async move {
            let attempt = already + 1;
            println!("[poll] attempt {attempt}/{MAX_ATTEMPTS}");

            if check(attempt) {
                println!("[poll] job complete on attempt {attempt}");
                return Ok(NodeOut {
                    update: Update {
                        attempts: 1,
                        finished: Some(true),
                        gave_up: None,
                    },
                    goto: Goto::end(),
                });
            }
            if attempt >= MAX_ATTEMPTS {
                println!("[poll] giving up after {attempt} attempts");
                return Ok(NodeOut {
                    update: Update {
                        attempts: 1,
                        finished: None,
                        gave_up: Some(true),
                    },
                    goto: Goto::end(),
                });
            }
            // Otherwise: increment counter and loop. In real code this
            // is where you'd `tokio::time::sleep` for a backoff window.
            Ok(NodeOut {
                update: Update {
                    attempts: 1,
                    finished: None,
                    gave_up: None,
                },
                goto: Goto::node("poll"),
            })
        }
    });

    println!("--- {label} ---");
    let graph = Graph::<State>::new()
        .node("poll", poll)
        .start_at("poll")
        .compile()?;
    let final_state = graph.invoke(State::default(), Default::default()).await?;
    println!("final: {final_state:?}\n");
    Ok(final_state)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Path 1 — success: job finishes on attempt 3.
    run_once(
        "success path: completes on attempt 3",
        Arc::new(|attempt| attempt >= 3),
    )
    .await?;

    // Path 2 — timeout: status check never returns true; the
    // max-attempts cap kicks in.
    run_once("timeout path: never finishes", Arc::new(|_attempt| false)).await?;

    Ok(())
}
