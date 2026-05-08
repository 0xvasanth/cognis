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
//!   The agent kicked off a long-running export and got back a job
//!   ID. We poll the (stubbed) status endpoint up to 5 times. If
//!   the job finishes, we end successfully; if we hit the cap, we
//!   end with `gave_up = true` so the caller can surface that to the
//!   user.
//!
//! Run with:
//!   cargo run -p cognis-examples --example graphs_state_machine
//!
//! Sample output (against ollama / llama3.1):
//!   [poll] attempt 1/5
//!   [poll] attempt 2/5
//!   [poll] attempt 3/5
//!   [poll] job complete on attempt 3
//!
//!   final: State { attempts: 3, finished: true, gave_up: false }

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
        if let Some(f) = u.finished { self.finished = f; }
        if let Some(g) = u.gave_up { self.gave_up = g; }
    }
}

const MAX_ATTEMPTS: u32 = 5;

/// Stand-in for a status check. The real call would be HTTP — the
/// shape of the loop is the same.
fn job_done(attempt: u32) -> bool {
    // Pretend the job finishes on attempt 3.
    attempt >= 3
}

#[tokio::main]
async fn main() -> Result<()> {
    let poll = node_fn::<State, _, _>("poll", |s, _| {
        let already = s.attempts;
        async move {
            let attempt = already + 1;
            println!("[poll] attempt {attempt}/{MAX_ATTEMPTS}");

            if job_done(attempt) {
                println!("[poll] job complete on attempt {attempt}");
                return Ok(NodeOut {
                    update: Update { attempts: 1, finished: Some(true), gave_up: None },
                    goto: Goto::end(),
                });
            }
            if attempt >= MAX_ATTEMPTS {
                println!("[poll] giving up after {attempt} attempts");
                return Ok(NodeOut {
                    update: Update { attempts: 1, finished: None, gave_up: Some(true) },
                    goto: Goto::end(),
                });
            }
            // Otherwise: increment counter and loop. In real code this
            // is where you'd `tokio::time::sleep` for a backoff window.
            Ok(NodeOut {
                update: Update { attempts: 1, finished: None, gave_up: None },
                goto: Goto::node("poll"),
            })
        }
    });

    let graph = Graph::<State>::new()
        .node("poll", poll)
        .start_at("poll")
        .compile()?;
    let final_state = graph.invoke(State::default(), Default::default()).await?;
    println!("\nfinal: {final_state:?}");
    Ok(())
}
