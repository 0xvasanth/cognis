//! What you'll learn:
//!   How a single `apply` impl can run *different* reducer logic for
//!   each state field — `last_value` for one, `append` for another —
//!   so multiple producers contribute to a single state without
//!   races and without you writing a custom merger.
//!
//! Why this matters:
//!   Channels (a.k.a. reducers) are how `cognisgraph` lets multiple
//!   producers contribute to a single state field. The pattern is
//!   "define `apply`, return incremental updates from each node" —
//!   and once you internalise it, every multi-step state-machine
//!   you build is just picking the right reducer per field.
//!
//! Scenario:
//!   An onboarding agent collects three pieces of profile info over
//!   multiple turns: a `name` (last write wins), a list of `hobbies`
//!   (append every mention), and a list of `todo` follow-up items
//!   (append). Three turns each contribute partial updates; the
//!   final state shows the reducers in action.
//!
//! Run with:
//!   cargo run -p cognis-examples --example graphs_topic_channels
//!
//! Sample output (against ollama / llama3.1):
//!   name (last_value): Maya Suarez
//!   hobbies (append):  ["hiking", "photography", "rock climbing"]
//!   todo (append):     ["book a guide", "rent a lens"]

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct Profile {
    name: String,
    hobbies: Vec<String>,
    todo: Vec<String>,
}

#[derive(Default, Clone)]
struct Update {
    /// Reducer: last_value.
    set_name: Option<String>,
    /// Reducer: append.
    add_hobby: Option<String>,
    /// Reducer: append.
    add_todo: Option<String>,
}

impl GraphState for Profile {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(n) = u.set_name {
            self.name = n; // last_value
        }
        if let Some(h) = u.add_hobby {
            self.hobbies.push(h); // append
        }
        if let Some(t) = u.add_todo {
            self.todo.push(t); // append
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Three turns, each producing a different update shape. In a real
    // agent these would come from LLM-driven extraction over user
    // messages; here we hard-code so the reducer behaviour is visible.
    let turn1 = node_fn::<Profile, _, _>("turn1", |_, _| async {
        Ok(NodeOut {
            update: Update {
                set_name: Some("Maya".into()),
                add_hobby: Some("hiking".into()),
                add_todo: Some("book a guide".into()),
            },
            goto: Goto::node("turn2"),
        })
    });
    let turn2 = node_fn::<Profile, _, _>("turn2", |_, _| async {
        Ok(NodeOut {
            update: Update {
                set_name: None, // unchanged
                add_hobby: Some("photography".into()),
                add_todo: Some("rent a lens".into()),
            },
            goto: Goto::node("turn3"),
        })
    });
    let turn3 = node_fn::<Profile, _, _>("turn3", |_, _| async {
        Ok(NodeOut {
            // Imagine the user corrected their name on turn 3.
            update: Update {
                set_name: Some("Maya Suarez".into()),
                add_hobby: Some("rock climbing".into()),
                add_todo: None,
            },
            goto: Goto::end(),
        })
    });

    let g = Graph::<Profile>::new()
        .node("turn1", turn1)
        .node("turn2", turn2)
        .node("turn3", turn3)
        .start_at("turn1")
        .compile()?;
    let final_state = g.invoke(Profile::default(), Default::default()).await?;
    println!("name (last_value): {}", final_state.name);
    println!("hobbies (append):  {:?}", final_state.hobbies);
    println!("todo (append):     {:?}", final_state.todo);
    Ok(())
}
