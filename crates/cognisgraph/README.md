# cognis-graph

A state-aware graph execution engine for building complex AI agents and cyclic workflows.

## Purpose
`cognis-graph` implements a Pregel-inspired superstep executor for executing graphs where nodes communicate via shared state. It is designed for building agents that require loops, complex decision trees, and persistent state across execution steps.

## Key Features
- **Typed State**: Define your graph state as a Rust struct with per-field "reducers" (e.g., append to list, take last value).
- **Cyclic Execution**: Support for loops and conditional edges, enabling ReAct-style agent loops.
- **Checkpointers**: Built-in support for persisting graph state to memory, SQLite, or Postgres, allowing workflows to be paused and resumed.
- **Runnable Integration**: A compiled graph implements the `Runnable<S, S>` trait, making it compatible with the rest of the Cognis ecosystem.
- **Audit Logs**: Detailed execution tracking for debugging and observability.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
cognis-graph = "0.1.0"
```

### Basic Example: A Simple State Machine
```rust
use cognis_graph::*;
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize, GraphState, Clone)]
struct State {
    #[reducer(LastValue)]
    count: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut workflow = Graph::new();

    workflow.add_node("increment", node_fn(|mut state: State, _| async move {
        state.count += 1;
        Ok(state)
    }));

    workflow.set_entry_point("increment");
    workflow.add_edge("increment", END);

    let app = workflow.compile();
    let final_state = app.invoke(State::default(), RunnableConfig::default()).await?;
    
    println!("Final count: {}", final_state.count);
    Ok(())
}
```
