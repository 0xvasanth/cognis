//! End-to-end: `#[reducer(ephemeral)]` field is reset to `Default` at
//! the start of every superstep, so writes from step N do not leak into
//! step N+1.

#![allow(clippy::assign_op_pattern)]

use cognis_core::prelude::*;
use cognis_graph::{node_fn, Goto, Graph, GraphState, NodeOut};

#[derive(GraphState, Default, Clone, Debug, PartialEq, serde::Serialize)]
struct S {
    /// Reset between supersteps.
    #[reducer(ephemeral)]
    scratch: String,
    /// Persists across supersteps.
    #[reducer(append)]
    log: Vec<String>,
    #[reducer(add)]
    iter: u32,
}

#[tokio::test]
async fn ephemeral_field_is_reset_between_supersteps() {
    let g: Graph<S> = Graph::new()
        .node(
            "a",
            node_fn::<S, _, _>("a", |_state, _ctx| async {
                Ok(NodeOut {
                    update: SUpdate {
                        scratch: Some("hello".into()),
                        log: vec!["a-wrote".into()],
                        iter: 1,
                    },
                    goto: Goto::node("b"),
                })
            }),
        )
        .node(
            "b",
            node_fn::<S, _, _>("b", |state, _ctx| {
                let observed = if state.scratch.is_empty() {
                    "empty".to_string()
                } else {
                    format!("leaked:{}", state.scratch)
                };
                async move {
                    Ok(NodeOut {
                        update: SUpdate {
                            scratch: None,
                            log: vec![format!("b-observed:{observed}")],
                            iter: 1,
                        },
                        goto: Goto::end(),
                    })
                }
            }),
        )
        .start_at("a");

    let compiled = g.compile().expect("graph compiles");
    let final_state = compiled
        .invoke(S::default(), RunnableConfig::default())
        .await
        .expect("graph runs");

    assert_eq!(final_state.iter, 2);
    assert_eq!(
        final_state.log,
        vec!["a-wrote".to_string(), "b-observed:empty".to_string()]
    );
    // `b` did not write to `scratch`, and the engine reset it at the
    // start of step 1 — so it should still be empty.
    assert_eq!(final_state.scratch, "");
}
