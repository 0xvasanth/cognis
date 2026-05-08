//! End-to-end tests for slice-2c features: parallel fan-out, Send map-reduce,
//! interrupt + resume, real per-node streaming.

use std::sync::Arc;

use cognis_core::{CognisError, Event, InterruptKind, Runnable, RunnableConfig};
use cognis_graph::{node_fn, Checkpointer, Goto, Graph, GraphState, InMemoryCheckpointer, NodeOut};
use futures::StreamExt;

#[derive(Default, Clone, Debug, PartialEq, serde::Serialize)]
struct Counter {
    n: u32,
    log: Vec<String>,
}

#[derive(Default, Clone, Debug)]
struct CounterUpdate {
    n: u32,
    log: Vec<String>,
}

impl GraphState for Counter {
    type Update = CounterUpdate;
    fn apply(&mut self, u: Self::Update) {
        self.n += u.n;
        self.log.extend(u.log);
    }
}

#[tokio::test]
async fn goto_multiple_runs_in_parallel_and_merges() {
    // Fan-out: "fan" → ["a", "b", "c"] → all three run, each adds 1.
    // After the parallel superstep, "merge" runs, sees n == 3.
    let g = Graph::<Counter>::new()
        .node(
            "fan",
            node_fn::<Counter, _, _>("fan", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 0,
                        log: vec!["fan".into()],
                    },
                    goto: Goto::Multiple(vec!["a".into(), "b".into(), "c".into()]),
                })
            }),
        )
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 1,
                        log: vec!["a".into()],
                    },
                    goto: Goto::node("merge"),
                })
            }),
        )
        .node(
            "b",
            node_fn::<Counter, _, _>("b", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 1,
                        log: vec!["b".into()],
                    },
                    goto: Goto::node("merge"),
                })
            }),
        )
        .node(
            "c",
            node_fn::<Counter, _, _>("c", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 1,
                        log: vec!["c".into()],
                    },
                    goto: Goto::node("merge"),
                })
            }),
        )
        .node(
            "merge",
            node_fn::<Counter, _, _>("merge", |s, _| {
                let n = s.n;
                async move {
                    Ok(NodeOut {
                        update: CounterUpdate {
                            n: 0,
                            log: vec![format!("merge(n={n})")],
                        },
                        goto: Goto::end(),
                    })
                }
            }),
        )
        .start_at("fan")
        .compile()
        .unwrap();

    let final_state = g
        .invoke(Counter::default(), RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(final_state.n, 3);
    assert!(final_state.log.contains(&"fan".to_string()));
    assert!(final_state.log.contains(&"a".to_string()));
    assert!(final_state.log.contains(&"b".to_string()));
    assert!(final_state.log.contains(&"c".to_string()));
    // After the merge node sees n=3 (parallel branches all ran first).
    assert!(final_state.log.iter().any(|s| s == "merge(n=3)"));
}

#[tokio::test]
async fn goto_send_dispatches_per_target_payloads() {
    // "fan" sends 3 different payloads to "process". Each invocation of
    // "process" reads its payload via ctx.payload() and adds the value
    // to the counter.
    let g = Graph::<Counter>::new()
        .node(
            "fan",
            node_fn::<Counter, _, _>("fan", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate::default(),
                    goto: Goto::Send(vec![
                        ("process".into(), serde_json::json!({"add": 10})),
                        ("process".into(), serde_json::json!({"add": 20})),
                        ("process".into(), serde_json::json!({"add": 30})),
                    ]),
                })
            }),
        )
        .node(
            "process",
            node_fn::<Counter, _, _>("process", |_, ctx| {
                let payload = ctx.payload().cloned().unwrap_or(serde_json::Value::Null);
                async move {
                    let add = payload["add"].as_u64().unwrap_or(0) as u32;
                    Ok(NodeOut {
                        update: CounterUpdate {
                            n: add,
                            log: vec![format!("p({add})")],
                        },
                        goto: Goto::end(),
                    })
                }
            }),
        )
        .start_at("fan")
        .compile()
        .unwrap();

    let final_state = g
        .invoke(Counter::default(), RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(final_state.n, 60);
    assert_eq!(final_state.log.len(), 3);
}

#[tokio::test]
async fn interrupt_before_pauses_and_resumes() {
    let cp: Arc<dyn Checkpointer<Counter>> = Arc::new(InMemoryCheckpointer::<Counter>::new());

    let g = Graph::<Counter>::new()
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 1,
                        log: vec!["a".into()],
                    },
                    goto: Goto::node("b"),
                })
            }),
        )
        .node(
            "b",
            node_fn::<Counter, _, _>("b", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 100,
                        log: vec!["b".into()],
                    },
                    goto: Goto::end(),
                })
            }),
        )
        .start_at("a")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone())
        .with_interrupt_before(["b"]);

    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;

    let err = g.invoke(Counter::default(), cfg).await.unwrap_err();
    let (saved_step, saved_node) = match err {
        CognisError::GraphInterrupted {
            step,
            node,
            kind: InterruptKind::Before,
            ..
        } => (step, node),
        other => panic!("expected GraphInterrupted Before; got {other:?}"),
    };
    assert_eq!(saved_node, "b");

    // Recover state from checkpointer.
    let recovered = cp.load(run_id, Some(saved_step)).await.unwrap().unwrap();
    assert_eq!(recovered.n, 1);
    assert_eq!(recovered.log, vec!["a"]);

    // Resume — slice-2c resume re-runs from start with the recovered state.
    // Use a fresh graph without interrupts so the resume runs to completion.
    // The recovered state (n=1, log=["a"]) is the input; running a again
    // adds 1 more → n=2, then b adds 100 → n=102.
    let g_no_interrupt = Graph::<Counter>::new()
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 1,
                        log: vec!["a-resumed".into()],
                    },
                    goto: Goto::node("b"),
                })
            }),
        )
        .node(
            "b",
            node_fn::<Counter, _, _>("b", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate {
                        n: 100,
                        log: vec!["b".into()],
                    },
                    goto: Goto::end(),
                })
            }),
        )
        .start_at("a")
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    let final_state = g_no_interrupt
        .resume(run_id, saved_step, recovered, RunnableConfig::default())
        .await
        .unwrap();
    // Point-of-interrupt resume: the engine reloads the active set
    // saved at `step` (containing only `b` — the node we were about
    // to run when the interrupt fired) and dispatches it directly.
    // Recovered base n=1, b adds 100 → total 101. (Old re-dispatch-start
    // behavior would have re-run `a` and produced 102.)
    assert_eq!(final_state.n, 1 + 100);
}

#[tokio::test]
async fn interrupt_without_checkpointer_errors() {
    let g = Graph::<Counter>::new()
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut::end_with(CounterUpdate::default()))
            }),
        )
        .start_at("a")
        .compile()
        .unwrap()
        .with_interrupt_before(["a"]);

    let err = g
        .invoke(Counter::default(), RunnableConfig::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("checkpointer"));
}

#[tokio::test]
async fn interrupt_referencing_unknown_node_errors() {
    let cp: Arc<dyn Checkpointer<Counter>> = Arc::new(InMemoryCheckpointer::<Counter>::new());
    let g = Graph::<Counter>::new()
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut::end_with(CounterUpdate::default()))
            }),
        )
        .start_at("a")
        .compile()
        .unwrap()
        .with_checkpointer(cp)
        .with_interrupt_before(["ghost"]);

    let err = g
        .invoke(Counter::default(), RunnableConfig::default())
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("ghost"));
}

#[tokio::test]
async fn stream_events_real_time_per_node() {
    let g = Graph::<Counter>::new()
        .node(
            "a",
            node_fn::<Counter, _, _>("a", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate::default(),
                    goto: Goto::node("b"),
                })
            }),
        )
        .node(
            "b",
            node_fn::<Counter, _, _>("b", |_, _| async move {
                Ok(NodeOut {
                    update: CounterUpdate::default(),
                    goto: Goto::end(),
                })
            }),
        )
        .start_at("a")
        .compile()
        .unwrap();

    let mut s = g
        .stream_events(Counter::default(), RunnableConfig::default())
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(e) = s.next().await {
        events.push(e);
    }
    let names: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            Event::OnNodeStart { node, .. } => Some(node.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["a", "b"]);
    assert!(events.iter().any(|e| matches!(e, Event::OnEnd { .. })));
}
