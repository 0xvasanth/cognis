//! End-to-end test: a fake agent loop without LLMs. Two nodes:
//!   "think" decides to act (if work remaining) or end.
//!   "act"   does one unit of work, increments a counter, loops back.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cognis_core::prelude::*;
use cognis_core::CognisError;
use cognis_graph::{
    node_fn, Checkpointer, CompiledGraph, Goto, Graph, GraphState, InMemoryCheckpointer, NodeOut,
};

/// Simulated agent state: a list of "messages" (strings) and a step counter.
#[derive(Default, Clone, Debug, PartialEq)]
struct AgentState {
    messages: Vec<String>,
    iterations: u32,
}

#[derive(Default, Clone)]
struct AgentStateUpdate {
    messages: Vec<String>,
    iterations: u32,
}

impl GraphState for AgentState {
    type Update = AgentStateUpdate;
    fn apply(&mut self, u: Self::Update) {
        self.messages.extend(u.messages);
        self.iterations += u.iterations;
    }
}

fn build_agent() -> CompiledGraph<AgentState> {
    Graph::<AgentState>::new()
        .node(
            "think",
            node_fn::<AgentState, _, _>("think", |s, _ctx| {
                let iter = s.iterations;
                async move {
                    if iter >= 3 {
                        Ok(NodeOut {
                            update: AgentStateUpdate {
                                messages: vec!["[think] done".into()],
                                iterations: 0,
                            },
                            goto: Goto::end(),
                        })
                    } else {
                        Ok(NodeOut {
                            update: AgentStateUpdate {
                                messages: vec![format!("[think] iteration {iter}")],
                                iterations: 0,
                            },
                            goto: Goto::node("act"),
                        })
                    }
                }
            }),
        )
        .node(
            "act",
            node_fn::<AgentState, _, _>("act", |s, _ctx| {
                let iter = s.iterations;
                async move {
                    Ok(NodeOut {
                        update: AgentStateUpdate {
                            messages: vec![format!("[act] iteration {iter}")],
                            iterations: 1,
                        },
                        goto: Goto::node("think"),
                    })
                }
            }),
        )
        .start_at("think")
        .compile()
        .expect("agent compiles")
}

#[tokio::test]
async fn agent_loops_until_termination() {
    let agent = build_agent();
    let final_state = agent
        .invoke(AgentState::default(), RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(final_state.iterations, 3);
    // 3 think calls (last says done) + 3 act calls = 6 messages + 1 "done" message = 7 total.
    assert_eq!(final_state.messages.len(), 7);
    assert_eq!(final_state.messages.last().unwrap(), "[think] done");
}

#[tokio::test]
async fn observer_receives_node_events() {
    let count = Arc::new(AtomicUsize::new(0));
    let count2 = count.clone();
    let observer: Arc<dyn Observer> = Arc::new(move |e: &Event| {
        if matches!(e, Event::OnNodeStart { .. } | Event::OnNodeEnd { .. }) {
            count2.fetch_add(1, Ordering::SeqCst);
        }
    });
    let cfg = RunnableConfig::default().with_observer(observer);
    let agent = build_agent();
    let _ = agent.invoke(AgentState::default(), cfg).await.unwrap();
    // 4 think runs + 3 act runs = 7 nodes; each emits OnNodeStart + OnNodeEnd = 14 events.
    assert_eq!(count.load(Ordering::SeqCst), 14);
}

#[tokio::test]
async fn recursion_limit_blocks_infinite_loops() {
    let agent = build_agent();
    let cfg = RunnableConfig::default().with_recursion_limit(2);
    let err = agent.invoke(AgentState::default(), cfg).await.unwrap_err();
    assert!(matches!(err, CognisError::RecursionLimit { limit: 2 }));
}

#[tokio::test]
async fn checkpointer_saves_each_step() {
    let cp: Arc<dyn Checkpointer<AgentState>> = Arc::new(InMemoryCheckpointer::<AgentState>::new());
    let agent = build_agent().with_checkpointer(cp.clone());
    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let _ = agent.invoke(AgentState::default(), cfg).await.unwrap();

    let steps = cp.list(run_id).await.unwrap();
    assert_eq!(steps.len(), 7); // 7 supersteps total

    // Latest checkpoint matches the final state
    let latest = cp.load(run_id, None).await.unwrap().unwrap();
    assert_eq!(latest.iterations, 3);
}

#[tokio::test]
async fn time_travel_via_explicit_step() {
    let cp: Arc<dyn Checkpointer<AgentState>> = Arc::new(InMemoryCheckpointer::<AgentState>::new());
    let agent = build_agent().with_checkpointer(cp.clone());
    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let _ = agent.invoke(AgentState::default(), cfg).await.unwrap();

    // After step 0 (the first "think" running once with iterations=0 and routing to act):
    let s0 = cp.load(run_id, Some(0)).await.unwrap().unwrap();
    assert_eq!(s0.iterations, 0); // think didn't increment
    assert_eq!(s0.messages.len(), 1);

    // After step 1 (act ran once, incremented):
    let s1 = cp.load(run_id, Some(1)).await.unwrap().unwrap();
    assert_eq!(s1.iterations, 1);
    assert_eq!(s1.messages.len(), 2);
}
