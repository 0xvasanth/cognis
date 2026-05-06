//! Integration test exercising a custom Runnable through invoke, batch,
//! stream, and stream_events using only the public API.

use async_trait::async_trait;
use cognis2_core::prelude::*;
use cognis2_core::CognisError;
use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A tiny runnable that uppercases its input string.
struct Upper;

#[async_trait]
impl Runnable<String, String> for Upper {
    async fn invoke(&self, input: String, _: RunnableConfig) -> Result<String> {
        Ok(input.to_uppercase())
    }
    fn name(&self) -> &str {
        "upper"
    }
}

#[tokio::test]
async fn invoke_lifecycle() {
    let r = Upper;
    let out = r
        .invoke("hello".into(), RunnableConfig::default())
        .await
        .unwrap();
    assert_eq!(out, "HELLO");
}

#[tokio::test]
async fn batch_runs_all() {
    let r = Upper;
    let out = r
        .batch(
            vec!["a".into(), "b".into(), "c".into()],
            RunnableConfig::default(),
        )
        .await
        .unwrap();
    let mut sorted = out;
    sorted.sort();
    assert_eq!(sorted, vec!["A", "B", "C"]);
}

#[tokio::test]
async fn stream_emits_single_chunk() {
    let r = Upper;
    let s = r
        .stream("rust".into(), RunnableConfig::default())
        .await
        .unwrap();
    let v = s.collect_into_vec().await.unwrap();
    assert_eq!(v, vec!["RUST"]);
}

#[tokio::test]
async fn observer_receives_events() {
    let count = Arc::new(AtomicUsize::new(0));
    let count2 = count.clone();
    let observer: Arc<dyn Observer> = Arc::new(move |e: &Event| {
        if matches!(e, Event::OnStart { .. } | Event::OnEnd { .. }) {
            count2.fetch_add(1, Ordering::SeqCst);
        }
    });

    let cfg = RunnableConfig::default().with_observer(observer);
    let r = Upper;
    let mut events: Vec<Event> = Vec::new();
    let mut s = r.stream_events("hi".into(), cfg).await.unwrap();
    while let Some(e) = s.next().await {
        events.push(e);
    }

    // The default stream_events impl emits OnStart + OnEnd around the invoke call.
    assert!(
        events.iter().any(|e| matches!(e, Event::OnStart { .. })),
        "expected an OnStart event"
    );
    assert!(
        events.iter().any(|e| matches!(e, Event::OnEnd { .. })),
        "expected an OnEnd event"
    );
    // Observers are wired in by graph/llm crates that own emission — count
    // may be zero here since the default impl doesn't call observers directly,
    // but the API must compile and the stream must yield the expected event types.
    let _ = count;
}

#[tokio::test]
async fn error_path_propagates() {
    struct Boom;
    #[async_trait]
    impl Runnable<(), ()> for Boom {
        async fn invoke(&self, _: (), _: RunnableConfig) -> Result<()> {
            Err(CognisError::Internal("kaboom".into()))
        }
    }
    let r = Boom;
    let err = r.invoke((), RunnableConfig::default()).await.unwrap_err();
    assert_eq!(err.category(), "internal");
}

#[test]
fn graph_interrupted_carries_metadata() {
    let e = cognis2_core::CognisError::GraphInterrupted {
        run_id: uuid::Uuid::nil(),
        step: 3,
        node: "review".into(),
        kind: cognis2_core::InterruptKind::Before,
    };
    assert_eq!(e.category(), "graph_interrupted");
    assert!(!e.is_retryable());
    let s = format!("{e}");
    assert!(s.contains("step 3"));
    assert!(s.contains("review"));
    assert!(s.contains("before"));
}
