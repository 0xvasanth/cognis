//! Stream all callback events produced during execution of a runnable.
//!
//! This module provides the [`stream_events`] function, the Rust equivalent of
//! Python's `Runnable.astream_events(version="v2")`. It wires up an
//! [`EventStreamCallbackHandler`] into the runnable config, spawns the
//! invocation in a background task, and returns a stream of [`StreamEvent`]
//! values that the caller can consume in real time.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::error::Result;
use crate::tracers::event_stream::{
    EventStreamCallbackHandler, EventType, RootEventFilter, StreamEvent,
};

use super::base::Runnable;
use super::config::{ensure_config, RunnableConfig};

/// Stream all events produced during execution of a runnable.
///
/// Creates an [`EventStreamCallbackHandler`], injects it into the config's
/// callback list, spawns `runnable.invoke()` in a background tokio task, and
/// returns a `Stream` of [`StreamEvent`] values. The stream yields events as
/// they are emitted by the callback handler and completes once the invoke task
/// finishes and the channel is drained.
///
/// This is the Rust equivalent of Python's `astream_events(version="v2")`.
///
/// # Arguments
///
/// * `runnable` - The runnable to execute, wrapped in `Arc` so it can be
///   shared with the spawned task.
/// * `input` - The input value to pass to `runnable.invoke()`.
/// * `config` - Optional configuration. The handler is appended to whatever
///   callbacks are already present.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use futures::StreamExt;
/// use rustchain_core::runnables::{RunnableLambda, events::stream_events};
///
/// let runnable = Arc::new(RunnableLambda::new("double", |v| async move {
///     Ok(serde_json::json!(v.as_i64().unwrap() * 2))
/// }));
///
/// let mut stream = stream_events(runnable, serde_json::json!(5), None).await?;
/// while let Some(event) = stream.next().await {
///     println!("{:?}", event?);
/// }
/// ```
pub async fn stream_events(
    runnable: Arc<dyn Runnable>,
    input: Value,
    config: Option<RunnableConfig>,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
    stream_events_with_filter(runnable, input, config, RootEventFilter::default()).await
}

/// Like [`stream_events`] but with a custom [`RootEventFilter`] to control
/// which events are emitted.
pub async fn stream_events_with_filter(
    runnable: Arc<dyn Runnable>,
    input: Value,
    config: Option<RunnableConfig>,
    filter: RootEventFilter,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
    // 1. Create the event stream handler and take its receiver.
    let handler = Arc::new(EventStreamCallbackHandler::new(256, filter));
    let receiver = handler
        .take_receiver()
        .expect("receiver should be available on a fresh handler");

    // 2. Build a config that includes this handler in callbacks.
    let mut cfg = ensure_config(config.as_ref());
    cfg.callbacks
        .push(handler.clone() as Arc<dyn CallbackHandler>);

    // Assign a run ID if one is not already set.
    let run_id = cfg.run_id.unwrap_or_else(Uuid::new_v4);
    cfg.run_id = Some(run_id);

    // 3. Capture the runnable name for the wrapping chain events.
    let runnable_name = runnable.name().to_string();
    let input_clone = input.clone();

    // 4. Spawn the invoke in a background task.
    //    We emit on_chain_start / on_chain_end / on_chain_error around the
    //    invoke so that every stream_events call produces at least a start
    //    and end (or error) event, matching Python's v2 behaviour.
    let handler_for_task = handler.clone();
    tokio::spawn(async move {
        let serialized =
            serde_json::json!({"name": runnable_name, "id": ["Runnable", &runnable_name]});

        // Emit on_chain_start.
        let _ = handler_for_task
            .on_chain_start(&serialized, &input_clone, run_id, None)
            .await;

        // Run the actual runnable.
        let result = runnable.invoke(input_clone.clone(), Some(&cfg)).await;

        match &result {
            Ok(output) => {
                let _ = handler_for_task
                    .on_chain_end(output, run_id, None)
                    .await;
            }
            Err(e) => {
                let _ = handler_for_task
                    .on_chain_error(&e.to_string(), run_id, None)
                    .await;
            }
        }

        // Drop the handler reference so the sender side closes once all
        // references are gone. The `handler` Arc in the outer scope will be
        // dropped after this function returns, and the one inside `cfg.callbacks`
        // is dropped here when `cfg` goes out of scope.
        drop(handler_for_task);
        drop(cfg);
    });

    // 5. Convert the mpsc::Receiver into a Stream of Result<StreamEvent>.
    let stream = ReceiverStream::new(receiver);
    let mapped = futures::StreamExt::map(stream, Ok);

    Ok(Box::pin(mapped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{RunnableLambda, RunnableSequence};
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn test_stream_events_produces_chain_start_and_end() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new("doubler", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        }));

        let mut stream = stream_events(runnable, json!(5), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        // Should have at least on_chain_start and on_chain_end.
        assert!(
            events.len() >= 2,
            "expected at least 2 events, got {}",
            events.len()
        );

        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainEnd);

        // The end event should carry the output.
        let end_event = events.last().unwrap();
        assert_eq!(end_event.data.output, Some(json!(10)));
    }

    #[tokio::test]
    async fn test_stream_events_correct_event_types() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new("identity", |v: Value| async move {
            Ok(v)
        }));

        let mut stream = stream_events(runnable, json!("hello"), None).await.unwrap();

        let mut event_types = Vec::new();
        while let Some(evt) = stream.next().await {
            let evt = evt.unwrap();
            event_types.push(evt.event.clone());
        }

        assert!(
            event_types.contains(&EventType::OnChainStart),
            "missing on_chain_start, got: {:?}",
            event_types
        );
        assert!(
            event_types.contains(&EventType::OnChainEnd),
            "missing on_chain_end, got: {:?}",
            event_types
        );
    }

    #[tokio::test]
    async fn test_stream_events_completes_after_invoke() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new("slow", |v: Value| async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Ok(v)
        }));

        let mut stream = stream_events(runnable, json!(42), None).await.unwrap();

        let mut count = 0;
        while let Some(evt) = stream.next().await {
            evt.unwrap();
            count += 1;
        }

        // Stream must have completed (the while loop exited).
        assert!(count >= 2, "stream should have yielded events before completing");
    }

    #[tokio::test]
    async fn test_stream_events_with_sequence() {
        let step1 = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 1))
        })) as Arc<dyn Runnable>;

        let step2 = Arc::new(RunnableLambda::new("double", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })) as Arc<dyn Runnable>;

        let sequence = Arc::new(
            RunnableSequence::new(vec![step1, step2]).unwrap()
        ) as Arc<dyn Runnable>;

        let mut stream = stream_events(sequence, json!(3), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        // Should have start and end for the outer sequence.
        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainEnd);

        // The final output should be (3 + 1) * 2 = 8.
        let end_event = events.last().unwrap();
        assert_eq!(end_event.data.output, Some(json!(8)));

        // The name should reflect the sequence.
        assert_eq!(events.first().unwrap().name, "RunnableSequence");
    }

    #[tokio::test]
    async fn test_stream_events_carries_input_in_start() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new("echo", |v: Value| async move {
            Ok(v)
        }));

        let input = json!({"query": "test"});
        let mut stream = stream_events(runnable, input.clone(), None).await.unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.event, EventType::OnChainStart);
        assert_eq!(first.data.input, Some(input));
    }

    #[tokio::test]
    async fn test_stream_events_error_produces_chain_error() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new("failing", |_v: Value| async move {
            Err(crate::error::RustChainError::Other("deliberate failure".into()))
        }));

        let mut stream = stream_events(runnable, json!(1), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainError);
        assert!(events.last().unwrap().data.error.is_some());
    }
}
