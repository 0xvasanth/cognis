//! Integration tests for AgentExecutor::astream_events.

use std::sync::Arc;

use futures::StreamExt;

use cognis::agents::AgentExecutor;
use cognis_core::language_models::fake::FakeMessagesListChatModel;
use cognis_core::messages::{AIMessage, HumanMessage, Message};
use cognis_core::tracers::event_stream::EventType;

#[tokio::test]
async fn test_astream_events_emits_chain_start_and_end() {
    // Fake model that returns a simple AI response (no tool calls).
    let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
        AIMessage::new("Hello, world!"),
    )]));

    let executor = AgentExecutor::builder()
        .model(model)
        .max_iterations(1)
        .build();

    let messages = vec![Message::Human(HumanMessage::new("Hi"))];
    let mut stream = executor.astream_events(messages).await.unwrap();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }

    let event_types: Vec<&EventType> = events.iter().map(|e| &e.event).collect();
    assert!(
        event_types.contains(&&EventType::OnChainStart),
        "expected OnChainStart, got {:?}",
        event_types
    );
    assert!(
        event_types.contains(&&EventType::OnChainEnd),
        "expected OnChainEnd, got {:?}",
        event_types
    );
}

#[tokio::test]
async fn test_astream_events_stream_completes() {
    let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
        AIMessage::new("Done"),
    )]));

    let executor = AgentExecutor::builder()
        .model(model)
        .max_iterations(1)
        .build();

    let messages = vec![Message::Human(HumanMessage::new("Test"))];
    let stream = executor.astream_events(messages).await.unwrap();

    // Collect all events — should terminate, not hang.
    let events: Vec<_> = stream.collect().await;
    assert!(!events.is_empty(), "expected at least one event");
}
