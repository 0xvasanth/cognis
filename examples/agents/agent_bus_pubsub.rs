//! AgentBus — topic-based pub/sub for inter-agent broadcast.
//! Each subscriber gets every message published to topics they're on.

use cognis::prelude::*;
use cognis::{AgentBus, AgentMessage};

#[tokio::main]
async fn main() -> Result<()> {
    let bus = AgentBus::new();

    // Two subscribers on the "alerts" topic, one on "planning".
    let mut a1 = bus.subscribe("alerts").await;
    let mut a2 = bus.subscribe("alerts").await;
    let mut p1 = bus.subscribe("planning").await;

    // Publish concurrently with subscribers awaiting.
    let bus2 = bus.clone();
    let pub_task = tokio::spawn(async move {
        bus2.publish("alerts", msg("system", "fire")).await;
        bus2.publish("alerts", msg("system", "all clear")).await;
        bus2.publish("planning", msg("user", "draft Q3 plan")).await;
    });

    // Each alerts subscriber sees both alert messages.
    println!("=== alerts subscriber #1 ===");
    println!("  {}", a1.recv().await.unwrap().content.content());
    println!("  {}", a1.recv().await.unwrap().content.content());
    println!("=== alerts subscriber #2 ===");
    println!("  {}", a2.recv().await.unwrap().content.content());
    println!("  {}", a2.recv().await.unwrap().content.content());
    println!("=== planning subscriber ===");
    println!("  {}", p1.recv().await.unwrap().content.content());

    pub_task.await.unwrap();
    println!(
        "\ntopics={} subscribers(alerts)={} subscribers(planning)={}",
        bus.topic_count().await,
        bus.subscriber_count("alerts").await,
        bus.subscriber_count("planning").await
    );
    Ok(())
}

fn msg(from: &str, body: &str) -> AgentMessage {
    AgentMessage {
        from: from.into(),
        to: "broadcast".into(),
        content: Message::human(body),
        metadata: serde_json::Value::Null,
        ..Default::default()
    }
}
