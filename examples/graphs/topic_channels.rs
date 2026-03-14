//! Topic Channels Example
//!
//! Demonstrates pub/sub topic channels: TopicMessage, TopicFilter,
//! TopicChannel, TopicRouter, DeadLetterQueue, and TopicBus.
//!
//! Run with: `cargo run -p cognis-examples --example topic_channels`

#[path = "../shared.rs"]
mod shared;
use cognis_core::messages::Message;
use cognisgraph::channels::topic::{
    DeadLetterQueue, TopicBus, TopicChannel, TopicFilter, TopicMessage, TopicRouter,
};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TopicMessage creation
    let msg1 = TopicMessage::new("orders.created", json!({"order_id": 1001, "amount": 59.99}))
        .with_sender("order_service");
    println!("Message: {}", msg1.to_json());

    // TopicFilter variants
    let topics = [
        "orders.created",
        "orders.shipped",
        "logs.info",
        "logs.error",
        "metrics.cpu",
    ];
    let filter_prefix = TopicFilter::Prefix("orders.".to_string());
    let matches: Vec<_> = topics.iter().filter(|t| filter_prefix.matches(t)).collect();
    println!("Prefix('orders.') matches: {:?}", matches);

    // TopicChannel - buffered pub/sub
    let mut channel = TopicChannel::new();
    channel.publish(TopicMessage::new("orders.created", json!({"id": 1})));
    channel.publish(TopicMessage::new("orders.shipped", json!({"id": 2})));
    channel.publish(TopicMessage::new("logs.info", json!("health check ok")));

    let order_msgs = channel.subscribe(TopicFilter::Prefix("orders.".to_string()));
    println!("Order messages: {}", order_msgs.len());

    let consumed = channel.consume(TopicFilter::Exact("logs.info".to_string()));
    println!(
        "Consumed {} log messages, {} remaining",
        consumed.len(),
        channel.message_count()
    );

    // TopicRouter - rule-based routing
    let mut router = TopicRouter::new();
    router.add_route(TopicFilter::Prefix("orders.".to_string()), "order_handler");
    router.add_route(TopicFilter::Prefix("logs.".to_string()), "log_handler");

    let handlers = router.route(&TopicMessage::new("orders.created", Value::Null));
    println!("'orders.created' routed to: {:?}", handlers);

    // TopicBus - unified message bus
    let mut bus = TopicBus::new();
    bus.add_route(TopicFilter::Prefix("orders.".to_string()), "order_service");
    bus.add_route(TopicFilter::Prefix("logs.".to_string()), "log_service");

    bus.publish(TopicMessage::new("orders.created", json!({"order_id": 101})).with_sender("api"));
    bus.publish(TopicMessage::new("logs.error", json!("timeout")).with_sender("db"));
    bus.publish(TopicMessage::new("unknown.topic", json!("payload")).with_sender("notifier"));

    let stats = bus.stats();
    println!(
        "Bus: {} messages, {} topics, {} dead letters",
        stats.message_count, stats.topic_count, stats.dlq_count
    );

    // LLM demo
    let model = shared::get_chat_model(vec![
        "This message is about an order being placed, so it should be routed to 'orders.created'."
            .into(),
    ]);
    let messages = vec![
        Message::system("Classify messages into topics: orders.created, orders.shipped, logs.info, logs.error, metrics.cpu"),
        Message::human("Customer #42 just placed an order for 3 items totaling $127.50"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM classification: {}", gen.message.content().text());
    }

    Ok(())
}
