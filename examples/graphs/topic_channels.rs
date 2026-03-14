//! Topic Channels Example
//!
//! Demonstrates the LangGraph topic channel system for pub/sub style
//! message routing between graph nodes.
//!
//! Features shown:
//! - TopicMessage creation with topics, payloads, and senders
//! - TopicChannel for buffered pub/sub communication
//! - TopicFilter variants (Exact, Prefix, Pattern, All)
//! - TopicRouter for rule-based message routing
//! - DeadLetterQueue for unroutable messages
//! - TopicBus combining channel, router, and DLQ
//!
//! No API keys required.
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
    println!("=== Topic Channels Example ===\n");

    // -----------------------------------------------------------------------
    // 1. TopicMessage — structured messages with topics and metadata
    // -----------------------------------------------------------------------
    println!("--- 1. TopicMessage ---");
    println!("Structured messages with topic, payload, sender, and timestamp.\n");

    let msg1 = TopicMessage::new("orders.created", json!({"order_id": 1001, "amount": 59.99}))
        .with_sender("order_service");

    let msg2 = TopicMessage::new(
        "orders.shipped",
        json!({"order_id": 1001, "carrier": "FedEx"}),
    )
    .with_sender("shipping_service");

    let msg3 =
        TopicMessage::new("logs.info", json!("System started successfully")).with_sender("system");

    println!("Message 1: {}", msg1.to_json());
    println!("Message 2: {}", msg2.to_json());
    println!("Message 3: {}", msg3.to_json());
    println!("Message 1 age: {:?}", msg1.age());

    // -----------------------------------------------------------------------
    // 2. TopicFilter — filtering messages by topic
    // -----------------------------------------------------------------------
    println!("\n--- 2. TopicFilter ---");
    println!("Filter messages using Exact, Prefix, Pattern, or All.\n");

    let topics = [
        "orders.created",
        "orders.shipped",
        "orders.cancelled",
        "logs.info",
        "logs.error",
        "metrics.cpu",
    ];

    let filter_exact = TopicFilter::Exact("orders.created".to_string());
    let filter_prefix = TopicFilter::Prefix("orders.".to_string());
    let filter_pattern = TopicFilter::Pattern("*.error".to_string());
    let filter_all = TopicFilter::All;

    println!("Topics: {:?}\n", topics);

    print!("  Exact('orders.created') matches: ");
    let matches: Vec<&&str> = topics.iter().filter(|t| filter_exact.matches(t)).collect();
    println!("{:?}", matches);

    print!("  Prefix('orders.') matches:      ");
    let matches: Vec<&&str> = topics.iter().filter(|t| filter_prefix.matches(t)).collect();
    println!("{:?}", matches);

    print!("  Pattern('*.error') matches:      ");
    let matches: Vec<&&str> = topics
        .iter()
        .filter(|t| filter_pattern.matches(t))
        .collect();
    println!("{:?}", matches);

    print!("  All matches:                     ");
    let matches: Vec<&&str> = topics.iter().filter(|t| filter_all.matches(t)).collect();
    println!("{:?}", matches);

    // -----------------------------------------------------------------------
    // 3. TopicChannel — buffered pub/sub
    // -----------------------------------------------------------------------
    println!("\n--- 3. TopicChannel ---");
    println!("A buffered pub/sub channel that stores messages.\n");

    let mut channel = TopicChannel::new();

    // Publish messages
    channel.publish(TopicMessage::new("orders.created", json!({"id": 1})));
    channel.publish(TopicMessage::new("orders.shipped", json!({"id": 2})));
    channel.publish(TopicMessage::new("logs.info", json!("health check ok")));
    channel.publish(TopicMessage::new("orders.created", json!({"id": 3})));

    println!("Published 4 messages");
    println!("Total messages: {}", channel.message_count());
    println!("Unique topics: {:?}", channel.topics());

    // Subscribe to specific topics
    let order_msgs = channel.subscribe(TopicFilter::Prefix("orders.".to_string()));
    println!("\nSubscribe to 'orders.*': {} messages", order_msgs.len());
    for msg in &order_msgs {
        println!(
            "  [seq: {}] topic={}, payload={}",
            msg.sequence, msg.topic, msg.payload
        );
    }

    // Peek at latest message for a topic
    if let Some(latest) = channel.peek("orders.created") {
        println!(
            "\nLatest 'orders.created': seq={}, payload={}",
            latest.sequence, latest.payload
        );
    }

    // Consume (remove) messages matching a filter
    let consumed = channel.consume(TopicFilter::Exact("logs.info".to_string()));
    println!(
        "\nConsumed {} log messages, {} remaining in channel",
        consumed.len(),
        channel.message_count()
    );

    // -----------------------------------------------------------------------
    // 4. TopicRouter — rule-based routing
    // -----------------------------------------------------------------------
    println!("\n--- 4. TopicRouter ---");
    println!("Routes messages to named handlers based on filter rules.\n");

    let mut router = TopicRouter::new();
    router.add_route(TopicFilter::Prefix("orders.".to_string()), "order_handler");
    router.add_route(TopicFilter::Prefix("logs.".to_string()), "log_handler");
    router.add_route(
        TopicFilter::Exact("metrics.cpu".to_string()),
        "metrics_handler",
    );
    router.add_route(TopicFilter::All, "audit_handler");

    println!(
        "Registered {} routes: {:?}\n",
        router.len(),
        router.routes()
    );

    let test_messages = vec![
        TopicMessage::new("orders.created", Value::Null),
        TopicMessage::new("logs.error", Value::Null),
        TopicMessage::new("metrics.cpu", Value::Null),
        TopicMessage::new("unknown.topic", Value::Null),
    ];

    for msg in &test_messages {
        let handlers = router.route(msg);
        println!("  topic='{}' -> handlers: {:?}", msg.topic, handlers);
    }

    // -----------------------------------------------------------------------
    // 5. DeadLetterQueue — unroutable messages
    // -----------------------------------------------------------------------
    println!("\n--- 5. DeadLetterQueue ---");
    println!("Stores messages that could not be routed to any handler.\n");

    let mut dlq = DeadLetterQueue::new(5);
    dlq.add(TopicMessage::new("unknown.a", json!("payload_a")));
    dlq.add(TopicMessage::new("unknown.b", json!("payload_b")));
    dlq.add(TopicMessage::new("unknown.c", json!("payload_c")));

    println!("DLQ contains {} messages", dlq.len());

    let drained = dlq.drain();
    println!("Drained {} messages from DLQ:", drained.len());
    for msg in &drained {
        println!("  topic='{}', payload={}", msg.topic, msg.payload);
    }
    println!("DLQ after drain: {} messages", dlq.len());

    // -----------------------------------------------------------------------
    // 6. TopicBus — unified message bus
    // -----------------------------------------------------------------------
    println!("\n--- 6. TopicBus ---");
    println!("Combines TopicChannel, TopicRouter, and DeadLetterQueue.\n");

    let mut bus = TopicBus::new();

    // Register routes
    bus.add_route(TopicFilter::Prefix("orders.".to_string()), "order_service");
    bus.add_route(TopicFilter::Prefix("logs.".to_string()), "log_service");
    bus.add_route(
        TopicFilter::Pattern("metrics.*".to_string()),
        "metrics_service",
    );

    // Publish messages — routed ones and unrouted ones
    println!("Publishing messages to the bus...\n");

    let handlers = bus.publish(
        TopicMessage::new("orders.created", json!({"order_id": 101})).with_sender("api_gateway"),
    );
    println!("  'orders.created' -> routed to: {:?}", handlers);

    let handlers = bus.publish(
        TopicMessage::new("logs.error", json!("Database connection timeout"))
            .with_sender("db_client"),
    );
    println!("  'logs.error' -> routed to: {:?}", handlers);

    let handlers = bus
        .publish(TopicMessage::new("metrics.cpu", json!({"usage": 78.5})).with_sender("monitor"));
    println!("  'metrics.cpu' -> routed to: {:?}", handlers);

    let handlers = bus.publish(
        TopicMessage::new("notifications.email", json!({"to": "user@example.com"}))
            .with_sender("notifier"),
    );
    println!(
        "  'notifications.email' -> routed to: {:?} (unrouted!)",
        handlers
    );

    let handlers = bus.publish(
        TopicMessage::new("webhooks.github", json!({"event": "push"}))
            .with_sender("webhook_handler"),
    );
    println!(
        "  'webhooks.github' -> routed to: {:?} (unrouted!)",
        handlers
    );

    // Check stats
    let stats = bus.stats();
    println!("\nBus statistics:");
    println!("  Total messages:    {}", stats.message_count);
    println!("  Unique topics:     {}", stats.topic_count);
    println!("  Routing rules:     {}", stats.route_count);
    println!("  Dead letter queue: {}", stats.dlq_count);

    // Subscribe to specific messages
    let order_msgs = bus.subscribe(TopicFilter::Prefix("orders.".to_string()));
    println!("\nOrder messages in bus: {}", order_msgs.len());
    for msg in &order_msgs {
        println!(
            "  [seq: {}] sender={:?}, payload={}",
            msg.sequence, msg.sender, msg.payload
        );
    }

    // Check dead letters
    println!(
        "\nDead letter queue has {} unrouted messages",
        bus.dead_letters().len()
    );

    // Drain the dead letters for inspection
    let dead = bus.dead_letters_mut().drain();
    for msg in &dead {
        println!(
            "  Dead letter: topic='{}', sender={:?}",
            msg.topic, msg.sender
        );
    }

    // -----------------------------------------------------------------------
    // 7. Real LLM Demo — LLM-driven topic routing
    // -----------------------------------------------------------------------
    println!("\n--- 7. Real LLM Demo ---");
    println!("Ask an LLM to classify a message into a topic.\n");

    let model = shared::get_chat_model(vec![
        "This message is about an order being placed, so it should be routed to the topic 'orders.created'.".into(),
    ]);
    let messages = vec![
        Message::system("You are a message router. Given a message, classify it into one of these topics: orders.created, orders.shipped, logs.info, logs.error, metrics.cpu"),
        Message::human("Customer #42 just placed an order for 3 items totaling $127.50"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM topic classification: {}", gen.message.content().text());
    }

    println!("\n=== Topic Channels Example Complete ===");
    Ok(())
}
