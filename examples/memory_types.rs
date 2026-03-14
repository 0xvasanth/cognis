//! Memory Types Example
//!
//! Demonstrates the different conversation memory implementations:
//! - ConversationBufferMemory (stores all messages)
//! - ConversationWindowMemory (keeps last K turns)
//! - EntityMemory (tracks entities mentioned in conversation)
//! - TokenBufferMemory (trims by estimated token count)
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example memory_types`

mod shared;
use cognis::memory::token_buffer::{CharBasedTokenCounter, SimpleTokenCounter, TokenBufferMemory};
use cognis::memory::{
    BaseMemory, ConversationBufferMemory, ConversationWindowMemory, EntityMemory,
};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory Types Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ConversationBufferMemory
    // -----------------------------------------------------------------------
    println!("--- 1. ConversationBufferMemory ---");
    println!("Stores the complete conversation history without any trimming.\n");

    let buffer_mem = ConversationBufferMemory::new();

    // Add several conversation turns
    buffer_mem
        .save_context(
            &Message::human("Hello! My name is Alice."),
            &Message::ai("Hi Alice! Nice to meet you. How can I help?"),
        )
        .await?;
    buffer_mem
        .save_context(
            &Message::human("What is the capital of France?"),
            &Message::ai("The capital of France is Paris."),
        )
        .await?;
    buffer_mem
        .save_context(
            &Message::human("And what about Germany?"),
            &Message::ai("The capital of Germany is Berlin."),
        )
        .await?;

    let vars = buffer_mem.load_memory_variables().await?;
    let history = vars.get("history").unwrap().as_array().unwrap();
    println!("  Total messages stored: {}", history.len());
    println!("  Memory key: {}\n", buffer_mem.memory_key());

    // Show as formatted text
    let text_mem = ConversationBufferMemory::new().with_return_messages(false);
    text_mem
        .save_context(&Message::human("Hello!"), &Message::ai("Hi there!"))
        .await?;
    let text_vars = text_mem.load_memory_variables().await?;
    let text_history = text_vars.get("history").unwrap().as_str().unwrap();
    println!(
        "  As formatted text:\n  {}\n",
        text_history.replace('\n', "\n  ")
    );

    // Demonstrate clear
    buffer_mem.clear().await?;
    let after_clear = buffer_mem.load_memory_variables().await?;
    let cleared = after_clear.get("history").unwrap().as_array().unwrap();
    println!("  After clear: {} messages\n", cleared.len());

    // -----------------------------------------------------------------------
    // 2. ConversationWindowMemory
    // -----------------------------------------------------------------------
    println!("--- 2. ConversationWindowMemory ---");
    println!("Keeps only the last K conversation turns (pairs of messages).\n");

    let window_mem = ConversationWindowMemory::new(2); // keep last 2 turns

    println!("  Window size: 2 turns");
    println!("  Adding 4 conversation turns...\n");

    let turns = vec![
        (
            "Turn 1: What is Rust?",
            "Rust is a systems programming language.",
        ),
        (
            "Turn 2: Who created it?",
            "Graydon Hoare started Rust at Mozilla.",
        ),
        (
            "Turn 3: When was v1.0?",
            "Rust 1.0 was released on May 15, 2015.",
        ),
        (
            "Turn 4: What about async?",
            "Async/await was stabilized in Rust 1.39.",
        ),
    ];

    for (human, ai) in &turns {
        window_mem
            .save_context(&Message::human(*human), &Message::ai(*ai))
            .await?;
    }

    let vars = window_mem.load_memory_variables().await?;
    let history = vars.get("history").unwrap().as_array().unwrap();
    println!(
        "  Messages retained: {} (2 turns x 2 messages)",
        history.len()
    );
    println!("  Oldest turns (1 and 2) have been automatically dropped.\n");

    // -----------------------------------------------------------------------
    // 3. EntityMemory
    // -----------------------------------------------------------------------
    println!("--- 3. EntityMemory ---");
    println!("Tracks named entities mentioned in conversation.\n");

    let entity_mem = EntityMemory::new();

    entity_mem
        .save_context(
            &Message::human("Alice works at Google on the Rust compiler team."),
            &Message::ai("That sounds like an exciting role for Alice at Google!"),
        )
        .await?;

    entity_mem
        .save_context(
            &Message::human("Bob is Alice's manager. He also works at Google."),
            &Message::ai("It's great that Bob and Alice work together at Google."),
        )
        .await?;

    println!("  Entity count: {}", entity_mem.entity_count().await);
    let names = entity_mem.entity_names().await;
    println!("  Tracked entities: {:?}", names);

    // Get context for a query mentioning specific entities
    let context = entity_mem.get_context("Tell me about Alice and Bob.").await;
    println!(
        "\n  Entity context for 'Alice and Bob':\n  {}\n",
        context.replace('\n', "\n  ")
    );

    // Load full memory variables
    let vars = entity_mem.load_memory_variables().await?;
    let entity_text = vars.get("entities").unwrap().as_str().unwrap();
    println!(
        "  All entity summaries:\n  {}\n",
        entity_text.replace('\n', "\n  ")
    );

    // Demonstrate entity extraction directly
    let extracted = EntityMemory::extract_entities(
        "Charlie met Diana at the New York City conference hosted by OpenAI.",
    );
    println!("  Direct entity extraction from text: {:?}\n", extracted);

    // -----------------------------------------------------------------------
    // 4. TokenBufferMemory
    // -----------------------------------------------------------------------
    println!("--- 4. TokenBufferMemory ---");
    println!("Trims messages when estimated token count exceeds a limit.\n");

    // 4a. Using SimpleTokenCounter (word-based)
    println!("  4a. With SimpleTokenCounter (word-based heuristic):");
    let token_mem = TokenBufferMemory::new()
        .with_max_tokens(50)
        .with_counter(SimpleTokenCounter::new());

    token_mem
        .save_context(
            &Message::human("Tell me a long story about a brave knight who traveled across many lands."),
            &Message::ai("Once upon a time, there was a brave knight who rode through forests and mountains."),
        )
        .await?;

    println!(
        "    After first turn: {} tokens (limit: 50)",
        token_mem.total_tokens().await
    );
    println!("    Messages: {}\n", token_mem.get_messages().await.len());

    token_mem
        .save_context(
            &Message::human("What happened next in the story?"),
            &Message::ai("The knight discovered a hidden castle deep within an enchanted valley."),
        )
        .await?;

    println!(
        "    After second turn: {} tokens",
        token_mem.total_tokens().await
    );
    println!(
        "    Messages: {} (older messages trimmed to fit budget)\n",
        token_mem.get_messages().await.len()
    );

    // 4b. Using CharBasedTokenCounter
    println!("  4b. With CharBasedTokenCounter (4 chars per token):");
    let char_mem = TokenBufferMemory::new()
        .with_max_tokens(30)
        .with_counter(CharBasedTokenCounter::new(4.0));

    char_mem
        .save_context(
            &Message::human("Hello world!"),
            &Message::ai("Hi there, how can I help you today?"),
        )
        .await?;

    println!(
        "    Total tokens: {} (limit: 30)",
        char_mem.total_tokens().await
    );
    println!("    Messages: {}\n", char_mem.get_messages().await.len());

    // 4c. Using the builder pattern
    println!("  4c. Using the builder pattern:");
    let built_mem = TokenBufferMemory::builder()
        .max_tokens(100)
        .counter(SimpleTokenCounter::new())
        .memory_key("chat_log")
        .return_messages(false)
        .build();

    built_mem
        .save_context(&Message::human("Hello!"), &Message::ai("Hi! How are you?"))
        .await?;

    let vars = built_mem.load_memory_variables().await?;
    println!("    Memory key: {}", built_mem.memory_key());
    println!(
        "    Value (as text): {:?}\n",
        vars.get("chat_log").unwrap().as_str().unwrap()
    );

    // -----------------------------------------------------------------------
    // 5. Memory with Real LLM
    // -----------------------------------------------------------------------
    println!("--- 5. Memory with Real LLM ---");
    println!("Demonstrates memory-augmented conversation with an actual model.\n");

    let llm_mem = ConversationBufferMemory::new();

    let model = shared::get_chat_model(vec![
        "That's great! Rust is an excellent choice.".to_string(),
        "Your favorite programming language is Rust!".to_string(),
    ]);

    // Turn 1: User tells the model about their favorite language
    let user_msg_1 = Message::human("My favorite programming language is Rust");
    let ai_response_1 = model.invoke_messages(&[user_msg_1.clone()], None).await?;
    let ai_text_1 = ai_response_1.base.content.text();
    let ai_msg_1 = Message::ai(&ai_text_1);

    llm_mem.save_context(&user_msg_1, &ai_msg_1).await?;
    println!("  Turn 1:");
    println!("    User: My favorite programming language is Rust");
    println!("    AI:   {}\n", ai_text_1);

    // Turn 2: Ask the model to recall — include history from memory
    let vars = llm_mem.load_memory_variables().await?;
    let history_msgs = vars.get("history").unwrap().as_array().unwrap();

    // Build message list: history + new user question
    let mut messages: Vec<Message> = history_msgs
        .iter()
        .map(|v| serde_json::from_value::<Message>(v.clone()).unwrap())
        .collect();
    let user_msg_2 = Message::human("What's my favorite language?");
    messages.push(user_msg_2.clone());

    let ai_response_2 = model.invoke_messages(&messages, None).await?;
    let ai_text_2 = ai_response_2.base.content.text();
    let ai_msg_2 = Message::ai(&ai_text_2);

    llm_mem.save_context(&user_msg_2, &ai_msg_2).await?;
    println!("  Turn 2 (with memory context):");
    println!("    User: What's my favorite language?");
    println!("    AI:   {}\n", ai_text_2);

    // Show final memory contents
    let final_vars = llm_mem.load_memory_variables().await?;
    let final_history = final_vars.get("history").unwrap().as_array().unwrap();
    println!("  Memory now contains {} messages (2 turns)\n", final_history.len());

    println!("=== Done ===");
    Ok(())
}
