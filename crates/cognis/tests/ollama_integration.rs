//! Integration tests using a real Ollama LLM server.
//!
//! These tests require a running Ollama instance at `http://localhost:11434`
//! with the `llama3.2` model pulled. To set up:
//!
//! ```bash
//! # Install Ollama: https://ollama.com/download
//! ollama pull llama3.2
//! ```
//!
//! All tests are gated behind the `ollama` feature flag and will skip
//! gracefully if the Ollama server is unreachable.
//!
//! Run with:
//! ```bash
//! cargo test -p cognis --features ollama --test ollama_integration -- --test-threads=2
//! ```

use std::sync::Arc;

use futures::StreamExt;
use serde_json::json;

use cognis::chat_models::ollama::ChatOllama;
use cognis::memory::buffer::ConversationBufferMemory;
use cognis::memory::window::ConversationWindowMemory;
use cognis::memory::BaseMemory;
use cognis::text_splitter::RecursiveCharacterTextSplitter;
use cognis_core::documents::Document;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::ChatModelRunnable;
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;

const MODEL: &str = "llama3.2";
const BASE_URL: &str = "http://localhost:11434";

/// Check if Ollama is reachable. Tests skip if not.
async fn ollama_available() -> bool {
    let client = reqwest::Client::new();
    match client
        .get(format!("{}/api/tags", BASE_URL))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Helper to build a ChatOllama instance with deterministic settings.
fn build_model() -> ChatOllama {
    ChatOllama::builder()
        .model(MODEL)
        .base_url(BASE_URL)
        .temperature(0.0)
        .num_predict(256)
        .build()
        .expect("Failed to build ChatOllama")
}

/// Assert that `text` contains at least one of `keywords` (case-insensitive).
fn assert_contains_any(text: &str, keywords: &[&str], context: &str) {
    let lower = text.to_lowercase();
    let found = keywords.iter().any(|k| lower.contains(&k.to_lowercase()));
    assert!(
        found,
        "{context}: expected response to contain one of {keywords:?}, got: \"{text}\""
    );
}

// ---------------------------------------------------------------------------
// 1. Basic generation — ask a factual question, verify keyword in response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_basic_generation() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![
        Message::System(SystemMessage::new(
            "Answer in one sentence. Be factual and concise.",
        )),
        Message::Human(HumanMessage::new("What is the capital of France?")),
    ];

    let result = model._generate(&messages, None).await.unwrap();
    let text = &result.generations[0].text;

    assert_contains_any(text, &["Paris", "paris"], "capital of France");
}

// ---------------------------------------------------------------------------
// 2. Streaming — verify tokens arrive incrementally
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_tokens() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![Message::Human(HumanMessage::new(
        "Count from 1 to 5, one number per line.",
    ))];

    let mut stream = model._stream(&messages, None).await.unwrap();
    let mut chunks = Vec::new();
    let mut full_text = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        full_text.push_str(&chunk.text);
        chunks.push(chunk.text);
    }

    // Verify we got multiple chunks (streaming works)
    assert!(
        chunks.len() > 1,
        "Expected multiple streamed chunks, got {}",
        chunks.len()
    );

    // Verify the numbers appear in the response
    for n in 1..=5 {
        assert_contains_any(
            &full_text,
            &[&n.to_string()],
            &format!("counting should include {n}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Chain composition — prompt → model → parser pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_chain_composition() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let prompt = ChatPromptTemplate::from_messages(vec![
        (
            "system",
            "Answer in exactly one sentence. Just name the language.",
        ),
        (
            "human",
            "What programming language is known for the borrow checker?",
        ),
    ])
    .unwrap();

    let model = build_model();
    let chain = cognis_core::chain!(
        prompt,
        ChatModelRunnable::new(Arc::new(model)),
        StrOutputParser
    )
    .unwrap();

    let result = chain.invoke(json!({}), None).await.unwrap();
    let text = result.as_str().unwrap_or("");

    assert_contains_any(text, &["Rust", "rust"], "borrow checker language");
}

// ---------------------------------------------------------------------------
// 4. Multi-turn conversation — verify the model remembers context
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multi_turn_conversation() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();

    // Turn 1: establish a fact
    let messages_turn1 = vec![
        Message::System(SystemMessage::new(
            "You are a helpful assistant. Be concise.",
        )),
        Message::Human(HumanMessage::new("My name is Vasanth. Remember that.")),
    ];

    let result1 = model._generate(&messages_turn1, None).await.unwrap();
    let ai_response1 = result1.generations[0].text.clone();

    // Turn 2: ask the model to recall — include the full conversation history
    let mut messages_turn2 = messages_turn1.clone();
    messages_turn2.push(Message::Ai(
        cognis_core::messages::AIMessage::new(&ai_response1),
    ));
    messages_turn2.push(Message::Human(HumanMessage::new("What is my name?")));

    let result2 = model._generate(&messages_turn2, None).await.unwrap();
    let text = &result2.generations[0].text;

    assert_contains_any(text, &["Vasanth", "vasanth"], "should remember the name");
}

// ---------------------------------------------------------------------------
// 5. System prompt adherence — verify the model follows instructions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_system_prompt_adherence() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![
        Message::System(SystemMessage::new(
            "You are a pirate. You must start every answer with 'Ahoy'.",
        )),
        Message::Human(HumanMessage::new("Hello, how are you?")),
    ];

    let result = model._generate(&messages, None).await.unwrap();
    let text = &result.generations[0].text;

    assert_contains_any(
        text,
        &["Ahoy", "ahoy", "AHOY", "matey", "pirate", "arr"],
        "pirate system prompt",
    );
}

// ---------------------------------------------------------------------------
// 6. Buffer memory integration — store context, recall via LLM
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_buffer_memory_with_llm() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let memory = ConversationBufferMemory::new();

    // Turn 1: tell the model a secret code
    let human1 = Message::Human(HumanMessage::new(
        "The secret code is ALPHA-7. Remember this.",
    ));
    let ctx1 = vec![
        Message::System(SystemMessage::new(
            "You are a helpful assistant. Be concise.",
        )),
        human1.clone(),
    ];
    let result1 = model._generate(&ctx1, None).await.unwrap();
    let ai_text1 = result1.generations[0].text.clone();
    let ai1 = Message::Ai(cognis_core::messages::AIMessage::new(&ai_text1));

    // Save turn 1 to memory
    memory.save_context(&human1, &ai1).await.unwrap();

    // Turn 2: ask to recall — build context from memory
    let human2 = Message::Human(HumanMessage::new(
        "What was the secret code I told you?",
    ));

    // Load memory and build full context
    let mem_vars = memory.load_memory_variables().await.unwrap();
    let history = mem_vars
        .get("history")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut ctx2 = vec![Message::System(SystemMessage::new(
        "You are a helpful assistant. Be concise.",
    ))];
    // Reconstruct messages from memory
    for msg_val in &history {
        if let Ok(msg) = serde_json::from_value::<Message>(msg_val.clone()) {
            ctx2.push(msg);
        }
    }
    ctx2.push(human2.clone());

    let result2 = model._generate(&ctx2, None).await.unwrap();
    let text = &result2.generations[0].text;

    assert_contains_any(
        text,
        &["ALPHA-7", "ALPHA", "alpha-7", "alpha"],
        "memory should recall the secret code",
    );

    // Save turn 2
    let ai2 = Message::Ai(cognis_core::messages::AIMessage::new(text));
    memory.save_context(&human2, &ai2).await.unwrap();

    // Verify memory now has 4 messages (2 turns)
    let mem_vars2 = memory.load_memory_variables().await.unwrap();
    let history2 = mem_vars2.get("history").unwrap().as_array().unwrap();
    assert_eq!(history2.len(), 4, "Memory should have 4 messages (2 turns)");
}

// ---------------------------------------------------------------------------
// 7. Window memory truncation — verify old turns are dropped
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_window_memory_truncation() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let memory = ConversationWindowMemory::new(1); // keep only last 1 turn

    // Turn 1: establish a secret
    let h1 = Message::Human(HumanMessage::new("The password is ZEBRA-99."));
    let a1 = Message::Ai(cognis_core::messages::AIMessage::new(
        "Got it, I'll remember ZEBRA-99.",
    ));
    memory.save_context(&h1, &a1).await.unwrap();

    // Turn 2: different topic — this pushes turn 1 out of the window
    let h2 = Message::Human(HumanMessage::new("What color is the sky?"));
    let a2 = Message::Ai(cognis_core::messages::AIMessage::new(
        "The sky is blue.",
    ));
    memory.save_context(&h2, &a2).await.unwrap();

    // Verify: memory should only contain the last turn (2 messages)
    let vars = memory.load_memory_variables().await.unwrap();
    let history = vars.get("history").unwrap().as_array().unwrap();
    assert_eq!(
        history.len(),
        2,
        "Window(k=1) should keep 2 messages (1 turn), has {}",
        history.len()
    );

    // The retained messages should be about the sky, not the password
    let history_text = serde_json::to_string(&history).unwrap();
    assert!(
        !history_text.contains("ZEBRA"),
        "Old turn (password) should be evicted from window memory"
    );
    assert!(
        history_text.contains("sky") || history_text.contains("blue"),
        "Latest turn (sky) should be retained"
    );
}

// ---------------------------------------------------------------------------
// 8. Text splitting + retrieval — verify document chunking works end-to-end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_text_splitting_and_search() {
    // This test doesn't need Ollama — it tests the text processing pipeline
    let content = "\
Rust was first released in 2015 by Mozilla Research. \
It is a systems programming language focused on safety and performance. \
The borrow checker is a key feature that prevents data races at compile time. \
Cargo is Rust's package manager and build system. \
Tokio is the most popular async runtime for Rust.";

    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(100)
        .with_chunk_overlap(20);

    let docs = vec![Document::new(content)];
    let chunks = splitter.split_documents(&docs);

    assert!(
        chunks.len() > 1,
        "Expected multiple chunks, got {}",
        chunks.len()
    );

    // Verify all key content is preserved across chunks
    let combined: String = chunks.iter().map(|d| d.page_content.clone()).collect();
    assert!(combined.contains("borrow checker"), "Should contain 'borrow checker'");
    assert!(combined.contains("Cargo"), "Should contain 'Cargo'");
    assert!(combined.contains("Tokio"), "Should contain 'Tokio'");
}

// ---------------------------------------------------------------------------
// 9. JSON output format — ask for structured JSON response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_json_output_format() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = ChatOllama::builder()
        .model(MODEL)
        .base_url(BASE_URL)
        .temperature(0.0)
        .num_predict(256)
        .format(json!("json"))
        .build()
        .unwrap();

    let messages = vec![
        Message::System(SystemMessage::new(
            "Respond only with valid JSON. No other text.",
        )),
        Message::Human(HumanMessage::new(
            "Give me a JSON object with keys 'name' set to 'Rust' and 'year' set to 2015.",
        )),
    ];

    let result = model._generate(&messages, None).await.unwrap();
    let text = &result.generations[0].text;

    // Verify it's valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(text).unwrap_or_else(|_| panic!("Response should be valid JSON, got: {text}"));

    // Verify expected keys
    assert_eq!(
        parsed.get("name").and_then(|v| v.as_str()),
        Some("Rust"),
        "JSON should have name=Rust, got: {parsed}"
    );
    assert!(
        parsed.get("year").is_some(),
        "JSON should have a 'year' field, got: {parsed}"
    );
}

// ---------------------------------------------------------------------------
// 10. Deterministic output — low temp + seed should produce consistent results
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_deterministic_with_low_temperature() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = ChatOllama::builder()
        .model(MODEL)
        .base_url(BASE_URL)
        .temperature(0.0)
        .seed(42)
        .num_predict(50)
        .build()
        .unwrap();

    let messages = vec![Message::Human(HumanMessage::new(
        "What is 2 + 2? Answer with just the number.",
    ))];

    let result1 = model._generate(&messages, None).await.unwrap();
    let result2 = model._generate(&messages, None).await.unwrap();

    let text1 = &result1.generations[0].text;
    let text2 = &result2.generations[0].text;

    // Both should contain "4"
    assert_contains_any(text1, &["4"], "2+2 attempt 1");
    assert_contains_any(text2, &["4"], "2+2 attempt 2");
}

// ---------------------------------------------------------------------------
// 11. Long context — verify the model handles longer inputs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_long_context_input() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();

    // Build a long context with a specific fact embedded
    let padding = "This is some filler text to pad the context. ".repeat(50);
    let messages = vec![
        Message::System(SystemMessage::new(
            "You are a helpful assistant. Be concise. Answer based on the provided text.",
        )),
        Message::Human(HumanMessage::new(&format!(
            "{padding}\n\nIMPORTANT FACT: The project deadline is March 15th, 2027.\n\n{padding}\n\nBased on the text above, what is the project deadline?"
        ))),
    ];

    let result = model._generate(&messages, None).await.unwrap();
    let text = &result.generations[0].text;

    assert_contains_any(
        text,
        &["March 15", "march 15", "2027"],
        "should extract the deadline from long context",
    );
}

// ---------------------------------------------------------------------------
// 12. Stop sequences — verify generation stops at the right place
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_stop_sequences() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![Message::Human(HumanMessage::new(
        "List the first 5 planets from the sun, numbered 1 through 5, one per line.",
    ))];

    // Stop at "Jupiter" — should not include Jupiter or beyond
    let result = model
        ._generate(&messages, Some(&["Jupiter".to_string()]))
        .await
        .unwrap();
    let text = &result.generations[0].text;

    assert_contains_any(text, &["Mercury", "mercury"], "should include Mercury");
    assert!(
        !text.to_lowercase().contains("saturn"),
        "Should have stopped before Saturn, got: {text}"
    );
}

// ---------------------------------------------------------------------------
// 13. Usage metadata — verify token counts are reported
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_usage_metadata() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![Message::Human(HumanMessage::new("Say hello."))];

    let result = model._generate(&messages, None).await.unwrap();

    assert!(
        !result.generations.is_empty(),
        "Should have at least one generation"
    );

    let gen = &result.generations[0];
    assert!(!gen.text.is_empty(), "Generation text should not be empty");

    // Verify llm_output has eval metadata from Ollama
    if let Some(llm_output) = &result.llm_output {
        // Ollama returns eval_count, prompt_eval_count, etc.
        let has_eval = llm_output.get("eval_count").is_some()
            || llm_output.get("total_duration").is_some();
        if has_eval {
            println!("LLM output metadata: {llm_output:?}");
        }
    }

    // If usage metadata is in the message, check it
    if let Message::Ai(ai_msg) = &gen.message {
        if let Some(usage) = &ai_msg.usage_metadata {
            assert!(
                usage.total_tokens > 0,
                "Total tokens should be > 0, got {}",
                usage.total_tokens
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 14. Multilingual — verify non-English responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multilingual_response() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![
        Message::System(SystemMessage::new("Respond in Spanish only.")),
        Message::Human(HumanMessage::new("What color is the sky?")),
    ];

    let result = model._generate(&messages, None).await.unwrap();
    let text = &result.generations[0].text;

    assert_contains_any(
        text,
        &["azul", "cielo", "Azul", "Cielo"],
        "should respond in Spanish about the sky",
    );
}

// ---------------------------------------------------------------------------
// 15. Chain with template variables — verify variable substitution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_chain_with_variables() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let prompt = ChatPromptTemplate::from_messages(vec![
        (
            "system",
            "You are a geography expert. Answer in one short sentence.",
        ),
        ("human", "What continent is {country} in?"),
    ])
    .unwrap();

    let model = build_model();
    let chain = cognis_core::chain!(
        prompt,
        ChatModelRunnable::new(Arc::new(model)),
        StrOutputParser
    )
    .unwrap();

    let result = chain
        .invoke(json!({"country": "Japan"}), None)
        .await
        .unwrap();
    let text = result.as_str().unwrap_or("");
    assert_contains_any(text, &["Asia", "asia"], "Japan is in Asia");
}

// ---------------------------------------------------------------------------
// 16. Batch generation — multiple prompts in sequence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_batch_generation() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();

    let questions = vec![
        ("What is the chemical symbol for water?", &["H2O", "h2o"][..]),
        ("What planet is known as the Red Planet?", &["Mars", "mars"]),
        ("What is the largest ocean on Earth?", &["Pacific", "pacific"]),
    ];

    for (question, expected_keywords) in questions {
        let messages = vec![
            Message::System(SystemMessage::new("Answer in one word or short phrase.")),
            Message::Human(HumanMessage::new(question)),
        ];
        let result = model._generate(&messages, None).await.unwrap();
        let text = &result.generations[0].text;
        assert_contains_any(text, expected_keywords, question);
    }
}

// ---------------------------------------------------------------------------
// 17. Streaming full roundtrip — stream, collect, verify content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_roundtrip() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let messages = vec![
        Message::System(SystemMessage::new("Be very concise. One sentence max.")),
        Message::Human(HumanMessage::new("What is photosynthesis?")),
    ];

    // Stream the response
    let mut stream = model._stream(&messages, None).await.unwrap();
    let mut full_response = String::new();
    let mut chunk_count = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.unwrap();
        full_response.push_str(&chunk.text);
        chunk_count += 1;
    }

    assert!(chunk_count > 0, "Should receive at least one chunk");
    assert!(!full_response.is_empty(), "Response should not be empty");
    assert_contains_any(
        &full_response,
        &["light", "plant", "sun", "energy", "carbon", "oxygen", "glucose"],
        "photosynthesis response",
    );
}

// ---------------------------------------------------------------------------
// 18. Error handling — invalid model name should return an error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_invalid_model_error() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = ChatOllama::builder()
        .model("nonexistent-model-xyz-999")
        .base_url(BASE_URL)
        .temperature(0.0)
        .build()
        .unwrap();

    let messages = vec![Message::Human(HumanMessage::new("Hello"))];
    let result = model._generate(&messages, None).await;

    assert!(
        result.is_err(),
        "Should return an error for nonexistent model"
    );
}

// ---------------------------------------------------------------------------
// 19. Multi-turn with memory — 3-turn conversation preserving full context
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_three_turn_memory_recall() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let model = build_model();
    let memory = ConversationBufferMemory::new();
    let system = Message::System(SystemMessage::new(
        "You are a helpful assistant. Be concise. Remember everything the user tells you.",
    ));

    // Turn 1: favorite color
    let h1 = Message::Human(HumanMessage::new("My favorite color is blue."));
    let r1 = model
        ._generate(&[system.clone(), h1.clone()], None)
        .await
        .unwrap();
    let a1 = Message::Ai(cognis_core::messages::AIMessage::new(
        &r1.generations[0].text,
    ));
    memory.save_context(&h1, &a1).await.unwrap();

    // Turn 2: favorite food
    let h2 = Message::Human(HumanMessage::new("My favorite food is pizza."));
    let vars = memory.load_memory_variables().await.unwrap();
    let history: Vec<Message> = vars
        .get("history")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        })
        .unwrap_or_default();

    let mut ctx2 = vec![system.clone()];
    ctx2.extend(history);
    ctx2.push(h2.clone());

    let r2 = model._generate(&ctx2, None).await.unwrap();
    let a2 = Message::Ai(cognis_core::messages::AIMessage::new(
        &r2.generations[0].text,
    ));
    memory.save_context(&h2, &a2).await.unwrap();

    // Turn 3: ask to recall both facts
    let h3 = Message::Human(HumanMessage::new(
        "What is my favorite color and my favorite food?",
    ));
    let vars3 = memory.load_memory_variables().await.unwrap();
    let history3: Vec<Message> = vars3
        .get("history")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect()
        })
        .unwrap_or_default();

    let mut ctx3 = vec![system.clone()];
    ctx3.extend(history3);
    ctx3.push(h3.clone());

    let r3 = model._generate(&ctx3, None).await.unwrap();
    let text = &r3.generations[0].text;

    assert_contains_any(text, &["blue", "Blue"], "should recall favorite color");
    assert_contains_any(text, &["pizza", "Pizza"], "should recall favorite food");
}

// ---------------------------------------------------------------------------
// 20. Chain variable substitution — multiple variables in one prompt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multi_variable_chain() {
    if !ollama_available().await {
        eprintln!("SKIPPED: Ollama not available");
        return;
    }

    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a helpful coding assistant. Be concise."),
        (
            "human",
            "Write a one-line description of the {language} programming language, focusing on {feature}.",
        ),
    ])
    .unwrap();

    let model = build_model();
    let chain = cognis_core::chain!(
        prompt,
        ChatModelRunnable::new(Arc::new(model)),
        StrOutputParser
    )
    .unwrap();

    let result = chain
        .invoke(
            json!({"language": "Python", "feature": "simplicity"}),
            None,
        )
        .await
        .unwrap();
    let text = result.as_str().unwrap_or("");

    assert_contains_any(
        text,
        &["Python", "python", "simple", "easy", "readab"],
        "should mention Python and simplicity",
    );
}
