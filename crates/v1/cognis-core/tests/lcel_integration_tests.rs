//! LCEL (LangChain Expression Language) composition integration tests.
//!
//! These tests verify end-to-end composition of Runnables using pipe, chain!,
//! RunnableParallel, RunnableBranch, assign, pick, retry, and fallbacks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis_core::chain;
use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::{ChatModelRunnable, FakeListChatModel};
use cognis_core::output_parsers::{JsonOutputParser, StrOutputParser};
use cognis_core::prompts::base::PromptTemplate;
use cognis_core::prompts::chat::ChatPromptTemplate;
use cognis_core::runnables::config::RunnableConfig;
use cognis_core::runnables::{
    Runnable, RunnableBranch, RunnableExt, RunnableLambda, RunnableParallel, RunnablePassthrough,
    RunnablePick,
};

// ---------------------------------------------------------------------------
// Test 1: Simple pipe chain
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_simple_pipe_chain() {
    let chain = RunnableLambda::new("add1", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    })
    .pipe(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }))
    .unwrap();

    let result = chain.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!(12)); // (5+1)*2 = 12
}

#[tokio::test]
async fn test_pipe_three_steps() {
    let chain = RunnableLambda::new("add1", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    })
    .pipe(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }))
    .unwrap()
    .pipe(RunnableLambda::new("to_string", |v: Value| async move {
        Ok(json!(format!("result={}", v.as_i64().unwrap())))
    }))
    .unwrap();

    let result = chain.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!("result=12"));
}

// ---------------------------------------------------------------------------
// Test 2: chain! macro
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_chain_macro() {
    let step1 = RunnableLambda::new("add10", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 10))
    });
    let step2 = RunnableLambda::new("multiply3", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 3))
    });
    let step3 = RunnableLambda::new("subtract1", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() - 1))
    });

    let seq = chain!(step1, step2, step3).unwrap();
    let result = seq.invoke(json!(2), None).await.unwrap();
    // (2 + 10) * 3 - 1 = 35
    assert_eq!(result, json!(35));
}

#[tokio::test]
async fn test_chain_macro_single_step() {
    let step = RunnableLambda::new("identity", |v: Value| async move { Ok(v) });
    let seq = chain!(step).unwrap();
    let result = seq.invoke(json!("hello"), None).await.unwrap();
    assert_eq!(result, json!("hello"));
}

// ---------------------------------------------------------------------------
// Test 3: Prompt template -> model -> output parser (LCEL chain)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_prompt_model_parser_chain() {
    // Build the prompt template
    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    .unwrap();

    // Build a fake chat model that always returns a fixed response
    let model = ChatModelRunnable::new(Arc::new(FakeListChatModel::new(vec![
        "The answer is 42.".to_string()
    ])));

    // StrOutputParser extracts the string
    let parser = StrOutputParser;

    // Compose: prompt | model | parser
    let chain = prompt.pipe(model).unwrap().pipe(parser).unwrap();

    let result = chain
        .invoke(json!({"question": "What is the meaning of life?"}), None)
        .await
        .unwrap();

    // The StrOutputParser will convert the AIMessage JSON to a string representation.
    // The result will be a JSON string containing the serialized AIMessage.
    assert!(result.is_string());
    let text = result.as_str().unwrap();
    assert!(
        text.contains("The answer is 42."),
        "Expected output to contain the model response, got: {}",
        text
    );
}

#[tokio::test]
async fn test_prompt_template_as_runnable() {
    let prompt = PromptTemplate::from_template("Hello, {name}! You are {age} years old.");
    let result = prompt
        .invoke(json!({"name": "Alice", "age": 30}), None)
        .await
        .unwrap();
    assert_eq!(result, json!("Hello, Alice! You are 30 years old."));
}

#[tokio::test]
async fn test_chat_prompt_template_as_runnable() {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are {role}"),
        ("human", "{question}"),
    ])
    .unwrap();

    let result = prompt
        .invoke(
            json!({"role": "a pirate", "question": "Where is the treasure?"}),
            None,
        )
        .await
        .unwrap();

    assert!(result.is_array());
    let messages = result.as_array().unwrap();
    assert_eq!(messages.len(), 2);
}

// ---------------------------------------------------------------------------
// Test 4: RunnableParallel
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_runnable_parallel() {
    let upper = RunnableLambda::new("upper", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().to_uppercase()))
    });
    let length = RunnableLambda::new("length", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().len()))
    });
    let reversed = RunnableLambda::new("reversed", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().chars().rev().collect::<String>()))
    });

    let mut steps: HashMap<String, Arc<dyn Runnable>> = HashMap::new();
    steps.insert("upper".to_string(), Arc::new(upper));
    steps.insert("length".to_string(), Arc::new(length));
    steps.insert("reversed".to_string(), Arc::new(reversed));

    let parallel = RunnableParallel::new(steps);
    let result = parallel.invoke(json!("hello"), None).await.unwrap();

    assert_eq!(result["upper"], json!("HELLO"));
    assert_eq!(result["length"], json!(5));
    assert_eq!(result["reversed"], json!("olleh"));
}

#[tokio::test]
async fn test_parallel_in_pipe_chain() {
    // First step: extract a field, then parallel process
    let extract = RunnableLambda::new("extract", |v: Value| async move {
        Ok(json!(v["text"].as_str().unwrap_or("")))
    });

    let mut steps: HashMap<String, Arc<dyn Runnable>> = HashMap::new();
    steps.insert(
        "upper".to_string(),
        Arc::new(RunnableLambda::new("upper", |v: Value| async move {
            Ok(json!(v.as_str().unwrap().to_uppercase()))
        })),
    );
    steps.insert(
        "length".to_string(),
        Arc::new(RunnableLambda::new("length", |v: Value| async move {
            Ok(json!(v.as_str().unwrap().len()))
        })),
    );

    let parallel = RunnableParallel::new(steps);
    let chain = extract.pipe(parallel).unwrap();

    let result = chain.invoke(json!({"text": "world"}), None).await.unwrap();

    assert_eq!(result["upper"], json!("WORLD"));
    assert_eq!(result["length"], json!(5));
}

// ---------------------------------------------------------------------------
// Test 5: assign + pick
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_assign_and_pick() {
    // Start with passthrough, assign a computed key, then pick specific keys
    let passthrough = RunnablePassthrough::new();

    let mut mapping: HashMap<String, Arc<dyn Runnable>> = HashMap::new();
    mapping.insert(
        "name_upper".to_string(),
        Arc::new(RunnableLambda::new("upper_name", |v: Value| async move {
            let name = v["name"].as_str().unwrap_or("");
            Ok(json!(name.to_uppercase()))
        })),
    );

    // passthrough.assign(mapping) creates passthrough | RunnableAssign
    let chain = passthrough
        .assign(mapping)
        .unwrap()
        .pipe(RunnablePick::many(vec![
            "name".to_string(),
            "name_upper".to_string(),
        ]))
        .unwrap();

    let result = chain
        .invoke(json!({"name": "alice", "age": 30}), None)
        .await
        .unwrap();

    assert_eq!(result["name"], json!("alice"));
    assert_eq!(result["name_upper"], json!("ALICE"));
    // "age" should be dropped by pick
    assert!(result.get("age").is_none());
}

#[tokio::test]
async fn test_pick_single_key() {
    let passthrough = RunnablePassthrough::new();
    let chain = passthrough.pick(vec!["name".to_string()]).unwrap();

    let result = chain
        .invoke(json!({"name": "Bob", "age": 25}), None)
        .await
        .unwrap();

    // Single key pick returns the value directly
    assert_eq!(result, json!("Bob"));
}

// ---------------------------------------------------------------------------
// Test 6: RunnableBranch (conditional routing)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_runnable_branch_first_condition() {
    let is_positive = RunnableLambda::new("is_positive", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() > 0))
    });
    let positive_handler = RunnableLambda::new("positive", |v: Value| async move {
        Ok(json!(format!("{} is positive", v.as_i64().unwrap())))
    });

    let is_zero = RunnableLambda::new("is_zero", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() == 0))
    });
    let zero_handler =
        RunnableLambda::new("zero", |_v: Value| async move { Ok(json!("it's zero")) });

    let default_handler = RunnableLambda::new("negative", |v: Value| async move {
        Ok(json!(format!("{} is negative", v.as_i64().unwrap())))
    });

    let branch = RunnableBranch::new(
        vec![
            (
                Arc::new(is_positive) as Arc<dyn Runnable>,
                Arc::new(positive_handler) as Arc<dyn Runnable>,
            ),
            (
                Arc::new(is_zero) as Arc<dyn Runnable>,
                Arc::new(zero_handler) as Arc<dyn Runnable>,
            ),
        ],
        Arc::new(default_handler) as Arc<dyn Runnable>,
    );

    // Positive number
    let result = branch.invoke(json!(5), None).await.unwrap();
    assert_eq!(result, json!("5 is positive"));

    // Zero
    let result = branch.invoke(json!(0), None).await.unwrap();
    assert_eq!(result, json!("it's zero"));

    // Negative number
    let result = branch.invoke(json!(-3), None).await.unwrap();
    assert_eq!(result, json!("-3 is negative"));
}

#[tokio::test]
async fn test_branch_default_fallthrough() {
    let always_false = RunnableLambda::new("false", |_v: Value| async move { Ok(json!(false)) });
    let never_handler =
        RunnableLambda::new(
            "never",
            |_v: Value| async move { Ok(json!("should not reach")) },
        );
    let default_handler = RunnableLambda::new("default", |v: Value| async move {
        Ok(json!(format!("default: {}", v)))
    });

    let branch = RunnableBranch::new(
        vec![(
            Arc::new(always_false) as Arc<dyn Runnable>,
            Arc::new(never_handler) as Arc<dyn Runnable>,
        )],
        Arc::new(default_handler) as Arc<dyn Runnable>,
    );

    let result = branch.invoke(json!(42), None).await.unwrap();
    assert_eq!(result, json!("default: 42"));
}

// ---------------------------------------------------------------------------
// Test 7: Pipe with retry and fallbacks
// ---------------------------------------------------------------------------

/// A runnable that fails N times before succeeding.
struct FailNTimesThenSucceed {
    fail_count: u32,
    attempts: AtomicU32,
}

impl FailNTimesThenSucceed {
    fn new(fail_count: u32) -> Self {
        Self {
            fail_count,
            attempts: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Runnable for FailNTimesThenSucceed {
    fn name(&self) -> &str {
        "FailNTimesThenSucceed"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        if attempt < self.fail_count {
            Err(CognisError::Other(format!("attempt {} failed", attempt)))
        } else {
            Ok(json!(format!(
                "success after {} failures: {}",
                attempt, input
            )))
        }
    }
}

#[tokio::test]
async fn test_pipe_with_retry() {
    let step1 = RunnableLambda::new("add1", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    });

    // Fails twice then succeeds. With 3 retries it should work.
    let step2 = FailNTimesThenSucceed::new(2);
    let step2_with_retry = step2.with_retry(3, 1); // 1ms delay for test speed

    let chain = step1.pipe(step2_with_retry).unwrap();
    let result = chain.invoke(json!(10), None).await.unwrap();

    assert!(result.as_str().unwrap().contains("success"));
    assert!(result.as_str().unwrap().contains("11"));
}

#[tokio::test]
async fn test_pipe_with_fallbacks() {
    // A runnable that always fails
    let failing = RunnableLambda::new("failing", |_v: Value| async move {
        Err::<Value, _>(CognisError::Other("primary failed".into()))
    });

    // A fallback that succeeds
    let fallback = RunnableLambda::new("fallback", |v: Value| async move {
        Ok(json!(format!("fallback handled: {}", v)))
    });

    let with_fallbacks = failing.with_fallbacks(vec![Arc::new(fallback) as Arc<dyn Runnable>]);

    let step1 = RunnableLambda::new("prep", |v: Value| async move {
        Ok(json!(v.as_str().unwrap().to_uppercase()))
    });

    let chain = step1.pipe(with_fallbacks).unwrap();
    let result = chain.invoke(json!("hello"), None).await.unwrap();

    assert_eq!(result, json!("fallback handled: \"HELLO\""));
}

#[tokio::test]
async fn test_retry_and_fallbacks_combined() {
    // A runnable that always fails (even after retries)
    let always_fails = RunnableLambda::new("always_fails", |_v: Value| async move {
        Err::<Value, _>(CognisError::Other("nope".into()))
    });
    let retrying_fails = always_fails.with_retry(2, 1);

    // Fallback that succeeds
    let fallback = RunnableLambda::new("fallback", |v: Value| async move {
        Ok(json!({"recovered": true, "input": v}))
    });

    let chain = retrying_fails.with_fallbacks(vec![Arc::new(fallback) as Arc<dyn Runnable>]);

    let result = chain.invoke(json!("test"), None).await.unwrap();
    assert_eq!(result["recovered"], json!(true));
    assert_eq!(result["input"], json!("test"));
}

// ---------------------------------------------------------------------------
// Test 8: Output parsers as Runnables
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_str_output_parser_in_chain() {
    let make_string =
        RunnableLambda::new("make", |_v: Value| async move { Ok(json!("hello world")) });
    let parser = StrOutputParser;
    let chain = make_string.pipe(parser).unwrap();

    let result = chain.invoke(json!(null), None).await.unwrap();
    assert_eq!(result, json!("hello world"));
}

#[tokio::test]
async fn test_json_output_parser_in_chain() {
    let make_json_str = RunnableLambda::new("make", |_v: Value| async move {
        Ok(json!(r#"{"name": "Alice", "age": 30}"#))
    });
    let parser = JsonOutputParser::new();
    let chain = make_json_str.pipe(parser).unwrap();

    let result = chain.invoke(json!(null), None).await.unwrap();
    assert_eq!(result["name"], json!("Alice"));
    assert_eq!(result["age"], json!(30));
}

#[tokio::test]
async fn test_json_output_parser_with_markdown_fences() {
    let make_fenced = RunnableLambda::new("make", |_v: Value| async move {
        Ok(json!("```json\n{\"key\": \"value\"}\n```"))
    });
    let parser = JsonOutputParser::new();
    let chain = make_fenced.pipe(parser).unwrap();

    let result = chain.invoke(json!(null), None).await.unwrap();
    assert_eq!(result["key"], json!("value"));
}

// ---------------------------------------------------------------------------
// Test 9: Batch invocation through a composed chain
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_batch_through_pipe() {
    let chain = RunnableLambda::new("add1", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() + 1))
    })
    .pipe(RunnableLambda::new("double", |v: Value| async move {
        Ok(json!(v.as_i64().unwrap() * 2))
    }))
    .unwrap();

    let results = chain
        .batch(vec![json!(1), json!(2), json!(3)], None)
        .await
        .unwrap();

    assert_eq!(results, vec![json!(4), json!(6), json!(8)]);
}

// ---------------------------------------------------------------------------
// Test 10: Complex multi-step chain (prompt -> lambda "model" -> parser)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_complex_multi_step_chain() {
    // Simulate: prompt template -> "model" (lambda) -> JSON parser
    let prompt = PromptTemplate::from_template("Generate JSON for: {topic}");

    let fake_model = RunnableLambda::new("fake_model", |v: Value| async move {
        let prompt_text = v.as_str().unwrap_or("");
        // Simulate model producing JSON output based on the prompt
        if prompt_text.contains("animals") {
            Ok(json!(
                r#"{"category": "animals", "examples": ["cat", "dog"]}"#
            ))
        } else {
            Ok(json!(r#"{"category": "unknown", "examples": []}"#))
        }
    });

    let parser = JsonOutputParser::new();

    let chain = prompt.pipe(fake_model).unwrap().pipe(parser).unwrap();

    let result = chain
        .invoke(json!({"topic": "animals"}), None)
        .await
        .unwrap();

    assert_eq!(result["category"], json!("animals"));
    assert_eq!(result["examples"], json!(["cat", "dog"]));
}

// ---------------------------------------------------------------------------
// Test 11: RunnablePassthrough + assign in a realistic scenario
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_passthrough_assign_realistic() {
    // Simulates a common LCEL pattern:
    // {"question": "..."} -> assign(context=retriever) -> prompt -> model
    let passthrough = RunnablePassthrough::new();

    let mut mapping: HashMap<String, Arc<dyn Runnable>> = HashMap::new();
    mapping.insert(
        "context".to_string(),
        Arc::new(RunnableLambda::new("retriever", |v: Value| async move {
            let q = v["question"].as_str().unwrap_or("");
            Ok(json!(format!("Retrieved context for: {}", q)))
        })),
    );
    mapping.insert(
        "question_upper".to_string(),
        Arc::new(RunnableLambda::new("upper", |v: Value| async move {
            let q = v["question"].as_str().unwrap_or("");
            Ok(json!(q.to_uppercase()))
        })),
    );

    let chain = passthrough.assign(mapping).unwrap();

    let result = chain
        .invoke(json!({"question": "What is Rust?"}), None)
        .await
        .unwrap();

    // Original keys preserved
    assert_eq!(result["question"], json!("What is Rust?"));
    // Computed keys added
    assert_eq!(
        result["context"],
        json!("Retrieved context for: What is Rust?")
    );
    assert_eq!(result["question_upper"], json!("WHAT IS RUST?"));
}

// ---------------------------------------------------------------------------
// Test 12: Branch inside a pipe chain
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_branch_in_pipe_chain() {
    // Classify input first, then branch on classification
    let classifier = RunnableLambda::new("classify", |v: Value| async move {
        let n = v.as_i64().unwrap_or(0);
        Ok(json!({"value": n, "is_even": n % 2 == 0}))
    });

    let is_even_check = RunnableLambda::new("check_even", |v: Value| async move {
        Ok(json!(v["is_even"].as_bool().unwrap_or(false)))
    });
    let even_handler = RunnableLambda::new("even", |v: Value| async move {
        Ok(json!(format!("{} is even", v["value"])))
    });
    let odd_handler = RunnableLambda::new("odd", |v: Value| async move {
        Ok(json!(format!("{} is odd", v["value"])))
    });

    let branch = RunnableBranch::new(
        vec![(
            Arc::new(is_even_check) as Arc<dyn Runnable>,
            Arc::new(even_handler) as Arc<dyn Runnable>,
        )],
        Arc::new(odd_handler) as Arc<dyn Runnable>,
    );

    let chain = classifier.pipe(branch).unwrap();

    assert_eq!(
        chain.invoke(json!(4), None).await.unwrap(),
        json!("4 is even")
    );
    assert_eq!(
        chain.invoke(json!(7), None).await.unwrap(),
        json!("7 is odd")
    );
}
