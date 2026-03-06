//! Tests for migrated modules: chat_sessions, chat_loaders, rate_limiters,
//! structured_query, indexing, tracers, load/serializable.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::stream;
use serde_json::{json, Value};
use uuid::Uuid;

// ============================================================
// Chat Sessions
// ============================================================

use rustchain_core::chat_sessions::ChatSession;
use rustchain_core::messages::{HumanMessage, Message};

#[test]
fn test_chat_session_new() {
    let msgs = vec![Message::Human(HumanMessage::new("hello"))];
    let session = ChatSession::new(msgs.clone());
    assert_eq!(session.messages.len(), 1);
    assert!(session.functions.is_empty());
}

#[test]
fn test_chat_session_with_functions() {
    let msgs = vec![Message::Human(HumanMessage::new("hi"))];
    let funcs = vec![json!({"name": "search", "parameters": {}})];
    let session = ChatSession::new(msgs).with_functions(funcs.clone());
    assert_eq!(session.functions.len(), 1);
    assert_eq!(session.functions[0]["name"], "search");
}

#[test]
fn test_chat_session_default() {
    let session = ChatSession::default();
    assert!(session.messages.is_empty());
    assert!(session.functions.is_empty());
}

#[test]
fn test_chat_session_serde_roundtrip() {
    let msgs = vec![Message::Human(HumanMessage::new("test"))];
    let session = ChatSession::new(msgs).with_functions(vec![json!({"name": "fn1"})]);
    let json_str = serde_json::to_string(&session).unwrap();
    let deserialized: ChatSession = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.messages.len(), 1);
    assert_eq!(deserialized.functions.len(), 1);
}

#[test]
fn test_chat_session_functions_skipped_when_empty() {
    let session = ChatSession::new(vec![]);
    let val = serde_json::to_value(&session).unwrap();
    // functions should be skipped when empty
    assert!(val.get("functions").is_none());
}

// ============================================================
// Chat Loaders
// ============================================================

use rustchain_core::chat_loaders::BaseChatLoader;
use rustchain_core::chat_loaders::ChatSessionStream;
use rustchain_core::error::Result;

struct TestChatLoader {
    sessions: Vec<ChatSession>,
}

#[async_trait]
impl BaseChatLoader for TestChatLoader {
    async fn lazy_load(&self) -> Result<ChatSessionStream> {
        let sessions: Vec<Result<ChatSession>> =
            self.sessions.iter().map(|s| Ok(s.clone())).collect();
        Ok(Box::pin(stream::iter(sessions)))
    }
}

#[tokio::test]
async fn test_chat_loader_lazy_load() {
    use futures::StreamExt;
    let loader = TestChatLoader {
        sessions: vec![
            ChatSession::new(vec![Message::Human(HumanMessage::new("a"))]),
            ChatSession::new(vec![Message::Human(HumanMessage::new("b"))]),
        ],
    };
    let mut stream = loader.lazy_load().await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.messages.len(), 1);
    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.messages.len(), 1);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_chat_loader_load() {
    let loader = TestChatLoader {
        sessions: vec![
            ChatSession::new(vec![Message::Human(HumanMessage::new("x"))]),
            ChatSession::new(vec![Message::Human(HumanMessage::new("y"))]),
            ChatSession::new(vec![Message::Human(HumanMessage::new("z"))]),
        ],
    };
    let all = loader.load().await.unwrap();
    assert_eq!(all.len(), 3);
}

#[tokio::test]
async fn test_chat_loader_empty() {
    let loader = TestChatLoader { sessions: vec![] };
    let all = loader.load().await.unwrap();
    assert!(all.is_empty());
}

// ============================================================
// Rate Limiters
// ============================================================

use rustchain_core::rate_limiters::{BaseRateLimiter, InMemoryRateLimiter};

#[test]
fn test_rate_limiter_non_blocking_initially_empty() {
    // Fresh limiter starts with 0 tokens, so first non-blocking acquire should fail.
    let limiter = InMemoryRateLimiter::new(1.0, 0.1, 1.0);
    let result = limiter.acquire(false).unwrap();
    assert!(!result);
}

#[test]
fn test_rate_limiter_non_blocking_after_wait() {
    let limiter = InMemoryRateLimiter::new(100.0, 0.01, 10.0);
    // Wait long enough for tokens to replenish.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let result = limiter.acquire(false).unwrap();
    assert!(result);
}

#[test]
fn test_rate_limiter_blocking() {
    let limiter = InMemoryRateLimiter::new(100.0, 0.01, 10.0);
    // Blocking acquire should eventually succeed.
    let result = limiter.acquire(true).unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_rate_limiter_async_blocking() {
    let limiter = InMemoryRateLimiter::new(100.0, 0.01, 10.0);
    let result = limiter.aacquire(true).await.unwrap();
    assert!(result);
}

#[tokio::test]
async fn test_rate_limiter_async_non_blocking() {
    let limiter = InMemoryRateLimiter::new(1.0, 0.1, 1.0);
    let result = limiter.aacquire(false).await.unwrap();
    assert!(!result);
}

#[test]
fn test_rate_limiter_default() {
    let limiter = InMemoryRateLimiter::default();
    // Default: 1 req/s, initially empty bucket.
    let result = limiter.acquire(false).unwrap();
    assert!(!result);
}

#[test]
fn test_rate_limiter_burst() {
    let limiter = InMemoryRateLimiter::new(1000.0, 0.001, 5.0);
    std::thread::sleep(std::time::Duration::from_millis(20));
    // Should be able to acquire multiple tokens after waiting.
    let mut count = 0;
    for _ in 0..10 {
        if limiter.acquire(false).unwrap() {
            count += 1;
        }
    }
    assert!(count >= 1 && count <= 5, "Got {} tokens", count);
}

// ============================================================
// Structured Query
// ============================================================

use rustchain_core::structured_query::{
    Comparator, Comparison, FilterDirective, Operation, Operator, StructuredQuery, Visitor,
};

#[test]
fn test_operator_as_str() {
    assert_eq!(Operator::And.as_str(), "and");
    assert_eq!(Operator::Or.as_str(), "or");
    assert_eq!(Operator::Not.as_str(), "not");
}

#[test]
fn test_comparator_as_str() {
    assert_eq!(Comparator::Eq.as_str(), "eq");
    assert_eq!(Comparator::Ne.as_str(), "ne");
    assert_eq!(Comparator::Gt.as_str(), "gt");
    assert_eq!(Comparator::Gte.as_str(), "gte");
    assert_eq!(Comparator::Lt.as_str(), "lt");
    assert_eq!(Comparator::Lte.as_str(), "lte");
    assert_eq!(Comparator::Contain.as_str(), "contain");
    assert_eq!(Comparator::Like.as_str(), "like");
    assert_eq!(Comparator::In.as_str(), "in");
    assert_eq!(Comparator::Nin.as_str(), "nin");
}

#[test]
fn test_comparison_new() {
    let cmp = Comparison::new(Comparator::Eq, "author", json!("Tolkien"));
    assert_eq!(cmp.comparator, Comparator::Eq);
    assert_eq!(cmp.attribute, "author");
    assert_eq!(cmp.value, json!("Tolkien"));
}

#[test]
fn test_operation_new() {
    let op = Operation::new(
        Operator::And,
        vec![
            FilterDirective::Comparison(Comparison::new(Comparator::Eq, "genre", json!("fantasy"))),
            FilterDirective::Comparison(Comparison::new(Comparator::Gt, "year", json!(2000))),
        ],
    );
    assert_eq!(op.operator, Operator::And);
    assert_eq!(op.arguments.len(), 2);
}

#[test]
fn test_structured_query_builder() {
    let q = StructuredQuery::new("books about elves")
        .with_filter(FilterDirective::Comparison(Comparison::new(
            Comparator::Eq,
            "genre",
            json!("fantasy"),
        )))
        .with_limit(10);
    assert_eq!(q.query, "books about elves");
    assert!(q.filter.is_some());
    assert_eq!(q.limit, Some(10));
}

#[test]
fn test_structured_query_serde_roundtrip() {
    let q = StructuredQuery::new("test query").with_limit(5);
    let json_str = serde_json::to_string(&q).unwrap();
    let deserialized: StructuredQuery = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.query, "test query");
    assert_eq!(deserialized.limit, Some(5));
}

struct TestVisitor;

impl Visitor for TestVisitor {
    fn allowed_comparators(&self) -> &[Comparator] {
        &[Comparator::Eq, Comparator::Gt, Comparator::Lt]
    }

    fn allowed_operators(&self) -> &[Operator] {
        &[Operator::And, Operator::Or]
    }

    fn visit_comparison(&self, comparison: &Comparison) -> Result<Value> {
        Ok(json!({
            comparison.attribute.clone(): {
                comparison.comparator.as_str(): comparison.value
            }
        }))
    }

    fn visit_operation(&self, operation: &Operation) -> Result<Value> {
        let mut results = Vec::new();
        for arg in &operation.arguments {
            results.push(arg.accept(self)?);
        }
        Ok(json!({
            format!("${}", operation.operator.as_str()): results
        }))
    }

    fn visit_structured_query(&self, query: &StructuredQuery) -> Result<(String, Value)> {
        let filter = match &query.filter {
            Some(f) => f.accept(self)?,
            None => Value::Null,
        };
        Ok((query.query.clone(), filter))
    }
}

#[test]
fn test_visitor_comparison() {
    let visitor = TestVisitor;
    let cmp = Comparison::new(Comparator::Eq, "color", json!("red"));
    let result = visitor.visit_comparison(&cmp).unwrap();
    assert_eq!(result, json!({"color": {"eq": "red"}}));
}

#[test]
fn test_visitor_operation() {
    let visitor = TestVisitor;
    let op = Operation::new(
        Operator::And,
        vec![
            FilterDirective::Comparison(Comparison::new(Comparator::Eq, "a", json!(1))),
            FilterDirective::Comparison(Comparison::new(Comparator::Gt, "b", json!(2))),
        ],
    );
    let result = visitor.visit_operation(&op).unwrap();
    assert_eq!(
        result,
        json!({"$and": [{"a": {"eq": 1}}, {"b": {"gt": 2}}]})
    );
}

#[test]
fn test_visitor_structured_query() {
    let visitor = TestVisitor;
    let q = StructuredQuery::new("find items").with_filter(FilterDirective::Comparison(
        Comparison::new(Comparator::Eq, "type", json!("book")),
    ));
    let (query_str, filter) = visitor.visit_structured_query(&q).unwrap();
    assert_eq!(query_str, "find items");
    assert_eq!(filter, json!({"type": {"eq": "book"}}));
}

#[test]
fn test_filter_directive_accept() {
    let visitor = TestVisitor;
    let directive =
        FilterDirective::Comparison(Comparison::new(Comparator::Lt, "price", json!(100)));
    let result = directive.accept(&visitor).unwrap();
    assert_eq!(result, json!({"price": {"lt": 100}}));
}

// ============================================================
// Indexing
// ============================================================

use rustchain_core::indexing::{InMemoryRecordManager, RecordManager};

#[tokio::test]
async fn test_record_manager_create_schema() {
    let rm = InMemoryRecordManager::new("test");
    assert!(rm.create_schema().await.is_ok());
}

#[tokio::test]
async fn test_record_manager_get_time() {
    let rm = InMemoryRecordManager::new("test");
    let t = rm.get_time().await.unwrap();
    assert!(t > 0.0);
}

#[tokio::test]
async fn test_record_manager_update_and_exists() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(
        &["key1".into(), "key2".into()],
        &[Some("group1".into()), None],
        None,
    )
    .await
    .unwrap();

    let exists = rm
        .exists(&["key1".into(), "key2".into(), "key3".into()])
        .await
        .unwrap();
    assert_eq!(exists, vec![true, true, false]);
}

#[tokio::test]
async fn test_record_manager_list_keys() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(
        &["a".into(), "b".into(), "c".into()],
        &[Some("g1".into()), Some("g1".into()), Some("g2".into())],
        None,
    )
    .await
    .unwrap();

    // All keys
    let all = rm.list_keys(None, None, None, None).await.unwrap();
    assert_eq!(all.len(), 3);

    // Filter by group
    let g1 = rm
        .list_keys(None, None, Some(&["g1".into()]), None)
        .await
        .unwrap();
    assert_eq!(g1.len(), 2);
    assert!(g1.contains(&"a".to_string()));
    assert!(g1.contains(&"b".to_string()));

    // With limit
    let limited = rm.list_keys(None, None, None, Some(2)).await.unwrap();
    assert_eq!(limited.len(), 2);
}

#[tokio::test]
async fn test_record_manager_delete_keys() {
    let rm = InMemoryRecordManager::new("test");
    rm.update(&["x".into(), "y".into()], &[None, None], None)
        .await
        .unwrap();

    rm.delete_keys(&["x".into()]).await.unwrap();

    let exists = rm.exists(&["x".into(), "y".into()]).await.unwrap();
    assert_eq!(exists, vec![false, true]);
}

#[tokio::test]
async fn test_record_manager_namespace() {
    let rm = InMemoryRecordManager::new("my_namespace");
    assert_eq!(rm.namespace(), "my_namespace");
}

#[tokio::test]
async fn test_record_manager_time_at_least() {
    let rm = InMemoryRecordManager::new("test");
    let future_time = 99999999999.0;
    rm.update(&["k".into()], &[None], Some(future_time))
        .await
        .unwrap();

    // Key should exist with the forced timestamp
    let keys = rm
        .list_keys(Some(future_time), None, None, None)
        .await
        .unwrap();
    assert!(keys.is_empty()); // updated_at == future_time, not < future_time

    let keys = rm
        .list_keys(Some(future_time + 1.0), None, None, None)
        .await
        .unwrap();
    assert_eq!(keys, vec!["k".to_string()]);
}

// ============================================================
// Tracers
// ============================================================

use rustchain_core::tracers::base::{BaseTracer, InMemoryTracer};
use rustchain_core::tracers::schemas::{Run, RunType};

#[test]
fn test_run_new() {
    let id = Uuid::new_v4();
    let run = Run::new(id, "test_run", RunType::Llm, json!({"prompt": "hi"}));
    assert_eq!(run.id, id);
    assert_eq!(run.name, "test_run");
    assert_eq!(run.run_type, RunType::Llm);
    assert!(run.outputs.is_none());
    assert!(run.error.is_none());
    assert!(run.parent_run_id.is_none());
    assert!(run.child_runs.is_empty());
    assert!(run.tags.is_empty());
}

#[test]
fn test_run_serde_roundtrip() {
    let id = Uuid::new_v4();
    let mut run = Run::new(id, "chain_run", RunType::Chain, json!({"input": "x"}));
    run.outputs = Some(json!({"output": "y"}));
    run.tags = vec!["tag1".into()];

    let json_str = serde_json::to_string(&run).unwrap();
    let deserialized: Run = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.id, id);
    assert_eq!(deserialized.name, "chain_run");
    assert_eq!(deserialized.outputs, Some(json!({"output": "y"})));
    assert_eq!(deserialized.tags, vec!["tag1"]);
}

#[test]
fn test_run_types() {
    let types = vec![
        RunType::Llm,
        RunType::ChatModel,
        RunType::Chain,
        RunType::Tool,
        RunType::Retriever,
    ];
    for rt in types {
        let run = Run::new(Uuid::new_v4(), "test", rt, Value::Null);
        let json_str = serde_json::to_string(&run).unwrap();
        let deserialized: Run = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.run_type, rt);
    }
}

#[test]
fn test_in_memory_tracer_start_end_trace() {
    let mut tracer = InMemoryTracer::new();
    let run_id = Uuid::new_v4();
    let run = Run::new(run_id, "test", RunType::Llm, json!("input"));

    tracer.start_trace(run);
    assert!(tracer.run_map().contains_key(&run_id));

    let ended = tracer.end_trace(run_id);
    assert!(ended.is_some());
    assert!(!tracer.run_map().contains_key(&run_id));
}

#[test]
fn test_in_memory_tracer_default() {
    let tracer = InMemoryTracer::default();
    assert!(tracer.runs.is_empty());
    assert!(tracer.run_map().is_empty());
}

#[tokio::test]
async fn test_in_memory_tracer_persist_run() {
    let tracer = InMemoryTracer::new();
    let run = Run::new(Uuid::new_v4(), "test", RunType::Tool, Value::Null);
    // persist_run is a no-op for InMemoryTracer but should not error.
    assert!(tracer.persist_run(&run).await.is_ok());
}

#[test]
fn test_tracer_create_llm_run() {
    let run_id = Uuid::new_v4();
    let parent_id = Uuid::new_v4();
    let run =
        InMemoryTracer::create_llm_run(run_id, "llm", json!({"prompts": ["hi"]}), Some(parent_id));
    assert_eq!(run.run_type, RunType::Llm);
    assert_eq!(run.parent_run_id, Some(parent_id));
}

#[test]
fn test_tracer_create_chain_run() {
    let run = InMemoryTracer::create_chain_run(Uuid::new_v4(), "chain", json!({}), None);
    assert_eq!(run.run_type, RunType::Chain);
    assert!(run.parent_run_id.is_none());
}

#[test]
fn test_tracer_create_tool_run() {
    let run = InMemoryTracer::create_tool_run(Uuid::new_v4(), "search", "query text", None);
    assert_eq!(run.run_type, RunType::Tool);
    assert_eq!(run.inputs, json!("query text"));
}

#[test]
fn test_tracer_create_retriever_run() {
    let run =
        InMemoryTracer::create_retriever_run(Uuid::new_v4(), "retriever", "search terms", None);
    assert_eq!(run.run_type, RunType::Retriever);
    assert_eq!(run.inputs, json!("search terms"));
}

// ============================================================
// Load / Serializable
// ============================================================

use rustchain_core::load::serializable::{
    dumpd, dumps, make_serialized_secret, to_json_not_implemented, Serializable,
};

#[allow(dead_code)]
struct TestSerializable {
    name: String,
    api_key: String,
}

impl Serializable for TestSerializable {
    fn is_lc_serializable(&self) -> bool {
        true
    }

    fn get_lc_namespace(&self) -> Vec<String> {
        vec!["rustchain".into(), "llms".into()]
    }

    fn lc_secrets(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("api_key".into(), "MY_API_KEY".into());
        m
    }

    fn lc_id(&self) -> Vec<String> {
        vec!["rustchain".into(), "llms".into(), "TestLLM".into()]
    }

    fn to_json(&self) -> Value {
        json!({
            "lc": 1,
            "type": "constructor",
            "id": self.lc_id(),
            "kwargs": {
                "name": self.name,
                "api_key": {
                    "lc": 1,
                    "type": "secret",
                    "id": ["MY_API_KEY"]
                }
            }
        })
    }
}

#[test]
fn test_serializable_is_lc_serializable() {
    let obj = TestSerializable {
        name: "test".into(),
        api_key: "secret".into(),
    };
    assert!(obj.is_lc_serializable());
}

#[test]
fn test_serializable_lc_id() {
    let obj = TestSerializable {
        name: "test".into(),
        api_key: "secret".into(),
    };
    assert_eq!(obj.lc_id(), vec!["rustchain", "llms", "TestLLM"]);
}

#[test]
fn test_serializable_lc_namespace() {
    let obj = TestSerializable {
        name: "test".into(),
        api_key: "secret".into(),
    };
    assert_eq!(obj.get_lc_namespace(), vec!["rustchain", "llms"]);
}

#[test]
fn test_serializable_secrets() {
    let obj = TestSerializable {
        name: "test".into(),
        api_key: "secret".into(),
    };
    let secrets = obj.lc_secrets();
    assert_eq!(secrets.get("api_key").unwrap(), "MY_API_KEY");
}

#[test]
fn test_dumpd() {
    let obj = TestSerializable {
        name: "my_llm".into(),
        api_key: "sk-123".into(),
    };
    let val = dumpd(&obj);
    assert_eq!(val["lc"], 1);
    assert_eq!(val["type"], "constructor");
    assert_eq!(val["kwargs"]["name"], "my_llm");
    // API key should be serialized as a secret marker, not the actual value.
    assert_eq!(val["kwargs"]["api_key"]["type"], "secret");
}

#[test]
fn test_dumps_compact() {
    let obj = TestSerializable {
        name: "x".into(),
        api_key: "k".into(),
    };
    let s = dumps(&obj, false);
    assert!(s.contains("\"constructor\""));
    assert!(!s.contains('\n'));
}

#[test]
fn test_dumps_pretty() {
    let obj = TestSerializable {
        name: "x".into(),
        api_key: "k".into(),
    };
    let s = dumps(&obj, true);
    assert!(s.contains('\n'));
}

#[test]
fn test_to_json_not_implemented() {
    let val = to_json_not_implemented("MyClass", "MyClass(x=1)", vec!["rustchain".into()]);
    assert_eq!(val["type"], "not_implemented");
    assert_eq!(val["id"], json!(["rustchain", "MyClass"]));
    assert_eq!(val["repr"], "MyClass(x=1)");
}

#[test]
fn test_make_serialized_secret() {
    let val = make_serialized_secret(vec!["OPENAI_API_KEY".into()]);
    assert_eq!(val["type"], "secret");
    assert_eq!(val["id"], json!(["OPENAI_API_KEY"]));
    assert_eq!(val["lc"], 1);
}

// ============================================================
// Document Loaders (tests for completeness)
// ============================================================

use rustchain_core::document_loaders::{BaseLoader, DocumentStream};
use rustchain_core::documents::Document;

struct TestDocLoader {
    docs: Vec<Document>,
}

#[async_trait]
impl BaseLoader for TestDocLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let docs: Vec<Result<Document>> = self.docs.iter().map(|d| Ok(d.clone())).collect();
        Ok(Box::pin(stream::iter(docs)))
    }
}

#[tokio::test]
async fn test_doc_loader_load() {
    let loader = TestDocLoader {
        docs: vec![Document::new("doc1"), Document::new("doc2")],
    };
    let loaded = loader.load().await.unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].page_content, "doc1");
}

#[tokio::test]
async fn test_doc_loader_lazy_load() {
    use futures::StreamExt;
    let loader = TestDocLoader {
        docs: vec![Document::new("single")],
    };
    let mut stream = loader.lazy_load().await.unwrap();
    let doc = stream.next().await.unwrap().unwrap();
    assert_eq!(doc.page_content, "single");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_doc_loader_empty() {
    let loader = TestDocLoader { docs: vec![] };
    let loaded = loader.load().await.unwrap();
    assert!(loaded.is_empty());
}
