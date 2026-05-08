//! Wiremock integration tests for LangfuseExporter.

#![cfg(feature = "langfuse")]

use std::time::SystemTime;

use cognis_trace::{
    exporter::TraceExporter,
    span::{Generation, Span, SpanBuilder, SpanKind, TokenUsage},
    LangfuseConfig, LangfuseExporter,
};
use secrecy::SecretString;
use uuid::Uuid;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn cfg(host: String) -> LangfuseConfig {
    LangfuseConfig {
        host,
        public_key: "pk-test".into(),
        secret_key: SecretString::from("sk-test".to_string()),
        ..Default::default()
    }
}

fn root_span() -> Span {
    let id = Uuid::new_v4();
    SpanBuilder::open(
        id,
        None,
        id,
        SpanKind::Chain,
        "root",
        None,
        SystemTime::now(),
    )
    .finish_ok(None, SystemTime::now())
}

fn child_generation(parent: Uuid, trace: Uuid) -> Span {
    let id = Uuid::new_v4();
    let mut b = SpanBuilder::open(
        id,
        Some(parent),
        trace,
        SpanKind::Generation,
        "openai.gpt-4o",
        None,
        SystemTime::now(),
    )
    .with_generation(Generation {
        model: "gpt-4o-2024-08-06".into(),
        provider: "openai".into(),
        usage: TokenUsage {
            input: 100,
            output: 50,
            cache_read: 0,
            cache_write: 0,
        },
        ..Default::default()
    });
    b.span.metadata.clear();
    b.finish_ok(None, SystemTime::now())
}

#[tokio::test]
async fn export_posts_ingestion_with_basic_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .and(header("Authorization", "Basic cGstdGVzdDpzay10ZXN0"))
        .respond_with(ResponseTemplate::new(207).set_body_json(serde_json::json!({
            "successes": [],
            "errors": []
        })))
        .expect(1)
        .mount(&server)
        .await;

    let exp = LangfuseExporter::new(cfg(server.uri())).unwrap();
    let root = root_span();
    let child = child_generation(root.run_id, root.trace_id);
    exp.export_spans(vec![root, child]).await.unwrap();
}

#[tokio::test]
async fn root_span_emits_trace_create_in_addition_to_span_or_generation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = req.body_json().unwrap();
            let batch = body["batch"].as_array().expect("batch array");
            let types: Vec<&str> = batch.iter().map(|e| e["type"].as_str().unwrap()).collect();
            assert!(
                types.contains(&"trace-create"),
                "missing trace-create in {types:?}"
            );
            assert!(
                types.contains(&"span-create") || types.contains(&"generation-create"),
                "missing observation in {types:?}"
            );
            ResponseTemplate::new(207)
                .set_body_json(serde_json::json!({"successes":[],"errors":[]}))
        })
        .expect(1)
        .mount(&server)
        .await;

    let exp = LangfuseExporter::new(cfg(server.uri())).unwrap();
    let root = root_span();
    exp.export_spans(vec![root]).await.unwrap();
}

#[tokio::test]
async fn non_2xx_returns_backend_status_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    let exp = LangfuseExporter::new(cfg(server.uri())).unwrap();
    let root = root_span();
    let err = exp.export_spans(vec![root]).await.unwrap_err();
    match err {
        cognis_trace::TraceError::BackendStatus {
            backend, status, ..
        } => {
            assert_eq!(backend, "langfuse");
            assert_eq!(status, 401);
        }
        e => panic!("wrong error: {e:?}"),
    }
}
