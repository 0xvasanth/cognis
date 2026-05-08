//! What you'll learn:
//!   How a `Tool` can carry its own state — here a connection-pool
//!   handle and a default target language — so per-call args stay
//!   minimal and the LLM doesn't have to repeat configuration on
//!   every invocation.
//!
//! Why this matters:
//!   Real tools wrap services with config and resources: API keys,
//!   base URLs, default IDs, **HTTP connection pools**. Putting that
//!   state on the tool's receiver keeps the args schema (which the
//!   LLM sees) focused on what actually varies per call, and lets
//!   the underlying HTTP client reuse one TCP connection pool across
//!   every translation request.
//!
//! Scenario:
//!   A translator tool that converts user-typed messages into
//!   Spanish by default. The HTTP client is built once and shared
//!   across calls — the LLM only ever sees `text` (and an optional
//!   `target` override), not the URL, headers, or pool config.
//!
//! Run with:
//!   cargo run -p cognis-examples --example tools_stateful_http
//!
//! Sample output (against ollama / llama3.1):
//!   schema (what the LLM sees):
//!   {
//!     "properties": {
//!       "target": {
//!         "description": "Optional ISO-639 target code; defaults to the tool's configured default language.",
//!         "type": [
//!           "string",
//!           "null"
//!   ...
//!   override (fr)  -> {"original":"good morning","target":"fr","translated":"[fr via https://translate.example.invalid] (stubbed) good morning"}
//!
//!   total HTTP calls through the shared pool: 4

use async_trait::async_trait;
use cognis::prelude::*;
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::tools::{SchemaBasedTool, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TranslateArgs {
    /// Text to translate.
    text: String,
    /// Optional ISO-639 target code; defaults to the tool's
    /// configured default language.
    target: Option<String>,
}

/// Stand-in for a real HTTP client. In production this is a
/// `reqwest::Client` (or any pooled HTTP client) — reused across
/// every tool call so the TCP/TLS handshake doesn't repeat.
struct HttpPool {
    base_url: String,
    calls: std::sync::atomic::AtomicU64,
}

impl HttpPool {
    fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            calls: std::sync::atomic::AtomicU64::new(0),
        }
    }
    /// Stub `POST` returning a fake translated payload. Increments a
    /// counter so the demo can prove the pool was reused.
    async fn post(&self, _path: &str, body: &str, target: &str) -> String {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("[{target} via {}] (stubbed) {body}", self.base_url)
    }
}

/// Tool state: the shared HTTP pool + the default target. In
/// production this is where API keys, base URLs, and timeouts live.
struct Translator {
    /// One pool, reused across calls. The point of the example.
    http: HttpPool,
    default_target: String,
}

impl Translator {
    fn new(default_target: impl Into<String>) -> Self {
        Self {
            http: HttpPool::new("https://translate.example.invalid"),
            default_target: default_target.into(),
        }
    }
}

#[async_trait]
impl SchemaBasedTool for Translator {
    type Params = TranslateArgs;
    type Output = Value;
    fn name(&self) -> &str {
        "translate"
    }
    fn description(&self) -> &str {
        "Translate text into a target language."
    }
    async fn execute_typed(&self, args: TranslateArgs) -> Result<Value> {
        let target = args.target.unwrap_or_else(|| self.default_target.clone());

        // In real code: `self.http.post("/translate").json(&body).send().await?`.
        // The key point: `self.http` is shared across every call, so
        // the connection pool warms up once and stays warm.
        let translated = self.http.post("/translate", &args.text, &target).await;

        Ok(json!({
            "original": args.text,
            "target": target,
            "translated": translated,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let t = Translator::new("es");
    println!(
        "schema (what the LLM sees):\n{:#}\n",
        Tool::args_schema(&t).unwrap()
    );

    // Three calls — same `Translator`, same pooled HTTP client.
    for text in ["hello", "where is the train station", "thanks!"] {
        let out = t
            .execute_typed(TranslateArgs {
                text: text.into(),
                target: None,
            })
            .await?;
        println!("default-target -> {out}");
    }

    // One call with an override.
    let fr = t
        .execute_typed(TranslateArgs {
            text: "good morning".into(),
            target: Some("fr".into()),
        })
        .await?;
    println!("override (fr)  -> {fr}");

    let total = t.http.calls.load(std::sync::atomic::Ordering::Relaxed);
    println!("\ntotal HTTP calls through the shared pool: {total}");
    Ok(())
}
