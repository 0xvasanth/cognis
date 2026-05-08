//! What you'll learn:
//!   How to call `health_check` on the underlying `LLMProvider` to
//!   get a `HealthStatus` — and surface a clear error to the user
//!   instead of failing on the first chat call.
//!
//! Why this matters:
//!   Before opening a session, verify the LLM provider is reachable.
//!   In production you don't want the first sign of a degraded
//!   provider to be a user-visible failure. Polling `health_check`
//!   from a sidecar — or just before each high-stakes call — gives
//!   you a uniform signal across every provider.
//!
//! Scenario:
//!   A pre-flight check at app startup. Before letting the user open
//!   a chat session, we verify the provider is reachable. If it
//!   isn't, we print a friendly message and exit non-zero so the
//!   supervisor can restart us — instead of letting the first user
//!   prompt fail.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example obs_health_monitoring
//!
//! Sample output (against ollama / llama3.1):
//!   [startup] checking provider "ollama" ...
//!   [startup] provider healthy: Healthy { latency_ms: 5 }
//!   [startup] safe to accept user sessions

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let provider = client.provider();

    println!("[startup] checking provider {:?} ...", provider.name());
    match provider.health_check().await {
        Ok(status) => {
            println!("[startup] provider healthy: {status:?}");
            println!("[startup] safe to accept user sessions");
        }
        Err(e) => {
            eprintln!("[startup] provider UNHEALTHY: {e}");
            eprintln!("[startup] not opening sessions — supervisor should restart");
            std::process::exit(1);
        }
    }
    Ok(())
}
