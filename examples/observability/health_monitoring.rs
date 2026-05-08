//! LLMProvider::health_check — every provider reports a HealthStatus.

use cognis_llm::Client;

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let client = Client::from_env()?;
    let status = client.provider().health_check().await?;
    println!("provider: {}", client.provider().name());
    println!("health: {status:?}");
    Ok(())
}
