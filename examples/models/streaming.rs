//! Token-by-token streaming via Client::stream.

use cognis::prelude::*;
use cognis_llm::Client;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let client = Client::from_env()?;
    let mut s = client.stream(vec![Message::human("Tell me a one-line joke.")]).await?;
    while let Some(c) = s.next().await {
        let c = c?;
        print!("{}", c.content);
    }
    println!();
    Ok(())
}
