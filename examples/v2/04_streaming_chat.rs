//! Token-by-token streaming via Client. Requires COGNIS_API_KEY (or
//! COGNIS_OPENAI_API_KEY) for OpenAI; or COGNIS_PROVIDER=ollama with a
//! running Ollama server.

use cognis2::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let provider_set = std::env::var("COGNIS_PROVIDER").is_ok();
    if !provider_set {
        println!(
            "skipping: set COGNIS_PROVIDER=openai (with COGNIS_OPENAI_API_KEY) \
             or COGNIS_PROVIDER=ollama to run this example."
        );
        return Ok(());
    }

    let client = Client::from_env()?;
    let mut s = client
        .stream(vec![Message::human("Tell me a one-line joke.")])
        .await?;
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        print!("{}", chunk.content);
    }
    println!();
    Ok(())
}
