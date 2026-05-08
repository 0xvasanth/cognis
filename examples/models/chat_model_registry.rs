//! V2 ProviderRegistry — instantiate providers from "id:model" strings.
//! ProviderRegistry::with_builtins pre-loads providers that are
//! compiled in via Cargo features.

use cognis::prelude::*;
use cognis_llm::{ProviderRegistry, ProviderSpec};

#[tokio::main]
async fn main() -> Result<()> {
    let reg = ProviderRegistry::with_builtins();
    println!("registered ids: {:?}", reg.ids());

    // Resolve a provider by id (no API call, just construction).
    if reg.ids().iter().any(|id| id == "ollama") {
        let provider = reg.build("ollama:llama3.2:1b", ProviderSpec::default())?;
        println!(
            "built provider: {} ({:?})",
            provider.name(),
            provider.provider_type()
        );
    }
    Ok(())
}
