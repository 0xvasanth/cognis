//! Same as planning_demo but explicit about composing multiple
//! middlewares (Planning + ModelRetry).

use cognis::prelude::*;
use cognis::{MiddlewarePipeline, ModelRetry, Planning};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let raw = Client::from_env()?;
    let pipe = MiddlewarePipeline::new()
        .push(ModelRetry::new(2))
        .push(Planning::new())
        .build(raw);

    let resp = pipe
        .invoke(
            vec![Message::human("Outline how to bake bread.")],
            Vec::new(),
            Default::default(),
        )
        .await?;
    println!("{}", resp.message.content());
    Ok(())
}
