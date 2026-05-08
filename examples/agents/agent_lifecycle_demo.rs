//! Lifecycle observation via the cognis_core observer plumbing. The
//! Agent emits OnStart/OnEnd/OnError events on its RunnableConfig
//! observers — same surface as any other Runnable.

use std::sync::Arc;

use cognis::prelude::*;
use cognis::AgentBuilder;
use cognis_core::stream::Observer;
use cognis_llm::Client;

struct LogObserver;
impl Observer for LogObserver {
    fn on_event(&self, e: &Event) {
        match e {
            Event::OnStart { runnable, .. } => println!("[start] {runnable}"),
            Event::OnNodeStart { node, step, .. } => println!("[node-start] step={step} {node}"),
            Event::OnEnd { runnable, .. } => println!("[end] {runnable}"),
            Event::OnError { error, .. } => println!("[error] {error}"),
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .build()?;

    // Observers attach via RunnableConfig, not AgentBuilder. We surface
    // them by streaming events instead.
    use futures::StreamExt;
    let mut s = agent.stream(Message::human("Say hello.")).await?;
    let obs = Arc::new(LogObserver);
    while let Some(ev) = s.next().await {
        
        obs.on_event(&ev);
    }
    Ok(())
}
