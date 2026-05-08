//! Observer that prints every Event — minimal "tracing"-style demo.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::Observer;

struct Printer;
impl Observer for Printer {
    fn on_event(&self, e: &Event) {
        match e {
            Event::OnStart { runnable, run_id, .. } => println!("[start] {runnable} ({run_id})"),
            Event::OnEnd { runnable, run_id, .. }   => println!("[end]   {runnable} ({run_id})"),
            Event::OnError { error, run_id }       => println!("[err]   ({run_id}): {error}"),
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cfg = RunnableConfig::default();
    cfg.observers.push(Arc::new(Printer));
    let chain = lambda(|n: i32| async move { Ok::<_, CognisError>(n * 2) });
    let out = chain.invoke(7, cfg).await?;
    println!("result: {out}");
    Ok(())
}
