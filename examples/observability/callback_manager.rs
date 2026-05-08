//! HandlerBuilder + CallbackManager — typed callbacks routed through
//! a single observer entry on `RunnableConfig`.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::callbacks::{CallbackManager, HandlerBuilder, HandlerObserver};
use cognis_core::compose::lambda;
use cognis_core::Observer;

#[tokio::main]
async fn main() -> Result<()> {
    let make = || HandlerBuilder::new()
        .on_chain_start(|name, _, run| println!("[start] {name} run_id={run}"))
        .on_chain_end(|name, _, run| println!("[end]   {name} run_id={run}"))
        .build();
    let mgr = CallbackManager::new().push(Arc::new(make()));
    println!("handlers registered: {}", mgr.len());

    let observer: Arc<dyn Observer> = Arc::new(HandlerObserver(make()));
    let mut cfg = RunnableConfig::default();
    cfg.observers.push(observer);
    let chain = lambda(|x: i32| async move { Ok::<_, CognisError>(x + 1) });
    let out = chain.invoke(10, cfg).await?;
    println!("out: {out}");
    Ok(())
}
