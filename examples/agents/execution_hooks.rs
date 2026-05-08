//! Per-call lifecycle hooks via cognis_core::wrappers::WithListeners.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::wrappers::{ListenerBuilder, WithListeners};

#[tokio::main]
async fn main() -> Result<()> {
    let inner = lambda(|x: u32| async move { Ok::<_, cognis_core::CognisError>(x * 3) });
    let listener = ListenerBuilder::new()
        .on_start(|inp, _cfg| println!("[start] input={inp}"))
        .on_end(|inp, out, _cfg| println!("[end]   {inp} -> {out}"))
        .with_name("triple-listener")
        .build();
    let chain = WithListeners::new(inner).push(Arc::new(listener));
    let _ = chain.invoke(7, Default::default()).await?;
    Ok(())
}
