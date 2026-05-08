//! Plain Observer — capture every Event the Runnable emits.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::Observer;

struct Count(AtomicUsize);
impl Observer for Count {
    fn on_event(&self, _: &Event) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let counter = Arc::new(Count(AtomicUsize::new(0)));
    let mut cfg = RunnableConfig::default();
    cfg.observers.push(counter.clone());

    let chain = lambda(|x: i32| async move { Ok::<_, CognisError>(x * 2) });
    let out = chain.invoke(21, cfg).await?;
    println!("output: {out}");
    println!("events seen: {}", counter.0.load(Ordering::Relaxed));
    Ok(())
}
