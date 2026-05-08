//! Cache wrapper — memoize a Runnable's outputs by a derived key.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::wrappers::{Cache, MemoryCache};

#[tokio::main]
async fn main() -> Result<()> {
    let calls = Arc::new(AtomicU32::new(0));
    let c = calls.clone();
    let inner = lambda(move |s: String| {
        let c = c.clone();
        async move {
            c.fetch_add(1, Ordering::Relaxed);
            Ok::<_, CognisError>(s.to_uppercase())
        }
    });
    let backend = Arc::new(MemoryCache::<String, String>::new());
    let cached = Cache::new(inner, backend, |s: &String| s.clone());

    let _ = cached.invoke("hello".into(), Default::default()).await?;
    let _ = cached.invoke("hello".into(), Default::default()).await?; // hit
    let _ = cached.invoke("world".into(), Default::default()).await?;
    println!("inner invocations: {}", calls.load(Ordering::Relaxed));
    Ok(())
}
