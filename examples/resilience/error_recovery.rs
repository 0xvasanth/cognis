//! Retry — wrap any Runnable with bounded retries + exponential backoff.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::wrappers::{Retry, RetryPolicy};

#[tokio::main]
async fn main() -> Result<()> {
    let n = Arc::new(AtomicU32::new(0));
    let n2 = n.clone();
    let flaky = lambda(move |_: ()| {
        let n = n2.clone();
        async move {
            let attempt = n.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt < 3 {
                Err(CognisError::Network {
                    status_code: Some(503),
                    message: format!("transient {attempt}"),
                })
            } else {
                Ok::<_, CognisError>(format!("ok at attempt {attempt}"))
            }
        }
    });

    let retried = Retry::new(
        flaky,
        RetryPolicy::new(5).with_initial_delay(Duration::from_millis(5)).with_backoff(2.0),
    );
    let out = retried.invoke((), Default::default()).await?;
    println!("{out} (total attempts: {})", n.load(Ordering::Relaxed));
    Ok(())
}
