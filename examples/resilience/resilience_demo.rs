//! Stack: Retry → Timeout → inner. Errors retried, but a single attempt
//! that exceeds the deadline aborts.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::wrappers::{Retry, RetryPolicy, Timeout};

#[tokio::main]
async fn main() -> Result<()> {
    let n = Arc::new(AtomicU32::new(0));
    let n2 = n.clone();
    let inner = lambda(move |_: ()| {
        let n = n2.clone();
        async move {
            let attempt = n.fetch_add(1, Ordering::Relaxed) + 1;
            if attempt < 3 {
                Err(CognisError::Network {
                    status_code: Some(503),
                    message: "transient".into(),
                })
            } else {
                Ok::<_, CognisError>("done")
            }
        }
    });
    let with_timeout = Timeout::new(inner, Duration::from_millis(50));
    let stack = Retry::new(
        with_timeout,
        RetryPolicy::new(5).with_initial_delay(Duration::from_millis(2)),
    );
    let out = stack.invoke((), Default::default()).await?;
    println!("{out} (attempts: {})", n.load(Ordering::Relaxed));
    Ok(())
}
