//! Cooperative cancellation via RunnableConfig::cancel_token.

use std::time::Duration;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    let token = CancellationToken::new();
    let mut cfg = RunnableConfig::default();
    cfg.cancel_token = Some(token.clone());

    let slow = lambda(|x: u32| async move {
        for i in 0..50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if i == 5 {
                println!("..tick {i}");
            }
        }
        Ok::<_, cognis_core::CognisError>(x * 2)
    });

    let handle = tokio::spawn(async move { slow.invoke(7, cfg).await });

    tokio::time::sleep(Duration::from_millis(200)).await;
    println!("cancelling...");
    token.cancel();

    match handle.await.unwrap() {
        Ok(v) => println!("finished naturally: {v}"),
        Err(e) => println!("cancelled / errored: {e}"),
    }
    Ok(())
}
