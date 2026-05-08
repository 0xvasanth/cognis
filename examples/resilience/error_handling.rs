//! Plain error propagation through a Runnable chain.

use cognis::prelude::*;
use cognis_core::compose::lambda;

#[tokio::main]
async fn main() -> Result<()> {
    let parse = lambda(|s: String| async move {
        s.trim()
            .parse::<i32>()
            .map_err(|e| CognisError::Internal(format!("bad number: {e}")))
    });
    match parse.invoke("42".into(), Default::default()).await {
        Ok(n) => println!("parsed: {n}"),
        Err(e) => println!("err: {e}"),
    }
    match parse
        .invoke("not a number".into(), Default::default())
        .await
    {
        Ok(n) => println!("parsed: {n}"),
        Err(e) => println!("err: {e}"),
    }
    Ok(())
}
