//! Custom Runnable<I, O>: invoke + the default streaming behavior.

use async_trait::async_trait;
use cognis::prelude::*;
use futures::StreamExt;

struct Doubler;

#[async_trait]
impl Runnable<u32, u32> for Doubler {
    async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
        Ok(input * 2)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let r = Doubler;

    println!("invoke: {}", r.invoke(5, RunnableConfig::default()).await?);

    let v = r.batch(vec![1, 2, 3, 4], RunnableConfig::default()).await?;
    println!("batch: {:?}", v);

    let mut s = r.stream(7, RunnableConfig::default()).await?;
    while let Some(item) = s.next().await {
        println!("stream item: {}", item?);
    }
    Ok(())
}
