//! Composition patterns: pipe, parallel, lambda. V2's idiomatic
//! replacement for V1's SequentialChain/ParallelChain/TransformChain.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::{lambda, Parallel};
use cognis_core::runnable_ext::RunnableExt;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Composition Patterns ===\n");

    // Sequential: pipe two lambdas. `.pipe()` is V2's LCEL `|`.
    let upper =
        lambda(|s: String| async move { Ok::<_, cognis_core::CognisError>(s.to_uppercase()) });
    let exclaim =
        lambda(|s: String| async move { Ok::<_, cognis_core::CognisError>(format!("{s}!")) });
    let seq = upper.pipe(exclaim);
    println!(
        "sequential: {}",
        seq.invoke("hello".into(), Default::default()).await?
    );

    // Parallel: fan input out to named branches.
    let len = Arc::new(lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(s.len())
    })) as Arc<dyn Runnable<String, usize>>;
    let words = Arc::new(lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(s.split_whitespace().count())
    })) as Arc<dyn Runnable<String, usize>>;
    let par = Parallel::<String, usize>::new()
        .branch("len", len)
        .branch("words", words);
    let out = par
        .invoke("hello there world".into(), Default::default())
        .await?;
    println!("parallel: {:?}", out);

    Ok(())
}
