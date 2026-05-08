//! Conditional routing via `Branch`. V2's replacement for V1's
//! ConditionalChain / BranchChain / SwitchChain.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::{lambda, Branch};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Conditional Routing ===\n");

    // Branches by string content.
    let math_branch = Arc::new(lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(format!("MATH: {s}"))
    })) as Arc<dyn Runnable<String, String>>;
    let chat_branch = Arc::new(lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(format!("CHAT: {s}"))
    })) as Arc<dyn Runnable<String, String>>;
    let default_branch = Arc::new(lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(format!("OTHER: {s}"))
    })) as Arc<dyn Runnable<String, String>>;

    let router = Branch::new(default_branch)
        .case(|s: &String| s.contains("calculate"), math_branch)
        .case(|s: &String| s.starts_with("hello"), chat_branch);

    for q in ["calculate 2+2", "hello there", "weather today"] {
        let out = router.invoke(q.to_string(), Default::default()).await?;
        println!("{q:>20} -> {out}");
    }
    Ok(())
}
