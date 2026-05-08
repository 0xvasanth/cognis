//! Pipe operator + schema introspection. V2's `RunnableExt::pipe` is
//! the `|` of LCEL; `WithSchema` opts a runnable into JSON-schema
//! introspection so generated docs / API serving knows the I/O shapes.

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::runnable_ext::RunnableExt;
use cognis_core::schemars::{self, JsonSchema};
use cognis_core::wrappers::WithSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Q {
    topic: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct A {
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Pipe + Schema Introspection ===\n");

    // Two-stage pipeline: extract topic → answer it.
    let stage1 = lambda(|q: Q| async move {
        Ok::<_, cognis_core::CognisError>(format!("question about {}", q.topic))
    });
    let stage2 = lambda(|s: String| async move {
        Ok::<_, cognis_core::CognisError>(A {
            answer: format!("Here's a fact about: {s}"),
        })
    });

    let chain = stage1.pipe(stage2);
    let wrapped: WithSchema<_, Q, A> = WithSchema::new(chain);

    println!("input schema:  {}", wrapped.input_schema().unwrap());
    println!("output schema: {}", wrapped.output_schema().unwrap());

    let out = wrapped
        .invoke(
            Q {
                topic: "rust".into(),
            },
            Default::default(),
        )
        .await?;
    println!("\nresult: {}", out.answer);
    Ok(())
}
