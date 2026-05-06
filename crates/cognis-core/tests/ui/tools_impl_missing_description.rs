use cognis_macros::tools_impl;
use serde::Deserialize;

#[derive(Deserialize, schemars::JsonSchema)]
struct P { a: u32 }

struct Bad;

#[tools_impl]
impl Bad {
    #[tool]
    async fn run(&self, _p: P) -> Result<(), ()> { Ok(()) }
}

fn main() {}
