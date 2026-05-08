use cognis_macros::tools_impl;

struct Bad;

#[tools_impl]
impl Bad {
    #[tool(description = "two args is too many in slice 1")]
    async fn run(&self, _a: u32, _b: u32) -> Result<(), ()> { Ok(()) }
}

fn main() {}
