use cognis_macros::tools_impl;

struct Empty;

#[tools_impl]
impl Empty {
    fn helper(&self) -> u32 { 0 }
}

fn main() {}
