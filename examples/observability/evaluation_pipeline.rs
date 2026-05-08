//! Simple eval loop — score a Runnable's outputs against expected values.

use cognis::prelude::*;
use cognis_core::compose::lambda;

#[tokio::main]
async fn main() -> Result<()> {
    let upper = lambda(|s: String| async move { Ok::<_, CognisError>(s.to_uppercase()) });
    let cases = [
        ("hello", "HELLO"),
        ("World", "WORLD"),
        ("Cognis", "cognis"),
    ];
    let mut pass = 0;
    for (i, (input, expected)) in cases.iter().enumerate() {
        let got = upper.invoke(input.to_string(), Default::default()).await?;
        let ok = got == *expected;
        pass += ok as usize;
        println!("{i}: {input:?} → {got:?}  expected {expected:?}  {}", if ok { "OK" } else { "FAIL" });
    }
    println!("passed {pass}/{}", cases.len());
    Ok(())
}
