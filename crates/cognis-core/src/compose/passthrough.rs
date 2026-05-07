//! Identity runnable.

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Identity runnable: outputs its input unchanged. Generic over `I`,
/// implementing `Runnable<I, I>` for any `I: Send + 'static`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Passthrough;

impl Passthrough {
    /// Construct a `Passthrough`.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<I> Runnable<I, I> for Passthrough
where
    I: Send + 'static,
{
    async fn invoke(&self, input: I, _: RunnableConfig) -> Result<I> {
        Ok(input)
    }
    fn name(&self) -> &str {
        "Passthrough"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn passes_through() {
        let p = Passthrough::new();
        let v: u32 = p.invoke(42, RunnableConfig::default()).await.unwrap();
        assert_eq!(v, 42);
    }
}
