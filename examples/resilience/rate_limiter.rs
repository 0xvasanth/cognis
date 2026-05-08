//! Token-bucket rate limiter from cognis::middleware. Caps the request
//! rate to a Client/Provider pipeline.

use std::sync::Arc;

use cognis::middleware::{RateLimit, RateLimiter, TokenBucket};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    let bucket: Arc<dyn RateLimiter> = Arc::new(TokenBucket::new(2.0, 2));
    let _limiter = RateLimit::new(bucket);
    println!("RateLimit middleware constructed (rate=2/s, burst=2)");
    println!("(wire it into a Client via .with_middleware(...) in a real app)");
    Ok(())
}
