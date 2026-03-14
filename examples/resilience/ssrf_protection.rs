//! SSRF Protection Example
//!
//! Validates URLs against SSRF rules, showing which pass and which are blocked.

#[path = "../shared.rs"]
mod shared;

use cognis_core::security::ssrf::{CidrRange, SsrfValidator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SSRF Protection ===\n");

    // Build a custom validator: HTTPS-only, one allowed internal domain,
    // and an extra blocked CIDR range.
    let validator = SsrfValidator::builder()
        .allow_domain("internal.mycompany.com")
        .allow_http(false)
        .block_range(CidrRange::parse("203.0.113.0/24").unwrap())
        .build();

    // URLs to validate — a mix of safe, private, metadata, and edge cases.
    let urls: &[(&str, &str)] = &[
        ("https://api.example.com/webhook", "public HTTPS"),
        (
            "https://hooks.slack.com/services/T00/B00/xxx",
            "public HTTPS",
        ),
        ("http://api.example.com/webhook", "public but HTTP"),
        (
            "https://internal.mycompany.com/api",
            "allowed internal domain",
        ),
        (
            "http://internal.mycompany.com/api",
            "allowed domain, but HTTP",
        ),
        ("http://10.0.0.1/admin", "private IP"),
        ("http://192.168.1.1/router", "private IP"),
        ("http://127.0.0.1:8080/debug", "loopback"),
        ("http://localhost:3000/api", "localhost"),
        ("http://169.254.169.254/latest/meta-data/", "AWS metadata"),
        (
            "http://metadata.google.internal/computeMetadata/v1/",
            "GCP metadata",
        ),
        ("https://203.0.113.50/api", "custom blocked CIDR"),
    ];

    for (url, label) in urls {
        let status = if validator.is_safe_url(url) {
            "PASS"
        } else {
            "BLOCK"
        };
        println!("  [{status}] {url}  ({label})");
    }

    println!("\n=== Done ===");
    Ok(())
}
