//! SSRF Protection Example
//!
//! Demonstrates the `cognis_core::security::ssrf` module for validating URLs
//! against Server-Side Request Forgery attacks. Shows default validation,
//! private IP blocking, cloud metadata blocking, custom configuration,
//! and integration with a chat model.

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::ChatModelRunnable;
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;
use cognis_core::security::ssrf::{CidrRange, SsrfValidator};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SSRF Protection Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Default validator — safe public URLs
    // -----------------------------------------------------------------------
    println!("--- 1. Validating safe public URLs ---\n");

    let validator = SsrfValidator::default();

    let safe_urls = [
        "https://api.example.com/webhook",
        "https://hooks.slack.com/services/T00/B00/xxx",
        "http://example.org/callback",
        "https://8.8.8.8/dns-query",
    ];

    for url in &safe_urls {
        match validator.validate_url(url) {
            Ok(validated) => println!("  SAFE: {}", validated.as_str()),
            Err(e) => println!("  BLOCKED: {} -> {}", url, e),
        }
    }

    // -----------------------------------------------------------------------
    // 2. Blocking private IP addresses
    // -----------------------------------------------------------------------
    println!("\n--- 2. Blocking private IP addresses ---\n");

    let private_urls = [
        "http://10.0.0.1/admin",
        "http://172.16.0.1/internal",
        "http://192.168.1.1/router",
        "http://127.0.0.1:8080/debug",
        "http://localhost:3000/api",
    ];

    for url in &private_urls {
        match validator.validate_url(url) {
            Ok(validated) => println!("  SAFE: {}", validated.as_str()),
            Err(e) => println!("  BLOCKED: {} -> {}", url, e),
        }
    }

    // -----------------------------------------------------------------------
    // 3. Blocking cloud metadata endpoints
    // -----------------------------------------------------------------------
    println!("\n--- 3. Blocking cloud metadata endpoints ---\n");

    let metadata_urls = [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.170.2/v2/metadata",
        "http://100.100.100.200/latest",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest",
    ];

    for url in &metadata_urls {
        match validator.validate_url(url) {
            Ok(validated) => println!("  SAFE: {}", validated.as_str()),
            Err(e) => println!("  BLOCKED: {} -> {}", url, e),
        }
    }

    // -----------------------------------------------------------------------
    // 4. Using the convenience method is_safe_url
    // -----------------------------------------------------------------------
    println!("\n--- 4. Convenience method: is_safe_url ---\n");

    println!(
        "  https://example.com  -> safe={}",
        validator.is_safe_url("https://example.com")
    );
    println!(
        "  http://10.0.0.1      -> safe={}",
        validator.is_safe_url("http://10.0.0.1")
    );
    println!(
        "  ftp://example.com    -> safe={}",
        validator.is_safe_url("ftp://example.com")
    );

    // -----------------------------------------------------------------------
    // 5. Custom validator with allowlists
    // -----------------------------------------------------------------------
    println!("\n--- 5. Custom validator with allowlists ---\n");

    let custom_validator = SsrfValidator::builder()
        .allow_domain("internal.mycompany.com")
        .allow_ip("10.0.0.5".parse().unwrap())
        .allow_http(false) // HTTPS only
        .block_range(CidrRange::parse("203.0.113.0/24").unwrap())
        .block_cloud_metadata_hostname("custom-metadata.internal")
        .build();

    let custom_tests = [
        ("http://internal.mycompany.com/api", "allowed domain, but HTTP"),
        ("https://internal.mycompany.com/api", "allowed domain + HTTPS"),
        ("https://10.0.0.5/hook", "allowed IP"),
        ("http://10.0.0.6/hook", "private IP, not allowed, HTTP"),
        ("http://example.com", "public but HTTP-only blocked"),
        ("https://203.0.113.50/api", "custom blocked range"),
        (
            "http://custom-metadata.internal/v1",
            "custom metadata hostname",
        ),
    ];

    for (url, description) in &custom_tests {
        let result = if custom_validator.is_safe_url(url) {
            "SAFE"
        } else {
            "BLOCKED"
        };
        println!("  {}: {} ({})", result, url, description);
    }

    // -----------------------------------------------------------------------
    // 6. Cloud metadata always blocked even with allow_private
    // -----------------------------------------------------------------------
    println!("\n--- 6. Cloud metadata always blocked (even with allow_private) ---\n");

    let permissive = SsrfValidator::builder().allow_private(true).build();

    println!(
        "  Private 10.0.0.1            -> safe={}",
        permissive.is_safe_url("http://10.0.0.1")
    );
    println!(
        "  Localhost 127.0.0.1         -> safe={}",
        permissive.is_safe_url("http://127.0.0.1:8080")
    );
    println!(
        "  AWS metadata 169.254.169.254 -> safe={}",
        permissive.is_safe_url("http://169.254.169.254/latest/meta-data/")
    );
    println!(
        "  GCP metadata hostname        -> safe={}",
        permissive.is_safe_url("http://metadata.google.internal/foo")
    );

    // -----------------------------------------------------------------------
    // 7. LLM-suggested URL validation
    // -----------------------------------------------------------------------
    println!("\n--- 7. Validating a URL suggested by the LLM ---\n");

    let model = shared::get_chat_model(vec![
        "https://api.github.com/repos/rust-lang/rust".to_string(),
    ]);

    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a helpful assistant. When asked for a URL, respond with only the URL and nothing else."),
        ("human", "Give me a URL to check the Rust programming language repository on GitHub."),
    ])?;

    let parser = StrOutputParser;
    let model_runnable = ChatModelRunnable::new(model);
    let chain = cognis_core::chain!(prompt, model_runnable, parser)?;

    let result = chain.invoke(json!({}), None).await?;
    let suggested_url = result.as_str().unwrap_or("").trim();
    println!("  LLM suggested: {}", suggested_url);

    match validator.validate_url(suggested_url) {
        Ok(validated) => println!("  Validation: SAFE - {}", validated.as_str()),
        Err(e) => println!("  Validation: BLOCKED - {}", e),
    }

    println!("\n=== SSRF Protection Example Complete ===");
    Ok(())
}
