//! What you'll learn:
//!   How `is_public_unicast` rejects loopback, private, link-local,
//!   and reserved IP ranges so a tool that fetches a user-supplied
//!   URL can't be tricked into hitting an internal endpoint.
//!
//! Why this matters:
//!   The moment you give an agent a `fetch_url` tool, you've opened
//!   the door to SSRF. A user (or a prompt-injection attacker) types
//!   `http://10.0.0.1/admin` and your service obligingly fetches it
//!   from inside your VPC. Vetting the resolved IP against
//!   `is_public_unicast` before opening the connection is the
//!   minimum bar for any URL-following tool.
//!
//! Scenario:
//!   The agent has a tool that takes a URL from user input and
//!   fetches it. Before opening the connection, it resolves the
//!   host and checks every candidate IP. We exercise the gate with
//!   a public DNS server, localhost, a private LAN address, and a
//!   link-local — only the public DNS passes.
//!
//! Run with:
//!   cargo run -p cognis-examples --example resilience_ssrf_protection
//!
//! Sample output (against ollama / llama3.1):
//!   ALLOW  8.8.8.8 (Google DNS)                   8.8.8.8
//!   BLOCK  127.0.0.1 (localhost)                  127.0.0.1  -- rejected: not a public-unicast IP (SSRF guard)
//!   BLOCK  10.0.0.5 (private LAN)                 10.0.0.5  -- rejected: not a public-unicast IP (SSRF guard)
//!   BLOCK  192.168.1.1 (home router)              192.168.1.1  -- rejected: not a public-unicast IP (SSRF guard)
//!   BLOCK  169.254.169.254 (cloud metadata!)      169.254.169.254  -- rejected: not a public-unicast IP (SSRF guard)
//!   BLOCK  ::1 (IPv6 loopback)                    ::1  -- rejected: not a public-unicast IP (SSRF guard)

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use cognis::prelude::*;
use cognis_core::is_public_unicast;

/// What your `fetch_url` tool would call before opening a socket.
fn safe_to_fetch(ip: &IpAddr) -> std::result::Result<(), &'static str> {
    if is_public_unicast(ip) {
        Ok(())
    } else {
        Err("rejected: not a public-unicast IP (SSRF guard)")
    }
}

fn main() -> Result<()> {
    let candidates: &[(&str, IpAddr)] = &[
        ("8.8.8.8 (Google DNS)",      IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))),
        ("127.0.0.1 (localhost)",     IpAddr::V4(Ipv4Addr::LOCALHOST)),
        ("10.0.0.5 (private LAN)",    IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5))),
        ("192.168.1.1 (home router)", IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))),
        ("169.254.169.254 (cloud metadata!)", IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))),
        ("::1 (IPv6 loopback)",       IpAddr::V6(Ipv6Addr::LOCALHOST)),
    ];
    for (label, ip) in candidates {
        match safe_to_fetch(ip) {
            Ok(_)  => println!("ALLOW  {label:<38} {ip}"),
            Err(e) => println!("BLOCK  {label:<38} {ip}  -- {e}"),
        }
    }
    Ok(())
}
