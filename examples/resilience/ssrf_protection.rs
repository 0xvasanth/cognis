//! `is_public_unicast` — vet a URL's host before letting a tool fetch it.
//! Blocks loopback, link-local, multicast, and reserved ranges.

use cognis::prelude::*;
use cognis_core::is_public_unicast;
use std::net::{IpAddr, Ipv4Addr};

fn main() {
    for ip in [
        IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
        IpAddr::V4(Ipv4Addr::new(169, 254, 0, 1)),
    ] {
        println!("{ip}  → public-unicast? {}", is_public_unicast(&ip));
    }
    let _: Result<()> = Ok(());
}
