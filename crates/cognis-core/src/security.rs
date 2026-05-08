//! Security utilities — primitives intentionally narrow.
//!
//! The shipped primitive is [`is_public_unicast`] for SSRF defense:
//! given an `IpAddr`, it returns `true` only for globally-routable
//! unicast addresses, refusing loopback, private, link-local, multicast,
//! CGNAT, documentation, and IPv6-mapped non-public ranges.
//!
//! Pair with your own URL-parsing for HTTP-shaped guards. The
//! `cognis::tools::HttpRequest` tool uses these checks under the hood;
//! they live here for reuse by other tools and user code.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// True iff `ip` is a globally-routable unicast address.
///
/// Refuses: loopback, unspecified, multicast, broadcast, private,
/// link-local, documentation, CGNAT (100.64/10), benchmarking
/// (198.18/15), 0.0.0.0/8, IPv6 ULA (fc00::/7), IPv6 link-local
/// (fe80::/10), IPv6 documentation (2001:db8::/32), IPv4-mapped IPv6
/// addresses pointing at non-public IPv4.
///
/// Mirrors the unstable `IpAddr::is_global` semantics for the cases
/// that matter for SSRF defense.
pub fn is_public_unicast(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

fn is_public_v4(v4: &Ipv4Addr) -> bool {
    let o = v4.octets();
    !(v4.is_loopback()
        || v4.is_private()
        || v4.is_unspecified()
        || v4.is_link_local()
        || v4.is_multicast()
        || v4.is_broadcast()
        || v4.is_documentation()
        // Carrier-grade NAT 100.64.0.0/10
        || (o[0] == 100 && (o[1] & 0xc0) == 0x40)
        // Benchmarking 198.18.0.0/15
        || (o[0] == 198 && (o[1] & 0xfe) == 18)
        // 0.0.0.0/8
        || o[0] == 0)
}

fn is_public_v6(v6: &Ipv6Addr) -> bool {
    if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
        return false;
    }
    // ULA fc00::/7
    if (v6.segments()[0] & 0xfe00) == 0xfc00 {
        return false;
    }
    // Link-local fe80::/10
    if (v6.segments()[0] & 0xffc0) == 0xfe80 {
        return false;
    }
    // IPv4-mapped — re-check the embedded IPv4.
    if let Some(v4) = matches_v4_mapped(v6) {
        if !is_public_v4(&v4) {
            return false;
        }
    }
    // Documentation 2001:db8::/32
    if v6.segments()[0] == 0x2001 && v6.segments()[1] == 0xdb8 {
        return false;
    }
    true
}

fn matches_v4_mapped(v6: &Ipv6Addr) -> Option<Ipv4Addr> {
    let s = v6.segments();
    if s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0xffff {
        Some(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            (s[6] & 0xff) as u8,
            (s[7] >> 8) as u8,
            (s[7] & 0xff) as u8,
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn public_v4_passes() {
        assert!(is_public_unicast(&ip("1.1.1.1")));
        assert!(is_public_unicast(&ip("8.8.8.8")));
    }

    #[test]
    fn private_v4_refused() {
        for s in [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "100.64.0.1",  // CGNAT
            "198.18.0.1",  // benchmarking
            "0.0.0.1",     // 0.0.0.0/8
            "169.254.0.1", // link-local
            "224.0.0.1",   // multicast
            "203.0.113.0", // documentation
        ] {
            assert!(!is_public_unicast(&ip(s)), "expected refused: {s}");
        }
    }

    #[test]
    fn public_v6_passes() {
        assert!(is_public_unicast(&ip("2606:4700:4700::1111")));
    }

    #[test]
    fn private_v6_refused() {
        for s in ["::1", "fc00::1", "fe80::1", "2001:db8::1"] {
            assert!(!is_public_unicast(&ip(s)), "expected refused: {s}");
        }
    }

    #[test]
    fn ipv4_mapped_v6_inherits_v4_decision() {
        let mapped_loopback: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert!(!is_public_unicast(&mapped_loopback));
        let mapped_public: IpAddr = "::ffff:1.1.1.1".parse().unwrap();
        assert!(is_public_unicast(&mapped_public));
    }
}
