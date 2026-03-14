//! SSRF (Server-Side Request Forgery) protection for validating URLs.
//!
//! This module provides utilities to validate user-provided URLs and prevent SSRF attacks
//! by blocking requests to:
//! - Private IP ranges (RFC 1918, loopback, link-local)
//! - Cloud metadata endpoints (AWS, GCP, Azure, etc.)
//! - Localhost addresses
//! - Invalid URL schemes
//!
//! # Example
//!
//! ```rust
//! use cognis_core::security::ssrf::SsrfValidator;
//!
//! let validator = SsrfValidator::default();
//! assert!(validator.validate_url("https://example.com/api").is_ok());
//! assert!(validator.validate_url("http://169.254.169.254/latest/meta-data/").is_err());
//! ```

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Specific SSRF violation types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsrfError {
    /// The URL scheme is not allowed (e.g. `ftp://`).
    InvalidScheme(String),
    /// The URL has no hostname.
    MissingHostname,
    /// The URL could not be parsed.
    InvalidUrl(String),
    /// The hostname or IP resolves to a private IP range.
    PrivateIp(String),
    /// The hostname or IP resolves to a localhost address.
    Localhost(String),
    /// The hostname or IP is a known cloud metadata endpoint.
    CloudMetadata(String),
    /// The hostname or IP falls within a blocked CIDR range.
    BlockedRange(String),
    /// The URL is blocked because HTTPS is required but HTTP was used.
    HttpNotAllowed,
}

impl fmt::Display for SsrfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScheme(s) => {
                write!(f, "Only HTTP/HTTPS URLs are allowed, got scheme: {s}")
            }
            Self::MissingHostname => write!(f, "URL must have a valid hostname"),
            Self::InvalidUrl(s) => write!(f, "Invalid URL: {s}"),
            Self::PrivateIp(ip) => write!(f, "URL resolves to private IP address: {ip}"),
            Self::Localhost(host) => write!(f, "Localhost URLs are not allowed: {host}"),
            Self::CloudMetadata(host) => {
                write!(f, "Cloud metadata endpoints are not allowed: {host}")
            }
            Self::BlockedRange(ip) => write!(f, "URL resolves to blocked IP range: {ip}"),
            Self::HttpNotAllowed => write!(f, "Only HTTPS URLs are allowed"),
        }
    }
}

impl std::error::Error for SsrfError {}

/// Result alias for SSRF operations.
pub type Result<T> = std::result::Result<T, SsrfError>;

// ---------------------------------------------------------------------------
// ValidatedUrl newtype
// ---------------------------------------------------------------------------

/// A URL that has been validated against SSRF attacks.
///
/// This newtype wraps a `String` and can only be constructed through
/// [`SsrfValidator::validate_url`], guaranteeing the URL has passed all
/// configured security checks.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidatedUrl(String);

impl ValidatedUrl {
    /// Returns the validated URL as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes `self` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ValidatedUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ValidatedUrl {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// CIDR range representation
// ---------------------------------------------------------------------------

/// A CIDR block (e.g. `10.0.0.0/8`) that can check whether an IP address
/// falls within the range. Supports both IPv4 and IPv6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CidrRange {
    network: IpAddr,
    prefix_len: u8,
}

impl CidrRange {
    /// Create a new CIDR range from a network address and prefix length.
    ///
    /// # Panics
    ///
    /// Panics if `prefix_len` exceeds 32 for IPv4 or 128 for IPv6.
    pub fn new(network: IpAddr, prefix_len: u8) -> Self {
        match &network {
            IpAddr::V4(_) => assert!(prefix_len <= 32, "IPv4 prefix length must be <= 32"),
            IpAddr::V6(_) => assert!(prefix_len <= 128, "IPv6 prefix length must be <= 128"),
        }
        Self {
            network,
            prefix_len,
        }
    }

    /// Parse a CIDR string such as `"10.0.0.0/8"` or `"::1/128"`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let (addr_str, prefix_str) = s
            .split_once('/')
            .ok_or_else(|| format!("missing '/' in CIDR notation: {s}"))?;
        let addr: IpAddr = addr_str
            .parse()
            .map_err(|e| format!("invalid IP in CIDR: {e}"))?;
        let prefix_len: u8 = prefix_str
            .parse()
            .map_err(|e| format!("invalid prefix length: {e}"))?;
        let max = if addr.is_ipv4() { 32 } else { 128 };
        if prefix_len > max {
            return Err(format!("prefix length {prefix_len} exceeds max {max}"));
        }
        Ok(Self::new(addr, prefix_len))
    }

    /// Check whether `ip` falls within this CIDR range.
    pub fn contains(&self, ip: &IpAddr) -> bool {
        match (&self.network, ip) {
            (IpAddr::V4(net), IpAddr::V4(addr)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let net_bits = u32::from(*net);
                let addr_bits = u32::from(*addr);
                let mask = u32::MAX << (32 - self.prefix_len);
                (net_bits & mask) == (addr_bits & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(addr)) => {
                if self.prefix_len == 0 {
                    return true;
                }
                let net_bits = u128::from(*net);
                let addr_bits = u128::from(*addr);
                let mask = u128::MAX << (128 - self.prefix_len);
                (net_bits & mask) == (addr_bits & mask)
            }
            _ => false, // IPv4 range vs IPv6 address or vice-versa
        }
    }
}

// ---------------------------------------------------------------------------
// Default blocked ranges and metadata endpoints
// ---------------------------------------------------------------------------

/// Returns the default set of private/reserved IP ranges that should be blocked.
fn default_blocked_ranges() -> Vec<CidrRange> {
    vec![
        // IPv4 private ranges
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)), 8), // Private Class A
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 0)), 12), // Private Class B
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(192, 168, 0, 0)), 16), // Private Class C
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 0)), 8), // Loopback
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(169, 254, 0, 0)), 16), // Link-local
        CidrRange::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8),  // Current network
        // IPv6 reserved ranges
        CidrRange::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 128), // IPv6 loopback
        CidrRange::new(IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0)), 7), // Unique local
        CidrRange::new(IpAddr::V6(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0)), 10), // Link-local
        CidrRange::new(IpAddr::V6(Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0)), 8), // Multicast
    ]
}

/// Cloud provider metadata IPs that are always blocked.
const CLOUD_METADATA_IPS: &[&str] = &[
    "169.254.169.254", // AWS, GCP, Azure, DigitalOcean, Oracle Cloud
    "169.254.170.2",   // AWS ECS task metadata
    "100.100.100.200", // Alibaba Cloud metadata
];

/// Cloud provider metadata hostnames that are always blocked.
const CLOUD_METADATA_HOSTNAMES: &[&str] = &[
    "metadata.google.internal", // GCP
    "metadata",                 // Generic
    "instance-data",            // AWS EC2
];

/// Localhost hostname variations.
const LOCALHOST_NAMES: &[&str] = &["localhost", "localhost.localdomain"];

// ---------------------------------------------------------------------------
// SsrfValidator
// ---------------------------------------------------------------------------

/// Validator that checks URLs for potential SSRF attacks.
///
/// Configurable through [`SsrfValidatorBuilder`].
///
/// # Default behaviour
///
/// By default the validator blocks:
/// - Private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
/// - Loopback addresses (127.0.0.0/8, ::1)
/// - Link-local addresses (169.254.0.0/16, fe80::/10)
/// - Cloud metadata endpoints (AWS, GCP, Azure, Alibaba)
/// - Non-HTTP/HTTPS schemes
///
/// # Example
///
/// ```rust
/// use cognis_core::security::ssrf::SsrfValidator;
///
/// let validator = SsrfValidator::default();
/// let validated = validator.validate_url("https://api.example.com/hook").unwrap();
/// assert_eq!(validated.as_str(), "https://api.example.com/hook");
/// ```
#[derive(Debug, Clone)]
pub struct SsrfValidator {
    blocked_ranges: Vec<CidrRange>,
    allowed_domains: Vec<String>,
    allowed_ips: Vec<IpAddr>,
    cloud_metadata_ips: Vec<String>,
    cloud_metadata_hostnames: Vec<String>,
    allow_private: bool,
    allow_http: bool,
}

impl Default for SsrfValidator {
    fn default() -> Self {
        Self {
            blocked_ranges: default_blocked_ranges(),
            allowed_domains: Vec::new(),
            allowed_ips: Vec::new(),
            cloud_metadata_ips: CLOUD_METADATA_IPS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            cloud_metadata_hostnames: CLOUD_METADATA_HOSTNAMES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            allow_private: false,
            allow_http: true,
        }
    }
}

impl SsrfValidator {
    /// Create a new builder for configuring an `SsrfValidator`.
    pub fn builder() -> SsrfValidatorBuilder {
        SsrfValidatorBuilder::new()
    }

    /// Validate a URL string against SSRF rules.
    ///
    /// Returns a [`ValidatedUrl`] on success or an [`SsrfError`] describing the
    /// specific violation.
    pub fn validate_url(&self, url: &str) -> Result<ValidatedUrl> {
        let (scheme, hostname, _port) = parse_url_parts(url)?;

        // Check scheme
        if !self.allow_http && scheme != "https" {
            return Err(SsrfError::HttpNotAllowed);
        }
        if scheme != "http" && scheme != "https" {
            return Err(SsrfError::InvalidScheme(scheme.to_string()));
        }

        let hostname_lower = hostname.to_lowercase();

        // ALWAYS block cloud metadata hostnames (even with allow_private)
        if self.is_cloud_metadata_hostname(&hostname_lower) {
            return Err(SsrfError::CloudMetadata(hostname_lower));
        }

        // Check allowlist — if the domain is explicitly allowed, skip further checks
        if self.is_domain_allowed(&hostname_lower) {
            return Ok(ValidatedUrl(url.to_string()));
        }

        // Try to parse hostname as IP
        if let Ok(ip) = hostname.parse::<IpAddr>() {
            // Check IP allowlist
            if self.allowed_ips.contains(&ip) {
                return Ok(ValidatedUrl(url.to_string()));
            }

            // Always block cloud metadata IPs
            if self.is_cloud_metadata_ip(hostname) {
                return Err(SsrfError::CloudMetadata(hostname.to_string()));
            }

            // Check localhost
            if !self.allow_private && is_loopback(&ip) {
                return Err(SsrfError::Localhost(hostname.to_string()));
            }

            // Check blocked ranges
            if !self.allow_private && self.is_ip_blocked(&ip) {
                return Err(SsrfError::PrivateIp(hostname.to_string()));
            }
        } else {
            // Hostname is not a raw IP — check for localhost names
            if !self.allow_private && LOCALHOST_NAMES.contains(&hostname_lower.as_str()) {
                return Err(SsrfError::Localhost(hostname_lower));
            }
        }

        Ok(ValidatedUrl(url.to_string()))
    }

    /// Non-throwing convenience method — returns `true` if the URL passes validation.
    pub fn is_safe_url(&self, url: &str) -> bool {
        self.validate_url(url).is_ok()
    }

    // -- internal helpers ---------------------------------------------------

    fn is_cloud_metadata_hostname(&self, hostname: &str) -> bool {
        self.cloud_metadata_hostnames.iter().any(|h| h == hostname)
    }

    fn is_cloud_metadata_ip(&self, ip_str: &str) -> bool {
        self.cloud_metadata_ips.iter().any(|s| s == ip_str)
    }

    fn is_domain_allowed(&self, hostname: &str) -> bool {
        self.allowed_domains.iter().any(|d| d == hostname)
    }

    fn is_ip_blocked(&self, ip: &IpAddr) -> bool {
        self.blocked_ranges.iter().any(|r| r.contains(ip))
    }
}

// ---------------------------------------------------------------------------
// SsrfValidatorBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring an [`SsrfValidator`].
///
/// # Example
///
/// ```rust
/// use cognis_core::security::ssrf::SsrfValidator;
///
/// let validator = SsrfValidator::builder()
///     .allow_domain("internal.mycompany.com")
///     .allow_http(false) // HTTPS only
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct SsrfValidatorBuilder {
    inner: SsrfValidator,
}

impl SsrfValidatorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            inner: SsrfValidator::default(),
        }
    }

    /// Add a domain to the allowlist. Allowed domains bypass private-IP checks
    /// but cloud metadata hostnames are **never** allowed.
    pub fn allow_domain(mut self, domain: &str) -> Self {
        self.inner.allowed_domains.push(domain.to_lowercase());
        self
    }

    /// Add an IP address to the allowlist.
    pub fn allow_ip(mut self, ip: IpAddr) -> Self {
        self.inner.allowed_ips.push(ip);
        self
    }

    /// Add an additional CIDR range to the blocked list.
    pub fn block_range(mut self, range: CidrRange) -> Self {
        self.inner.blocked_ranges.push(range);
        self
    }

    /// If `true`, private IPs and localhost are allowed (cloud metadata is still blocked).
    pub fn allow_private(mut self, allow: bool) -> Self {
        self.inner.allow_private = allow;
        self
    }

    /// If `true` (the default), both HTTP and HTTPS are accepted.
    /// Set to `false` to require HTTPS.
    pub fn allow_http(mut self, allow: bool) -> Self {
        self.inner.allow_http = allow;
        self
    }

    /// Add a custom cloud metadata IP to always block.
    pub fn block_cloud_metadata_ip(mut self, ip: &str) -> Self {
        self.inner.cloud_metadata_ips.push(ip.to_string());
        self
    }

    /// Add a custom cloud metadata hostname to always block.
    pub fn block_cloud_metadata_hostname(mut self, hostname: &str) -> Self {
        self.inner
            .cloud_metadata_hostnames
            .push(hostname.to_lowercase());
        self
    }

    /// Consume the builder and return the configured [`SsrfValidator`].
    pub fn build(self) -> SsrfValidator {
        self.inner
    }
}

impl Default for SsrfValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// URL parsing helpers (no external deps)
// ---------------------------------------------------------------------------

/// Lightweight URL parser that extracts scheme, hostname, and optional port.
/// We avoid pulling in the `url` crate to keep dependencies minimal.
fn parse_url_parts(url: &str) -> Result<(&str, &str, Option<u16>)> {
    // scheme
    let colon = url
        .find("://")
        .ok_or_else(|| SsrfError::InvalidUrl("missing scheme (e.g. https://)".into()))?;
    let scheme = &url[..colon];

    let after_scheme = &url[colon + 3..]; // skip "://"

    // strip userinfo (user:pass@)
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    let authority = match authority.rfind('@') {
        Some(idx) => &authority[idx + 1..],
        None => authority,
    };

    if authority.is_empty() {
        return Err(SsrfError::MissingHostname);
    }

    // Handle IPv6 addresses in brackets: [::1]:port
    if authority.starts_with('[') {
        let bracket_end = authority
            .find(']')
            .ok_or_else(|| SsrfError::InvalidUrl("unclosed bracket in IPv6 address".into()))?;
        let hostname = &authority[1..bracket_end];
        let port =
            if authority.len() > bracket_end + 1 && authority.as_bytes()[bracket_end + 1] == b':' {
                authority[bracket_end + 2..].parse::<u16>().ok()
            } else {
                None
            };
        return Ok((scheme, hostname, port));
    }

    // IPv4 / hostname with optional port
    let (hostname, port) = match authority.rsplit_once(':') {
        Some((h, p)) => {
            if let Ok(port_num) = p.parse::<u16>() {
                (h, Some(port_num))
            } else {
                // The part after ':' is not a valid port — treat the whole thing as hostname
                (authority, None)
            }
        }
        None => (authority, None),
    };

    if hostname.is_empty() {
        return Err(SsrfError::MissingHostname);
    }

    Ok((scheme, hostname, port))
}

/// Returns `true` if `ip` is a loopback address.
fn is_loopback(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => *v6 == Ipv6Addr::LOCALHOST,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    // -----------------------------------------------------------------------
    // CidrRange tests
    // -----------------------------------------------------------------------

    #[test]
    fn cidr_parse_ipv4() {
        let r = CidrRange::parse("10.0.0.0/8").unwrap();
        assert_eq!(r.prefix_len, 8);
        assert_eq!(r.network, IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)));
    }

    #[test]
    fn cidr_parse_ipv6() {
        let r = CidrRange::parse("::1/128").unwrap();
        assert_eq!(r.prefix_len, 128);
    }

    #[test]
    fn cidr_parse_invalid_no_slash() {
        assert!(CidrRange::parse("10.0.0.0").is_err());
    }

    #[test]
    fn cidr_parse_invalid_prefix() {
        assert!(CidrRange::parse("10.0.0.0/33").is_err());
    }

    #[test]
    fn cidr_contains_ipv4_class_a() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)), 8);
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(10, 255, 255, 255))));
        assert!(!r.contains(&IpAddr::V4(Ipv4Addr::new(11, 0, 0, 0))));
    }

    #[test]
    fn cidr_contains_ipv4_class_b() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 0)), 12);
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1))));
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(172, 31, 255, 255))));
        assert!(!r.contains(&IpAddr::V4(Ipv4Addr::new(172, 32, 0, 0))));
    }

    #[test]
    fn cidr_contains_ipv4_class_c() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::new(192, 168, 0, 0)), 16);
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
        assert!(!r.contains(&IpAddr::V4(Ipv4Addr::new(192, 169, 0, 0))));
    }

    #[test]
    fn cidr_contains_ipv6() {
        let r = CidrRange::new(IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0)), 7);
        assert!(r.contains(&IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1))));
        assert!(r.contains(&IpAddr::V6(Ipv6Addr::new(0xfdff, 0xff, 0, 0, 0, 0, 0, 0))));
        assert!(!r.contains(&IpAddr::V6(Ipv6Addr::new(0xfe00, 0, 0, 0, 0, 0, 0, 0))));
    }

    #[test]
    fn cidr_ipv4_does_not_contain_ipv6() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 0)), 8);
        assert!(!r.contains(&IpAddr::V6(Ipv6Addr::LOCALHOST)));
    }

    #[test]
    fn cidr_zero_prefix_matches_all() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))));
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(255, 255, 255, 255))));
    }

    #[test]
    fn cidr_32_prefix_exact_match() {
        let r = CidrRange::new(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)), 32);
        assert!(r.contains(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(!r.contains(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 9))));
    }

    // -----------------------------------------------------------------------
    // ValidatedUrl tests
    // -----------------------------------------------------------------------

    #[test]
    fn validated_url_display_and_as_str() {
        let v = ValidatedUrl("https://example.com".into());
        assert_eq!(v.as_str(), "https://example.com");
        assert_eq!(v.to_string(), "https://example.com");
        assert_eq!(v.as_ref(), "https://example.com");
    }

    #[test]
    fn validated_url_into_inner() {
        let v = ValidatedUrl("https://example.com".into());
        let s = v.into_inner();
        assert_eq!(s, "https://example.com");
    }

    // -----------------------------------------------------------------------
    // URL parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_simple_https() {
        let (scheme, host, port) = parse_url_parts("https://example.com/path").unwrap();
        assert_eq!(scheme, "https");
        assert_eq!(host, "example.com");
        assert_eq!(port, None);
    }

    #[test]
    fn parse_with_port() {
        let (scheme, host, port) = parse_url_parts("http://example.com:8080/path").unwrap();
        assert_eq!(scheme, "http");
        assert_eq!(host, "example.com");
        assert_eq!(port, Some(8080));
    }

    #[test]
    fn parse_ipv4_host() {
        let (_, host, _) = parse_url_parts("http://192.168.1.1/foo").unwrap();
        assert_eq!(host, "192.168.1.1");
    }

    #[test]
    fn parse_ipv6_host() {
        let (_, host, _) = parse_url_parts("http://[::1]:8080/foo").unwrap();
        assert_eq!(host, "::1");
    }

    #[test]
    fn parse_with_userinfo() {
        let (_, host, _) = parse_url_parts("http://user:pass@example.com/path").unwrap();
        assert_eq!(host, "example.com");
    }

    #[test]
    fn parse_no_scheme_fails() {
        assert!(parse_url_parts("example.com/path").is_err());
    }

    #[test]
    fn parse_empty_hostname_fails() {
        assert!(parse_url_parts("http:///path").is_err());
    }

    // -----------------------------------------------------------------------
    // Default SsrfValidator — public URLs (should pass)
    // -----------------------------------------------------------------------

    #[test]
    fn allows_public_https() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("https://example.com").is_ok());
    }

    #[test]
    fn allows_public_http() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://example.com").is_ok());
    }

    #[test]
    fn allows_public_https_with_path() {
        let v = SsrfValidator::default();
        assert!(v
            .validate_url("https://hooks.slack.com/services/xxx")
            .is_ok());
    }

    #[test]
    fn allows_public_ip() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("https://8.8.8.8/dns-query").is_ok());
    }

    // -----------------------------------------------------------------------
    // Default SsrfValidator — private IPs (should block)
    // -----------------------------------------------------------------------

    #[test]
    fn blocks_private_class_a() {
        let v = SsrfValidator::default();
        let err = v.validate_url("http://10.0.0.1").unwrap_err();
        assert!(matches!(err, SsrfError::PrivateIp(_)));
    }

    #[test]
    fn blocks_private_class_b() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://172.16.0.1").is_err());
    }

    #[test]
    fn blocks_private_class_c() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://192.168.1.1").is_err());
    }

    #[test]
    fn blocks_private_class_b_upper_bound() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://172.31.255.255").is_err());
    }

    #[test]
    fn allows_non_private_172() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://172.32.0.1").is_ok());
    }

    // -----------------------------------------------------------------------
    // Localhost blocking
    // -----------------------------------------------------------------------

    #[test]
    fn blocks_localhost_name() {
        let v = SsrfValidator::default();
        let err = v.validate_url("http://localhost:8080").unwrap_err();
        assert!(matches!(err, SsrfError::Localhost(_)));
    }

    #[test]
    fn blocks_localhost_localdomain() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://localhost.localdomain").is_err());
    }

    #[test]
    fn blocks_127_0_0_1() {
        let v = SsrfValidator::default();
        let err = v.validate_url("http://127.0.0.1:9090").unwrap_err();
        assert!(matches!(err, SsrfError::Localhost(_)));
    }

    #[test]
    fn blocks_127_x_x_x() {
        let v = SsrfValidator::default();
        // 127.1.2.3 is still in the loopback range
        assert!(v.validate_url("http://127.1.2.3").is_err());
    }

    #[test]
    fn blocks_ipv6_loopback() {
        let v = SsrfValidator::default();
        let err = v.validate_url("http://[::1]:3000").unwrap_err();
        assert!(matches!(err, SsrfError::Localhost(_)));
    }

    // -----------------------------------------------------------------------
    // Cloud metadata blocking
    // -----------------------------------------------------------------------

    #[test]
    fn blocks_aws_metadata_ip() {
        let v = SsrfValidator::default();
        let err = v
            .validate_url("http://169.254.169.254/latest/meta-data/")
            .unwrap_err();
        assert!(matches!(err, SsrfError::CloudMetadata(_)));
    }

    #[test]
    fn blocks_aws_ecs_metadata() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://169.254.170.2/v2/metadata").is_err());
    }

    #[test]
    fn blocks_alibaba_metadata() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://100.100.100.200/latest").is_err());
    }

    #[test]
    fn blocks_gcp_metadata_hostname() {
        let v = SsrfValidator::default();
        let err = v
            .validate_url("http://metadata.google.internal/computeMetadata/v1/")
            .unwrap_err();
        assert!(matches!(err, SsrfError::CloudMetadata(_)));
    }

    #[test]
    fn blocks_generic_metadata_hostname() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://metadata/latest").is_err());
    }

    #[test]
    fn blocks_instance_data_hostname() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://instance-data/latest").is_err());
    }

    #[test]
    fn blocks_cloud_metadata_even_with_allow_private() {
        let v = SsrfValidator::builder().allow_private(true).build();
        // Cloud metadata must ALWAYS be blocked
        assert!(v
            .validate_url("http://169.254.169.254/latest/meta-data/")
            .is_err());
        assert!(v
            .validate_url("http://metadata.google.internal/foo")
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Scheme validation
    // -----------------------------------------------------------------------

    #[test]
    fn blocks_ftp_scheme() {
        let v = SsrfValidator::default();
        let err = v.validate_url("ftp://example.com/file").unwrap_err();
        assert!(matches!(err, SsrfError::InvalidScheme(_)));
    }

    #[test]
    fn blocks_file_scheme() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn blocks_javascript_scheme() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("javascript://example.com").is_err());
    }

    #[test]
    fn https_only_blocks_http() {
        let v = SsrfValidator::builder().allow_http(false).build();
        let err = v.validate_url("http://example.com").unwrap_err();
        assert!(matches!(err, SsrfError::HttpNotAllowed));
    }

    #[test]
    fn https_only_allows_https() {
        let v = SsrfValidator::builder().allow_http(false).build();
        assert!(v.validate_url("https://example.com").is_ok());
    }

    // -----------------------------------------------------------------------
    // allow_private flag
    // -----------------------------------------------------------------------

    #[test]
    fn allow_private_permits_localhost() {
        let v = SsrfValidator::builder().allow_private(true).build();
        assert!(v.validate_url("http://localhost:8080").is_ok());
    }

    #[test]
    fn allow_private_permits_127_0_0_1() {
        let v = SsrfValidator::builder().allow_private(true).build();
        assert!(v.validate_url("http://127.0.0.1:3000").is_ok());
    }

    #[test]
    fn allow_private_permits_private_ips() {
        let v = SsrfValidator::builder().allow_private(true).build();
        assert!(v.validate_url("http://10.0.0.1").is_ok());
        assert!(v.validate_url("http://192.168.1.1").is_ok());
        assert!(v.validate_url("http://172.16.0.1").is_ok());
    }

    // -----------------------------------------------------------------------
    // Allowlisting domains / IPs
    // -----------------------------------------------------------------------

    #[test]
    fn allowed_domain_bypasses_checks() {
        let v = SsrfValidator::builder()
            .allow_domain("internal.mycompany.com")
            .build();
        assert!(v.validate_url("http://internal.mycompany.com/api").is_ok());
    }

    #[test]
    fn allowed_domain_case_insensitive() {
        let v = SsrfValidator::builder()
            .allow_domain("Internal.MyCompany.com")
            .build();
        assert!(v.validate_url("http://internal.mycompany.com/api").is_ok());
    }

    #[test]
    fn allowed_ip_bypasses_private_check() {
        let v = SsrfValidator::builder()
            .allow_ip(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)))
            .build();
        assert!(v.validate_url("http://10.0.0.5/hook").is_ok());
        // Other private IPs still blocked
        assert!(v.validate_url("http://10.0.0.6/hook").is_err());
    }

    #[test]
    fn allowed_domain_does_not_bypass_cloud_metadata() {
        // Even if you allowlist the cloud metadata hostname, it must still be blocked
        let v = SsrfValidator::builder()
            .allow_domain("metadata.google.internal")
            .build();
        assert!(v
            .validate_url("http://metadata.google.internal/foo")
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Builder: custom blocked ranges
    // -----------------------------------------------------------------------

    #[test]
    fn builder_add_custom_blocked_range() {
        let v = SsrfValidator::builder()
            .block_range(CidrRange::parse("203.0.113.0/24").unwrap())
            .build();
        assert!(v.validate_url("http://203.0.113.50").is_err());
        assert!(v.validate_url("http://203.0.114.1").is_ok());
    }

    #[test]
    fn builder_add_custom_cloud_metadata_ip() {
        let v = SsrfValidator::builder()
            .block_cloud_metadata_ip("10.10.10.10")
            .build();
        let err = v.validate_url("http://10.10.10.10/meta").unwrap_err();
        assert!(matches!(err, SsrfError::CloudMetadata(_)));
    }

    #[test]
    fn builder_add_custom_cloud_metadata_hostname() {
        let v = SsrfValidator::builder()
            .block_cloud_metadata_hostname("custom-metadata.internal")
            .build();
        let err = v
            .validate_url("http://custom-metadata.internal/v1/")
            .unwrap_err();
        assert!(matches!(err, SsrfError::CloudMetadata(_)));
    }

    // -----------------------------------------------------------------------
    // is_safe_url convenience
    // -----------------------------------------------------------------------

    #[test]
    fn is_safe_url_returns_true_for_public() {
        let v = SsrfValidator::default();
        assert!(v.is_safe_url("https://example.com"));
    }

    #[test]
    fn is_safe_url_returns_false_for_private() {
        let v = SsrfValidator::default();
        assert!(!v.is_safe_url("http://10.0.0.1"));
    }

    // -----------------------------------------------------------------------
    // Link-local blocking
    // -----------------------------------------------------------------------

    #[test]
    fn blocks_link_local_169_254() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://169.254.1.1").is_err());
    }

    #[test]
    fn blocks_current_network_0_0_0_0() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://0.0.0.1").is_err());
    }

    // -----------------------------------------------------------------------
    // SsrfError display
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_invalid_scheme() {
        let e = SsrfError::InvalidScheme("ftp".into());
        assert!(e.to_string().contains("ftp"));
    }

    #[test]
    fn error_display_missing_hostname() {
        let e = SsrfError::MissingHostname;
        assert!(e.to_string().contains("hostname"));
    }

    #[test]
    fn error_display_private_ip() {
        let e = SsrfError::PrivateIp("10.0.0.1".into());
        assert!(e.to_string().contains("10.0.0.1"));
    }

    #[test]
    fn error_display_cloud_metadata() {
        let e = SsrfError::CloudMetadata("169.254.169.254".into());
        assert!(e.to_string().contains("169.254.169.254"));
    }

    #[test]
    fn error_display_http_not_allowed() {
        let e = SsrfError::HttpNotAllowed;
        assert!(e.to_string().contains("HTTPS"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn url_with_query_string() {
        let v = SsrfValidator::default();
        assert!(v
            .validate_url("https://example.com/path?foo=bar&baz=1")
            .is_ok());
    }

    #[test]
    fn url_with_fragment() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("https://example.com/path#section").is_ok());
    }

    #[test]
    fn url_with_port_and_path() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("https://example.com:443/api/v1").is_ok());
    }

    #[test]
    fn url_with_userinfo_public() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("https://user:pass@example.com/path").is_ok());
    }

    #[test]
    fn url_with_userinfo_private_blocked() {
        let v = SsrfValidator::default();
        assert!(v
            .validate_url("http://admin:pass@192.168.1.1/admin")
            .is_err());
    }

    #[test]
    fn completely_invalid_url() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("not-a-url").is_err());
    }

    #[test]
    fn empty_string() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("").is_err());
    }

    #[test]
    fn ipv6_private_unique_local() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://[fc00::1]").is_err());
    }

    #[test]
    fn ipv6_link_local() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://[fe80::1]").is_err());
    }

    #[test]
    fn ipv6_public_allowed() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://[2607:f8b0:4004:800::200e]").is_ok());
    }

    #[test]
    fn localhost_uppercase() {
        let v = SsrfValidator::default();
        assert!(v.validate_url("http://LOCALHOST:8080").is_err());
    }

    #[test]
    fn metadata_hostname_uppercase() {
        let v = SsrfValidator::default();
        assert!(v
            .validate_url("http://METADATA.GOOGLE.INTERNAL/foo")
            .is_err());
    }

    #[test]
    fn alibaba_cloud_metadata_blocked() {
        let v = SsrfValidator::default();
        let err = v.validate_url("http://100.100.100.200/latest").unwrap_err();
        assert!(matches!(err, SsrfError::CloudMetadata(_)));
    }

    #[test]
    fn builder_default_equivalent() {
        let v1 = SsrfValidator::default();
        let v2 = SsrfValidatorBuilder::default().build();
        // Both should behave identically
        assert_eq!(
            v1.is_safe_url("https://example.com"),
            v2.is_safe_url("https://example.com")
        );
        assert_eq!(
            v1.is_safe_url("http://10.0.0.1"),
            v2.is_safe_url("http://10.0.0.1")
        );
    }
}
