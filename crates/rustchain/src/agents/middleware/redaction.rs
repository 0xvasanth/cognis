//! PII detection and redaction utilities.
//!
//! Provides detectors for common PII types (email, credit card, IP, MAC, URL)
//! and strategies for handling detected PII (block, redact, mask, hash).

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A custom PII detector function. Takes text and returns any PII matches found.
pub type Detector = Arc<dyn Fn(&str) -> Vec<PIIMatch> + Send + Sync>;

/// A match of personally identifiable information within text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PIIMatch {
    /// The type of PII detected (e.g., "email", "credit_card").
    pub pii_type: String,
    /// The matched value.
    pub value: String,
    /// Start byte offset in the original text.
    pub start: usize,
    /// End byte offset in the original text (exclusive).
    pub end: usize,
}

/// Errors that can occur during PII detection.
#[derive(Debug, Clone)]
pub enum PIIDetectionError {
    /// The input text was invalid or could not be processed.
    InvalidInput(String),
    /// A detector encountered an internal error.
    DetectorError(String),
}

impl fmt::Display for PIIDetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::DetectorError(msg) => write!(f, "Detector error: {}", msg),
        }
    }
}

impl std::error::Error for PIIDetectionError {}

/// Strategy for handling detected PII.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RedactionStrategy {
    /// Block the entire message from proceeding.
    Block,
    /// Replace the PII with a placeholder like `[REDACTED]`.
    Redact,
    /// Mask the PII, preserving length (e.g., `****@****.***`).
    Mask,
    /// Replace the PII with a deterministic hash.
    Hash,
}

/// A rule mapping a PII type to a redaction strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactionRule {
    /// The PII type this rule applies to (e.g., "email", "credit_card").
    pub pii_type: String,
    /// The strategy to apply when this PII type is detected.
    pub strategy: RedactionStrategy,
}

/// A resolved redaction rule: a specific match paired with the strategy to apply.
#[derive(Debug, Clone)]
pub struct ResolvedRedactionRule {
    /// The detected PII match.
    pub pii_match: PIIMatch,
    /// The strategy to apply.
    pub strategy: RedactionStrategy,
}

// ---------------------------------------------------------------------------
// Detectors
// ---------------------------------------------------------------------------

/// Detect email addresses in text.
pub fn detect_email(text: &str) -> Vec<PIIMatch> {
    let mut matches = Vec::new();
    // Simple email pattern: sequences matching local@domain.tld
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'@' {
            // Walk backwards to find start of local part
            let mut start = i;
            while start > 0 {
                let c = bytes[start - 1];
                if c.is_ascii_alphanumeric() || b"._%+-".contains(&c) {
                    start -= 1;
                } else {
                    break;
                }
            }
            // Walk forwards to find end of domain
            let mut end = i + 1;
            let mut has_dot = false;
            while end < len {
                let c = bytes[end];
                if c.is_ascii_alphanumeric() || c == b'.' || c == b'-' {
                    if c == b'.' {
                        has_dot = true;
                    }
                    end += 1;
                } else {
                    break;
                }
            }
            // Trim trailing dots
            while end > i + 1 && bytes[end - 1] == b'.' {
                end -= 1;
            }
            if start < i && end > i + 1 && has_dot {
                let value = &text[start..end];
                matches.push(PIIMatch {
                    pii_type: "email".into(),
                    value: value.to_string(),
                    start,
                    end,
                });
            }
        }
        i += 1;
    }
    matches
}

/// Perform Luhn check on a sequence of digits.
fn luhn_check(digits: &[u8]) -> bool {
    if digits.len() < 2 {
        return false;
    }
    let mut sum: u32 = 0;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut n = (d - b'0') as u32;
        if double {
            n *= 2;
            if n > 9 {
                n -= 9;
            }
        }
        sum += n;
        double = !double;
    }
    sum.is_multiple_of(10)
}

/// Detect credit card numbers in text (13-19 digits, optionally separated by spaces/dashes, validated with Luhn).
pub fn detect_credit_card(text: &str) -> Vec<PIIMatch> {
    let mut matches = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut digits = Vec::new();
            let mut end = i;
            while end < len {
                let c = bytes[end];
                if c.is_ascii_digit() {
                    digits.push(c);
                    end += 1;
                } else if c == b' ' || c == b'-' {
                    // Allow separators only if followed by more digits
                    if end + 1 < len && bytes[end + 1].is_ascii_digit() {
                        end += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if digits.len() >= 13 && digits.len() <= 19 && luhn_check(&digits) {
                let value = &text[start..end];
                matches.push(PIIMatch {
                    pii_type: "credit_card".into(),
                    value: value.to_string(),
                    start,
                    end,
                });
                i = end;
                continue;
            }
            i = end;
            continue;
        }
        i += 1;
    }
    matches
}

/// Detect IPv4 addresses in text.
pub fn detect_ip(text: &str) -> Vec<PIIMatch> {
    let mut matches = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut octets = Vec::new();
            let mut current_num = String::new();
            let mut j = i;

            while j < len {
                let c = bytes[j];
                if c.is_ascii_digit() {
                    current_num.push(c as char);
                    j += 1;
                } else if c == b'.' && !current_num.is_empty() && octets.len() < 3 {
                    octets.push(current_num.clone());
                    current_num.clear();
                    j += 1;
                } else {
                    break;
                }
            }
            if !current_num.is_empty() {
                octets.push(current_num);
            }

            if octets.len() == 4 {
                let valid = octets.iter().all(|o| {
                    if let Ok(n) = o.parse::<u16>() {
                        n <= 255 && (o.len() == 1 || !o.starts_with('0'))
                    } else {
                        false
                    }
                });
                if valid {
                    // Check that it's not preceded/followed by alphanumeric
                    let preceded_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
                    let followed_ok = j >= len || !bytes[j].is_ascii_alphanumeric();
                    if preceded_ok && followed_ok {
                        let value = &text[start..j];
                        matches.push(PIIMatch {
                            pii_type: "ip".into(),
                            value: value.to_string(),
                            start,
                            end: j,
                        });
                        i = j;
                        continue;
                    }
                }
            }
            i = start + 1;
            continue;
        }
        i += 1;
    }
    matches
}

/// Detect MAC addresses in text (formats: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX).
pub fn detect_mac_address(text: &str) -> Vec<PIIMatch> {
    let mut matches = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();

    // MAC address is exactly 17 chars: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX
    if len < 17 {
        return matches;
    }

    let mut i = 0;
    while i + 17 <= len {
        let slice = &bytes[i..i + 17];
        let sep = slice[2];
        if (sep == b':' || sep == b'-')
            && is_hex_pair(&slice[0..2])
            && slice[5] == sep
            && is_hex_pair(&slice[3..5])
            && slice[8] == sep
            && is_hex_pair(&slice[6..8])
            && slice[11] == sep
            && is_hex_pair(&slice[9..11])
            && slice[14] == sep
            && is_hex_pair(&slice[12..14])
            && is_hex_pair(&slice[15..17])
        {
            // Check boundaries
            let preceded_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let followed_ok = i + 17 >= len || !bytes[i + 17].is_ascii_alphanumeric();
            if preceded_ok && followed_ok {
                let value = &text[i..i + 17];
                matches.push(PIIMatch {
                    pii_type: "mac_address".into(),
                    value: value.to_string(),
                    start: i,
                    end: i + 17,
                });
                i += 17;
                continue;
            }
        }
        i += 1;
    }
    matches
}

fn is_hex_pair(bytes: &[u8]) -> bool {
    bytes.len() == 2
        && bytes[0].is_ascii_hexdigit()
        && bytes[1].is_ascii_hexdigit()
}

/// Detect URLs in text.
pub fn detect_url(text: &str) -> Vec<PIIMatch> {
    let mut matches = Vec::new();
    let prefixes = ["https://", "http://", "ftp://"];

    for prefix in &prefixes {
        let mut search_from = 0;
        while let Some(idx) = text[search_from..].find(prefix) {
            let start = search_from + idx;
            let mut end = start + prefix.len();
            let bytes = text.as_bytes();
            // URL continues until whitespace or certain delimiters
            while end < bytes.len() {
                let c = bytes[end];
                if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t'
                    || c == b'"' || c == b'\'' || c == b'>' || c == b'<'
                {
                    break;
                }
                end += 1;
            }
            // Trim trailing punctuation that's likely not part of the URL
            while end > start + prefix.len() {
                let c = bytes[end - 1];
                if c == b'.' || c == b',' || c == b';' || c == b')' {
                    end -= 1;
                } else {
                    break;
                }
            }
            if end > start + prefix.len() {
                let value = &text[start..end];
                matches.push(PIIMatch {
                    pii_type: "url".into(),
                    value: value.to_string(),
                    start,
                    end,
                });
            }
            search_from = end;
        }
    }
    matches
}

// ---------------------------------------------------------------------------
// Built-in detector registry
// ---------------------------------------------------------------------------

/// Returns a mapping of PII type names to their built-in detector functions.
pub fn builtin_detectors() -> Vec<(&'static str, Detector)> {
    vec![
        ("email", Arc::new(detect_email) as Detector),
        ("credit_card", Arc::new(detect_credit_card) as Detector),
        ("ip", Arc::new(detect_ip) as Detector),
        ("mac_address", Arc::new(detect_mac_address) as Detector),
        ("url", Arc::new(detect_url) as Detector),
    ]
}

/// Get a built-in detector by PII type name, if one exists.
pub fn get_builtin_detector(pii_type: &str) -> Option<Detector> {
    match pii_type {
        "email" => Some(Arc::new(detect_email) as Detector),
        "credit_card" => Some(Arc::new(detect_credit_card) as Detector),
        "ip" => Some(Arc::new(detect_ip) as Detector),
        "mac_address" => Some(Arc::new(detect_mac_address) as Detector),
        "url" => Some(Arc::new(detect_url) as Detector),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Strategy application
// ---------------------------------------------------------------------------

/// Apply a redaction strategy to a PII match within text, returning the modified text.
///
/// Returns `Err(PIIDetectionError)` for `Block` strategy — the caller should prevent
/// the message from proceeding.
pub fn apply_strategy(
    text: &str,
    pii_match: &PIIMatch,
    strategy: &RedactionStrategy,
) -> std::result::Result<String, PIIDetectionError> {
    match strategy {
        RedactionStrategy::Block => {
            Err(PIIDetectionError::InvalidInput(format!(
                "PII of type '{}' detected and blocked",
                pii_match.pii_type
            )))
        }
        RedactionStrategy::Redact => {
            let mut result = String::with_capacity(text.len());
            result.push_str(&text[..pii_match.start]);
            result.push_str(&format!("[REDACTED_{}]", pii_match.pii_type.to_uppercase()));
            result.push_str(&text[pii_match.end..]);
            Ok(result)
        }
        RedactionStrategy::Mask => {
            let mut result = String::with_capacity(text.len());
            result.push_str(&text[..pii_match.start]);
            let mask = mask_by_type(&pii_match.pii_type, &pii_match.value);
            result.push_str(&mask);
            result.push_str(&text[pii_match.end..]);
            Ok(result)
        }
        RedactionStrategy::Hash => {
            let mut result = String::with_capacity(text.len());
            result.push_str(&text[..pii_match.start]);
            // Non-cryptographic DJB2 hash — suitable for obfuscation only, not security.
            let hash = simple_hash(&pii_match.value);
            result.push_str(&format!("[HASH:{}]", hash));
            result.push_str(&text[pii_match.end..]);
            Ok(result)
        }
    }
}

/// Type-specific masking for PII values.
///
/// - Email: `u***@****.com` (keep first char of local part, mask domain except TLD)
/// - Credit card: `****-****-****-1234` (show last 4 digits)
/// - IP: `*.*.*.<last_octet>`
/// - Other: uniform `*` replacement preserving non-alphanumeric characters
fn mask_by_type(pii_type: &str, value: &str) -> String {
    match pii_type {
        "email" => mask_email(value),
        "credit_card" => mask_credit_card(value),
        "ip" => mask_ip(value),
        _ => uniform_mask(value),
    }
}

/// Mask an email: keep first char of local part, mask rest, mask domain except TLD.
fn mask_email(value: &str) -> String {
    if let Some(at_pos) = value.find('@') {
        let local = &value[..at_pos];
        let domain = &value[at_pos + 1..];

        // Local part: keep first char, mask the rest
        let masked_local: String = local
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c } else if c.is_alphanumeric() { '*' } else { c })
            .collect();

        // Domain: mask everything except the TLD (last segment after last dot)
        let masked_domain = if let Some(last_dot) = domain.rfind('.') {
            let domain_body = &domain[..last_dot];
            let tld = &domain[last_dot..]; // includes the dot
            let masked_body: String = domain_body
                .chars()
                .map(|c| if c.is_alphanumeric() { '*' } else { c })
                .collect();
            format!("{}{}", masked_body, tld)
        } else {
            uniform_mask(domain)
        };

        format!("{}@{}", masked_local, masked_domain)
    } else {
        uniform_mask(value)
    }
}

/// Mask a credit card: show last 4 digits in format `****-****-****-NNNN`.
fn mask_credit_card(value: &str) -> String {
    let digits: Vec<char> = value.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 4 {
        let last4: String = digits[digits.len() - 4..].iter().collect();
        format!("****-****-****-{}", last4)
    } else {
        uniform_mask(value)
    }
}

/// Mask an IP address: `*.*.*.<last_octet>`.
fn mask_ip(value: &str) -> String {
    let octets: Vec<&str> = value.split('.').collect();
    if octets.len() == 4 {
        format!("*.*.*.{}", octets[3])
    } else {
        uniform_mask(value)
    }
}

/// Uniform mask: replace alphanumeric chars with `*`, preserve others.
fn uniform_mask(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_alphanumeric() { '*' } else { c })
        .collect()
}

/// Apply multiple PII matches with a single strategy, processing from end to start
/// to preserve byte offsets.
pub fn apply_matches(
    text: &str,
    mut matches: Vec<PIIMatch>,
    strategy: &RedactionStrategy,
) -> std::result::Result<String, PIIDetectionError> {
    if matches.is_empty() {
        return Ok(text.to_string());
    }
    // Sort by start position descending so replacements don't invalidate offsets.
    matches.sort_by(|a, b| b.start.cmp(&a.start));
    let mut result = text.to_string();
    for m in &matches {
        result = apply_strategy(&result, m, strategy)?;
    }
    Ok(result)
}

/// Apply multiple resolved rules to text. Rules are applied from right to left to preserve offsets.
///
/// Returns `Err` if any rule uses the `Block` strategy.
pub fn apply_rules(
    text: &str,
    mut rules: Vec<ResolvedRedactionRule>,
) -> std::result::Result<String, PIIDetectionError> {
    // Sort by start position descending so replacements don't invalidate offsets.
    rules.sort_by(|a, b| b.pii_match.start.cmp(&a.pii_match.start));
    let mut result = text.to_string();
    for rule in &rules {
        result = apply_strategy(&result, &rule.pii_match, &rule.strategy)?;
    }
    Ok(result)
}

/// Simple non-cryptographic hash for PII obfuscation.
fn simple_hash(input: &str) -> String {
    let mut h: u64 = 5381;
    for byte in input.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u64);
    }
    format!("{:016x}", h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_email_basic() {
        let matches = detect_email("Contact us at user@example.com for info.");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].value, "user@example.com");
        assert_eq!(matches[0].pii_type, "email");
    }

    #[test]
    fn test_detect_email_multiple() {
        let matches = detect_email("a@b.com and c@d.org");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_detect_email_no_match() {
        let matches = detect_email("no email here");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_detect_credit_card_valid_visa() {
        // 4111111111111111 is a well-known Luhn-valid test number
        let matches = detect_credit_card("Card: 4111111111111111 end");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, "credit_card");
        assert_eq!(matches[0].value, "4111111111111111");
    }

    #[test]
    fn test_detect_credit_card_with_separators() {
        let matches = detect_credit_card("Card: 4111-1111-1111-1111 end");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_detect_credit_card_invalid_luhn() {
        let matches = detect_credit_card("Card: 1234567890123456 end");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_luhn_check_valid() {
        assert!(luhn_check(b"4111111111111111"));
        assert!(luhn_check(b"79927398713"));
    }

    #[test]
    fn test_luhn_check_invalid() {
        assert!(!luhn_check(b"1234567890123456"));
        assert!(!luhn_check(b"1"));
    }

    #[test]
    fn test_detect_ip_basic() {
        let matches = detect_ip("Server at 192.168.1.1 is up");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].value, "192.168.1.1");
        assert_eq!(matches[0].pii_type, "ip");
    }

    #[test]
    fn test_detect_ip_invalid_octet() {
        let matches = detect_ip("Not an IP: 999.999.999.999");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_detect_mac_address_colon() {
        let matches = detect_mac_address("MAC: AA:BB:CC:DD:EE:FF done");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].value, "AA:BB:CC:DD:EE:FF");
    }

    #[test]
    fn test_detect_mac_address_dash() {
        let matches = detect_mac_address("MAC: AA-BB-CC-DD-EE-FF done");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_detect_mac_address_none() {
        let matches = detect_mac_address("no mac here");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_detect_url_https() {
        let matches = detect_url("Visit https://example.com/path for info.");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].value, "https://example.com/path");
    }

    #[test]
    fn test_detect_url_http() {
        let matches = detect_url("Go to http://foo.bar/baz end");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_apply_strategy_redact() {
        let text = "Email: user@test.com is here";
        let m = PIIMatch {
            pii_type: "email".into(),
            value: "user@test.com".into(),
            start: 7,
            end: 20,
        };
        let result = apply_strategy(text, &m, &RedactionStrategy::Redact).unwrap();
        assert_eq!(result, "Email: [REDACTED_EMAIL] is here");
    }

    #[test]
    fn test_apply_strategy_mask() {
        let text = "Email: user@test.com is here";
        let m = PIIMatch {
            pii_type: "email".into(),
            value: "user@test.com".into(),
            start: 7,
            end: 20,
        };
        let result = apply_strategy(text, &m, &RedactionStrategy::Mask).unwrap();
        // Type-specific email masking: keep first char of local, mask domain except TLD
        assert_eq!(result, "Email: u***@****.com is here");
    }

    #[test]
    fn test_apply_strategy_hash() {
        let text = "Email: user@test.com is here";
        let m = PIIMatch {
            pii_type: "email".into(),
            value: "user@test.com".into(),
            start: 7,
            end: 20,
        };
        let result = apply_strategy(text, &m, &RedactionStrategy::Hash).unwrap();
        assert!(result.starts_with("Email: [HASH:"));
        assert!(result.ends_with("] is here"));
    }

    #[test]
    fn test_apply_strategy_block() {
        let text = "Email: user@test.com is here";
        let m = PIIMatch {
            pii_type: "email".into(),
            value: "user@test.com".into(),
            start: 7,
            end: 20,
        };
        let result = apply_strategy(text, &m, &RedactionStrategy::Block);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn test_apply_rules_multiple() {
        let text = "a@b.com and c@d.org";
        let rules = vec![
            ResolvedRedactionRule {
                pii_match: PIIMatch {
                    pii_type: "email".into(),
                    value: "a@b.com".into(),
                    start: 0,
                    end: 7,
                },
                strategy: RedactionStrategy::Redact,
            },
            ResolvedRedactionRule {
                pii_match: PIIMatch {
                    pii_type: "email".into(),
                    value: "c@d.org".into(),
                    start: 12,
                    end: 19,
                },
                strategy: RedactionStrategy::Redact,
            },
        ];
        let result = apply_rules(text, rules).unwrap();
        assert_eq!(result, "[REDACTED_EMAIL] and [REDACTED_EMAIL]");
    }

    #[test]
    fn test_mask_credit_card() {
        let m = PIIMatch {
            pii_type: "credit_card".into(),
            value: "4111111111111111".into(),
            start: 6,
            end: 22,
        };
        let result = apply_strategy("Card: 4111111111111111 end", &m, &RedactionStrategy::Mask).unwrap();
        assert_eq!(result, "Card: ****-****-****-1111 end");
    }

    #[test]
    fn test_mask_ip() {
        let m = PIIMatch {
            pii_type: "ip".into(),
            value: "192.168.1.1".into(),
            start: 4,
            end: 15,
        };
        let result = apply_strategy("IP: 192.168.1.1 done", &m, &RedactionStrategy::Mask).unwrap();
        assert_eq!(result, "IP: *.*.*.1 done");
    }

    #[test]
    fn test_apply_matches_multiple() {
        let text = "a@b.com and c@d.org";
        let matches = detect_email(text);
        let result = apply_matches(text, matches, &RedactionStrategy::Redact).unwrap();
        assert_eq!(result, "[REDACTED_EMAIL] and [REDACTED_EMAIL]");
    }

    #[test]
    fn test_builtin_detectors() {
        let detectors = builtin_detectors();
        assert_eq!(detectors.len(), 5);
        let names: Vec<&str> = detectors.iter().map(|(name, _)| *name).collect();
        assert!(names.contains(&"email"));
        assert!(names.contains(&"ip"));
    }

    #[test]
    fn test_get_builtin_detector() {
        let detector = get_builtin_detector("email");
        assert!(detector.is_some());
        let matches = detector.unwrap()("test@example.com");
        assert_eq!(matches.len(), 1);

        assert!(get_builtin_detector("unknown").is_none());
    }

    #[test]
    fn test_redaction_strategy_serde() {
        let s = serde_json::to_string(&RedactionStrategy::Mask).unwrap();
        assert_eq!(s, "\"mask\"");
        let d: RedactionStrategy = serde_json::from_str("\"block\"").unwrap();
        assert_eq!(d, RedactionStrategy::Block);
    }

    #[test]
    fn test_pii_match_serde() {
        let m = PIIMatch {
            pii_type: "email".into(),
            value: "a@b.com".into(),
            start: 0,
            end: 7,
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: PIIMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }
}
