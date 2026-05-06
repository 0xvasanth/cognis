//! Runtime validators called by code generated from `#[cognis::tool]` /
//! `#[tools_impl]` (when targeting `crate_path = "cognis2_llm"`).
//!
//! Each helper has a matching schema attribute on the field; the schema
//! attribute emits the JSON Schema keyword the LLM sees, the runtime
//! helper enforces the same constraint after deserialization.

use std::sync::OnceLock;

use cognis2_core::{CognisError, Result};

/// Private re-export of the `regex` crate for use by macro-generated code.
/// Users of the macro do not need `regex` in their own `Cargo.toml`.
#[doc(hidden)]
pub mod __regex {
    pub use regex::*;
}

/// Trait for types that can validate their own field constraints after
/// deserialization. Implemented by macro-generated args structs.
pub trait ValidateArgs {
    /// Run all field validators. Returns the first violation as a
    /// `CognisError::ToolValidation`.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Validate that `value` lies within `[min, max]` (inclusive).
pub fn check_range(field: &str, value: f64, min: Option<f64>, max: Option<f64>) -> Result<()> {
    if value.is_nan() {
        return Err(CognisError::ToolValidation(format!(
            "field `{field}`: value is NaN"
        )));
    }
    if let Some(m) = min {
        if value < m {
            return Err(CognisError::ToolValidation(format!(
                "field `{field}`: {value} is less than minimum {m}"
            )));
        }
    }
    if let Some(m) = max {
        if value > m {
            return Err(CognisError::ToolValidation(format!(
                "field `{field}`: {value} is greater than maximum {m}"
            )));
        }
    }
    Ok(())
}

/// Validate that `len` lies within `[min, max]` (inclusive).
pub fn check_length(field: &str, len: usize, min: Option<usize>, max: Option<usize>) -> Result<()> {
    if let Some(m) = min {
        if len < m {
            return Err(CognisError::ToolValidation(format!(
                "field `{field}`: length {len} is less than minimum {m}"
            )));
        }
    }
    if let Some(m) = max {
        if len > m {
            return Err(CognisError::ToolValidation(format!(
                "field `{field}`: length {len} is greater than maximum {m}"
            )));
        }
    }
    Ok(())
}

/// Validate that `value` is one of the allowed variants.
pub fn check_enum<S: AsRef<str>>(field: &str, value: &str, allowed: &[S]) -> Result<()> {
    if allowed.iter().any(|a| a.as_ref() == value) {
        return Ok(());
    }
    let list = allowed
        .iter()
        .map(|a| format!("`{}`", a.as_ref()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(CognisError::ToolValidation(format!(
        "field `{field}`: \"{value}\" must be one of [{list}]"
    )))
}

/// Validate that `value` matches `re`.
pub fn check_pattern(field: &str, value: &str, re: &regex::Regex) -> Result<()> {
    if re.is_match(value) {
        Ok(())
    } else {
        Err(CognisError::ToolValidation(format!(
            "field `{field}`: \"{value}\" does not match pattern `{}`",
            re.as_str()
        )))
    }
}

/// String formats supported by `#[schema(format(...))]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// RFC 5322-ish email (regex-checked).
    Email,
    /// URI — schema-only (not validated at runtime; grammar is too permissive).
    Uri,
    /// RFC 4122 UUID (regex-checked, case-insensitive).
    Uuid,
    /// ISO 8601 date-time — schema-only.
    DateTime,
    /// IPv4 (parsed via `std::net::Ipv4Addr`).
    Ipv4,
    /// IPv6 (parsed via `std::net::Ipv6Addr`).
    Ipv6,
}

impl Format {
    /// Canonical JSON-Schema `format` keyword value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Format::Email => "email",
            Format::Uri => "uri",
            Format::Uuid => "uuid",
            Format::DateTime => "date-time",
            Format::Ipv4 => "ipv4",
            Format::Ipv6 => "ipv6",
        }
    }
}

/// Validate `value` against the given format. Email and UUID are regex-
/// checked; IPv4/IPv6 are parsed; URI and DateTime are schema-only and
/// always pass at runtime.
pub fn check_format(field: &str, value: &str, fmt: Format) -> Result<()> {
    match fmt {
        Format::Email => {
            static RE: OnceLock<regex::Regex> = OnceLock::new();
            let re = RE.get_or_init(|| {
                regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
            });
            if !re.is_match(value) {
                return Err(CognisError::ToolValidation(format!(
                    "field `{field}`: \"{value}\" is not a valid email"
                )));
            }
        }
        Format::Uuid => {
            static RE: OnceLock<regex::Regex> = OnceLock::new();
            let re = RE.get_or_init(|| {
                regex::Regex::new(
                    r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                )
                .unwrap()
            });
            if !re.is_match(value) {
                return Err(CognisError::ToolValidation(format!(
                    "field `{field}`: \"{value}\" is not a valid UUID"
                )));
            }
        }
        Format::Ipv4 => {
            value.parse::<std::net::Ipv4Addr>().map_err(|_| {
                CognisError::ToolValidation(format!(
                    "field `{field}`: \"{value}\" is not a valid IPv4 address"
                ))
            })?;
        }
        Format::Ipv6 => {
            value.parse::<std::net::Ipv6Addr>().map_err(|_| {
                CognisError::ToolValidation(format!(
                    "field `{field}`: \"{value}\" is not a valid IPv6 address"
                ))
            })?;
        }
        Format::Uri | Format::DateTime => {} // schema-only
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_bounds() {
        assert!(check_range("x", 5.0, Some(0.0), Some(10.0)).is_ok());
        assert!(check_range("x", -1.0, Some(0.0), None).is_err());
        assert!(check_range("x", 11.0, None, Some(10.0)).is_err());
        assert!(check_range("x", f64::NAN, None, None).is_err());
    }

    #[test]
    fn length_bounds() {
        assert!(check_length("x", 5, Some(1), Some(10)).is_ok());
        assert!(check_length("x", 0, Some(1), None).is_err());
        assert!(check_length("x", 11, None, Some(10)).is_err());
    }

    #[test]
    fn enum_membership() {
        assert!(check_enum("x", "asc", &["asc", "desc"]).is_ok());
        assert!(check_enum("x", "other", &["asc", "desc"]).is_err());
    }

    #[test]
    fn pattern_matches() {
        let re = regex::Regex::new(r"^[a-z]+$").unwrap();
        assert!(check_pattern("x", "hello", &re).is_ok());
        assert!(check_pattern("x", "Hello", &re).is_err());
    }

    #[test]
    fn format_email() {
        assert!(check_format("e", "a@b.com", Format::Email).is_ok());
        assert!(check_format("e", "not-an-email", Format::Email).is_err());
    }

    #[test]
    fn format_uuid() {
        assert!(
            check_format("u", "550e8400-e29b-41d4-a716-446655440000", Format::Uuid).is_ok()
        );
        assert!(check_format("u", "not-a-uuid", Format::Uuid).is_err());
    }

    #[test]
    fn format_ipv4_ipv6() {
        assert!(check_format("ip", "127.0.0.1", Format::Ipv4).is_ok());
        assert!(check_format("ip", "300.0.0.1", Format::Ipv4).is_err());
        assert!(check_format("ip", "::1", Format::Ipv6).is_ok());
        assert!(check_format("ip", "not-ipv6", Format::Ipv6).is_err());
    }

    #[test]
    fn format_uri_datetime_pass() {
        // schema-only formats always pass at runtime
        assert!(check_format("u", "anything", Format::Uri).is_ok());
        assert!(check_format("d", "anything", Format::DateTime).is_ok());
    }

    #[test]
    fn validate_args_default_ok() {
        struct E;
        impl ValidateArgs for E {}
        assert!(E.validate().is_ok());
    }
}
