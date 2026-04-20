//! Parser for `#[schema(...)]` field-level attributes.
//!
//! Grammar (see `docs/plans/2026-04-19-tool-attribute-macro-design.md`):
//!
//! ```text
//! schema_attr   := '#[schema(' validator (',' validator)* ')]'
//! validator     := range_v | length_v | pattern_v | enum_v | format_v | items_v
//! ```
//!
//! This module only produces the **parsed structure**. Emission into JSON
//! Schema lives in `lib.rs::type_to_schema`; runtime validator emission is
//! in the (future) `#[cognis::tool]` attribute macro.

use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    LitInt, LitStr, Result, Token,
};

/// One fully-parsed `#[schema(...)]` attribute (may hold multiple validators).
#[derive(Debug, Clone, Default)]
pub struct SchemaAttr {
    pub validators: Vec<Validator>,
}

#[derive(Debug, Clone)]
pub enum Validator {
    Range {
        min: Option<f64>,
        max: Option<f64>,
    },
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
    Pattern(String),
    EnumValues(Vec<String>),
    Format(FormatName),
    Items(Box<SchemaAttr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatName {
    Email,
    Uri,
    Uuid,
    DateTime,
    Ipv4,
    Ipv6,
}

impl FormatName {
    pub fn as_str(self) -> &'static str {
        match self {
            FormatName::Email => "email",
            FormatName::Uri => "uri",
            FormatName::Uuid => "uuid",
            FormatName::DateTime => "date-time",
            FormatName::Ipv4 => "ipv4",
            FormatName::Ipv6 => "ipv6",
        }
    }
}

impl Parse for SchemaAttr {
    fn parse(input: ParseStream) -> Result<Self> {
        let items: Punctuated<Validator, Token![,]> = Punctuated::parse_separated_nonempty(input)?;
        Ok(SchemaAttr {
            validators: items.into_iter().collect(),
        })
    }
}

impl Parse for Validator {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident: syn::Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        match ident.to_string().as_str() {
            "range" => parse_range(&content),
            "length" => parse_length(&content),
            "pattern" => {
                let s: LitStr = content.parse()?;
                // Validate at macro time — prevents runtime panics in generated code.
                regex::Regex::new(&s.value())
                    .map_err(|e| syn::Error::new(s.span(), format!("invalid regex: {e}")))?;
                Ok(Validator::Pattern(s.value()))
            }
            "enum_values" => {
                let items: Punctuated<LitStr, Token![,]> =
                    Punctuated::parse_separated_nonempty(&content)?;
                Ok(Validator::EnumValues(
                    items.into_iter().map(|s| s.value()).collect(),
                ))
            }
            "format" => {
                let s: LitStr = content.parse()?;
                let name = match s.value().as_str() {
                    "email" => FormatName::Email,
                    "uri" => FormatName::Uri,
                    "uuid" => FormatName::Uuid,
                    "date-time" => FormatName::DateTime,
                    "ipv4" => FormatName::Ipv4,
                    "ipv6" => FormatName::Ipv6,
                    other => {
                        return Err(syn::Error::new(
                            s.span(),
                            format!(
                                "unknown format `{other}`; expected one of email, uri, uuid, date-time, ipv4, ipv6"
                            ),
                        ))
                    }
                };
                Ok(Validator::Format(name))
            }
            "items" => {
                let inner: SchemaAttr = content.parse()?;
                Ok(Validator::Items(Box::new(inner)))
            }
            other => Err(syn::Error::new(
                ident.span(),
                format!("unknown schema validator `{other}`"),
            )),
        }
    }
}

fn parse_range(input: ParseStream) -> Result<Validator> {
    let mut min = None;
    let mut max = None;
    let pairs: Punctuated<(syn::Ident, f64), Token![,]> =
        Punctuated::parse_separated_nonempty_with(input, |i| {
            let k: syn::Ident = i.parse()?;
            let _: Token![=] = i.parse()?;
            let negate = i.peek(Token![-]);
            if negate {
                let _: Token![-] = i.parse()?;
            }
            let lit: syn::Lit = i.parse()?;
            let mut val: f64 = match &lit {
                syn::Lit::Int(n) => n.base10_parse::<f64>()?,
                syn::Lit::Float(f) => f.base10_parse::<f64>()?,
                _ => {
                    return Err(syn::Error::new_spanned(
                        &lit,
                        "expected numeric literal for range bound",
                    ))
                }
            };
            if negate {
                val = -val;
            }
            Ok((k, val))
        })?;
    for (k, val) in pairs {
        match k.to_string().as_str() {
            "min" => min = Some(val),
            "max" => max = Some(val),
            other => {
                return Err(syn::Error::new(
                    k.span(),
                    format!("unknown range key `{other}`; expected min or max"),
                ))
            }
        }
    }
    Ok(Validator::Range { min, max })
}

fn parse_length(input: ParseStream) -> Result<Validator> {
    let mut min = None;
    let mut max = None;
    let pairs: Punctuated<(syn::Ident, LitInt), Token![,]> =
        Punctuated::parse_separated_nonempty_with(input, |i| {
            let k: syn::Ident = i.parse()?;
            let _: Token![=] = i.parse()?;
            let v: LitInt = i.parse()?;
            Ok((k, v))
        })?;
    for (k, v) in pairs {
        let val: usize = v.base10_parse()?;
        match k.to_string().as_str() {
            "min" => min = Some(val),
            "max" => max = Some(val),
            other => {
                return Err(syn::Error::new(
                    k.span(),
                    format!("unknown length key `{other}`; expected min or max"),
                ))
            }
        }
    }
    Ok(Validator::Length { min, max })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn parse(attr: syn::Attribute) -> SchemaAttr {
        attr.parse_args::<SchemaAttr>().unwrap()
    }

    #[test]
    fn range_both_bounds() {
        let a: syn::Attribute = parse_quote!(#[schema(range(min = 1, max = 50))]);
        let r = parse(a);
        assert!(matches!(
            r.validators[0],
            Validator::Range {
                min: Some(1.0),
                max: Some(50.0)
            }
        ));
    }

    #[test]
    fn range_max_only() {
        let a: syn::Attribute = parse_quote!(#[schema(range(max = 10))]);
        let r = parse(a);
        assert!(matches!(
            r.validators[0],
            Validator::Range {
                min: None,
                max: Some(10.0)
            }
        ));
    }

    #[test]
    fn pattern_validates_regex_at_macro_time() {
        let a: syn::Attribute = parse_quote!(#[schema(pattern("[invalid"))]);
        let err = a.parse_args::<SchemaAttr>().unwrap_err();
        assert!(err.to_string().contains("invalid regex"), "got {err}");
    }

    #[test]
    fn enum_values_parses() {
        let a: syn::Attribute = parse_quote!(#[schema(enum_values("asc", "desc"))]);
        let r = parse(a);
        if let Validator::EnumValues(v) = &r.validators[0] {
            assert_eq!(v, &vec!["asc".to_string(), "desc".to_string()]);
        } else {
            panic!("expected EnumValues");
        }
    }

    #[test]
    fn format_known_name_parses() {
        let a: syn::Attribute = parse_quote!(#[schema(format("email"))]);
        let r = parse(a);
        assert!(matches!(
            r.validators[0],
            Validator::Format(FormatName::Email)
        ));
    }

    #[test]
    fn format_unknown_name_errors() {
        let a: syn::Attribute = parse_quote!(#[schema(format("unknown"))]);
        let err = a.parse_args::<SchemaAttr>().unwrap_err();
        assert!(err.to_string().contains("unknown format"), "got {err}");
    }

    #[test]
    fn length_min_max() {
        let a: syn::Attribute = parse_quote!(#[schema(length(min = 3, max = 100))]);
        let r = parse(a);
        assert!(matches!(
            r.validators[0],
            Validator::Length {
                min: Some(3),
                max: Some(100)
            }
        ));
    }

    #[test]
    fn items_nested() {
        let a: syn::Attribute = parse_quote!(#[schema(items(range(min = 0)))]);
        let r = parse(a);
        assert!(matches!(&r.validators[0], Validator::Items(_)));
    }

    #[test]
    fn multiple_validators_on_one_attr() {
        let a: syn::Attribute = parse_quote!(#[schema(length(min = 1), pattern("^[a-z]+$"))]);
        let r = parse(a);
        assert_eq!(r.validators.len(), 2);
    }

    #[test]
    fn unknown_validator_rejected() {
        let a: syn::Attribute = parse_quote!(#[schema(weird(x = 1))]);
        let err = a.parse_args::<SchemaAttr>().unwrap_err();
        assert!(
            err.to_string().contains("unknown schema validator"),
            "got {err}"
        );
    }

    #[test]
    fn range_accepts_negative_min() {
        let a: syn::Attribute = parse_quote!(#[schema(range(min = -1, max = 10))]);
        let r = a.parse_args::<SchemaAttr>().unwrap();
        assert!(
            matches!(r.validators[0], Validator::Range { min: Some(v), max: Some(10.0) } if v == -1.0)
        );
    }

    #[test]
    fn range_accepts_negative_float() {
        let a: syn::Attribute = parse_quote!(#[schema(range(min = -0.5, max = 0.5))]);
        let r = a.parse_args::<SchemaAttr>().unwrap();
        assert!(
            matches!(r.validators[0], Validator::Range { min: Some(v), max: Some(w) } if v == -0.5 && w == 0.5)
        );
    }

    #[test]
    fn range_accepts_integer_beyond_i64_max() {
        // u64::MAX overflows i64 but fits f64 (with expected precision loss —
        // the f64 representation is still the right value for JSON Schema
        // minimum/maximum). Identified by cubic as a P2 restriction.
        let a: syn::Attribute = parse_quote!(#[schema(range(min = 0, max = 18446744073709551615))]);
        let r = a.parse_args::<SchemaAttr>().unwrap();
        assert!(matches!(
            r.validators[0],
            Validator::Range { min: Some(v), max: Some(_) } if v == 0.0
        ));
    }
}
