//! Standalone derive macros for generating OpenAPI-compatible JSON schemas.
//!
//! This crate is **framework-independent** — it has zero runtime dependencies
//! and generates no code that references any external framework crate.
//!
//! The only requirement for generated code is `serde_json` in scope.
//!
//! # Usage
//!
//! ```ignore
//! use cognis_macros::JsonSchema;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(JsonSchema, Serialize, Deserialize)]
//! struct SearchFilter {
//!     /// Minimum relevance score
//!     min_score: f64,
//!     /// Categories to include
//!     categories: Vec<String>,
//!     /// Optional max results
//!     limit: Option<u32>,
//! }
//!
//! // Static schema — no instance needed
//! let schema = SearchFilter::json_schema();
//! // {"type":"object","properties":{...},"required":["min_score","categories"]}
//! ```
//!
//! # Supported types
//!
//! | Rust type | JSON Schema |
//! |-----------|-------------|
//! | `String` | `{"type": "string"}` |
//! | `f32`, `f64` | `{"type": "number"}` |
//! | `i8`..`i128`, `u8`..`u128`, `usize`, `isize` | `{"type": "integer"}` |
//! | `bool` | `{"type": "boolean"}` |
//! | `Vec<T>` | `{"type": "array", "items": <T>}` |
//! | `Option<T>` | schema of T, removed from required |
//! | `HashMap<String, V>` | `{"type": "object", "additionalProperties": <V>}` |
//! | `serde_json::Value` | `{}` (any) |
//! | Nested struct with `#[derive(JsonSchema)]` | recursive object schema |
//! | Enum with `#[derive(JsonSchema)]` | `{"type": "string", "enum": [...]}` |

mod graph_state;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Expr, Fields, Lit, Meta, Type};

// ---------------------------------------------------------------------------
// #[derive(JsonSchema)]
// ---------------------------------------------------------------------------

/// Derive macro that generates a `pub fn json_schema() -> serde_json::Value`
/// inherent method returning an OpenAPI-compatible JSON Schema.
///
/// Works on **structs** (produces `"type": "object"` with properties) and
/// **enums** (produces `"type": "string"` with enum values).
///
/// # Field-level behaviour
///
/// - Doc comments (`///`) become `"description"` in the schema.
/// - `Option<T>` fields are excluded from `"required"`.
/// - `#[serde(skip)]` fields are excluded entirely.
/// - `#[serde(rename = "new_name")]` uses the renamed key.
/// - `#[serde(default)]` removes the field from `"required"`.
/// - Nested structs that also derive `JsonSchema` produce nested object schemas.
/// - Enums with `#[serde(rename = "...")]` on variants use the renamed values.
#[proc_macro_derive(JsonSchema, attributes(schema))]
pub fn derive_json_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_json_schema_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Backward-compatible alias for `#[derive(JsonSchema)]`.
#[proc_macro_derive(ToolSchema, attributes(schema))]
pub fn derive_tool_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_json_schema_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ---------------------------------------------------------------------------
// #[derive(GraphState)]
// ---------------------------------------------------------------------------

/// Derive macro for generating graph state schemas with per-field reducers.
///
/// # Attributes
///
/// - `#[reducer(append)]` — append arrays/values
/// - `#[reducer(last_value)]` — overwrite with latest (default)
/// - `#[reducer(add)]` — add numeric values
/// - `#[reducer(merge)]` — deep-merge JSON objects
#[proc_macro_derive(GraphState, attributes(reducer))]
pub fn derive_graph_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    graph_state::derive_graph_state(input).into()
}

// =========================================================================
// Implementation
// =========================================================================

fn derive_json_schema_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => generate_struct_schema(name, &f.named),
            _ => Err(syn::Error::new_spanned(
                name,
                "JsonSchema derive for structs only supports named fields",
            )),
        },
        Data::Enum(data) => {
            let variants: Vec<String> = data
                .variants
                .iter()
                .map(|v| {
                    if let Some(renamed) = get_serde_rename(&v.attrs) {
                        renamed
                    } else {
                        v.ident.to_string()
                    }
                })
                .collect();

            let variant_literals: Vec<_> = variants.iter().map(|v| quote! { #v }).collect();

            Ok(quote! {
                impl #name {
                    /// Returns the OpenAPI-compatible JSON Schema for this enum.
                    pub fn json_schema() -> serde_json::Value {
                        serde_json::json!({
                            "type": "string",
                            "enum": [#(#variant_literals),*]
                        })
                    }
                }
            })
        }
        _ => Err(syn::Error::new_spanned(
            name,
            "JsonSchema derive only supports structs and enums",
        )),
    }
}

fn generate_struct_schema(
    name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let schema_body = generate_schema_body(fields)?;
    Ok(quote! {
        impl #name {
            /// Returns the OpenAPI-compatible JSON Schema for this struct.
            pub fn json_schema() -> serde_json::Value {
                #schema_body
            }
        }
    })
}

// =========================================================================
// Schema body generation
// =========================================================================

fn generate_schema_body(
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let mut property_inserts = Vec::new();
    let mut required_inserts = Vec::new();

    for field in fields {
        if has_serde_skip(&field.attrs) {
            continue;
        }

        let field_ident = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "expected named field"))?;

        let json_key = if let Some(renamed) = get_serde_rename(&field.attrs) {
            renamed
        } else {
            field_ident.to_string()
        };

        let description = get_doc_comment(&field.attrs);
        let has_default = has_serde_default(&field.attrs);
        let (inner_ty, is_option) = unwrap_option_type(&field.ty);
        let schema_expr = type_to_schema(inner_ty);

        let property_value = if let Some(desc) = &description {
            quote! {
                {
                    let mut __schema = #schema_expr;
                    if let Some(obj) = __schema.as_object_mut() {
                        obj.insert("description".to_string(), serde_json::Value::String(#desc.to_string()));
                    }
                    __schema
                }
            }
        } else {
            schema_expr
        };

        property_inserts.push(quote! {
            __properties.insert(#json_key.to_string(), #property_value);
        });

        if !is_option && !has_default {
            required_inserts.push(quote! {
                __required.push(serde_json::Value::String(#json_key.to_string()));
            });
        }
    }

    Ok(quote! {
        {
            let mut __properties = serde_json::Map::new();
            let mut __required: Vec<serde_json::Value> = Vec::new();

            #(#property_inserts)*
            #(#required_inserts)*

            let mut __schema = serde_json::json!({
                "type": "object",
                "properties": serde_json::Value::Object(__properties),
            });

            if !__required.is_empty() {
                __schema["required"] = serde_json::Value::Array(__required);
            }

            __schema
        }
    })
}

// =========================================================================
// Type → JSON Schema mapping (no framework references)
// =========================================================================

fn type_to_schema(ty: &Type) -> TokenStream2 {
    match ty {
        Type::Path(type_path) => {
            let segments = &type_path.path.segments;
            let last_segment = segments.last().unwrap();
            let type_name = last_segment.ident.to_string();

            match type_name.as_str() {
                "String" | "str" => quote! { serde_json::json!({"type": "string"}) },
                "f32" | "f64" => quote! { serde_json::json!({"type": "number"}) },
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" => {
                    quote! { serde_json::json!({"type": "integer"}) }
                }
                "bool" => quote! { serde_json::json!({"type": "boolean"}) },
                "Vec" => {
                    if let Some(inner) = extract_generic_arg(&last_segment.arguments) {
                        let items_schema = type_to_schema(inner);
                        quote! {
                            serde_json::json!({
                                "type": "array",
                                "items": #items_schema
                            })
                        }
                    } else {
                        quote! { serde_json::json!({"type": "array"}) }
                    }
                }
                "HashMap" | "BTreeMap" => {
                    if let Some(value_ty) = extract_second_generic_arg(&last_segment.arguments) {
                        let value_schema = type_to_schema(value_ty);
                        quote! {
                            serde_json::json!({
                                "type": "object",
                                "additionalProperties": #value_schema
                            })
                        }
                    } else {
                        quote! { serde_json::json!({"type": "object"}) }
                    }
                }
                "Value" => quote! { serde_json::json!({}) },
                "Option" => {
                    if let Some(inner) = extract_generic_arg(&last_segment.arguments) {
                        type_to_schema(inner)
                    } else {
                        quote! { serde_json::json!({}) }
                    }
                }
                // Any other type — call its json_schema() inherent method
                _ => {
                    quote! { #ty::json_schema() }
                }
            }
        }
        Type::Reference(type_ref) => type_to_schema(&type_ref.elem),
        _ => {
            quote! { #ty::json_schema() }
        }
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn extract_generic_arg(args: &syn::PathArguments) -> Option<&Type> {
    match args {
        syn::PathArguments::AngleBracketed(ab) => ab.args.iter().find_map(|arg| match arg {
            syn::GenericArgument::Type(ty) => Some(ty),
            _ => None,
        }),
        _ => None,
    }
}

fn extract_second_generic_arg(args: &syn::PathArguments) -> Option<&Type> {
    match args {
        syn::PathArguments::AngleBracketed(ab) => {
            let mut types = ab.args.iter().filter_map(|arg| match arg {
                syn::GenericArgument::Type(ty) => Some(ty),
                _ => None,
            });
            types.next();
            types.next()
        }
        _ => None,
    }
}

fn get_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            match &attr.meta {
                Meta::NameValue(nv) => {
                    if let Expr::Lit(expr_lit) = &nv.value {
                        if let Lit::Str(s) = &expr_lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                    }
                    None
                }
                _ => None,
            }
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join(" "))
    }
}

fn has_serde_skip(attrs: &[Attribute]) -> bool {
    has_serde_attr(attrs, "skip")
}

fn has_serde_default(attrs: &[Attribute]) -> bool {
    has_serde_attr(attrs, "default")
}

fn has_serde_attr(attrs: &[Attribute], attr_name: &str) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let mut found = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(attr_name) {
                found = true;
            }
            Ok(())
        });
        if found {
            return true;
        }
    }
    false
}

fn get_serde_rename(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let mut rename_val: Option<String> = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                let value = meta.value()?;
                let s: Lit = value.parse()?;
                if let Lit::Str(lit) = s {
                    rename_val = Some(lit.value());
                }
            }
            Ok(())
        });
        if rename_val.is_some() {
            return rename_val;
        }
    }
    None
}

fn unwrap_option_type(ty: &Type) -> (&Type, bool) {
    if let Type::Path(type_path) = ty {
        if let Some(last) = type_path.path.segments.last() {
            if last.ident == "Option" {
                if let Some(inner) = extract_generic_arg(&last.arguments) {
                    return (inner, true);
                }
            }
        }
    }
    (ty, false)
}

#[cfg(test)]
mod tests {
    fn to_snake_case(s: &str) -> String {
        let mut result = String::new();
        for (i, ch) in s.chars().enumerate() {
            if ch.is_uppercase() {
                if i > 0 {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap());
            } else {
                result.push(ch);
            }
        }
        result
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("CalculatorTool"), "calculator_tool");
        assert_eq!(to_snake_case("Search"), "search");
    }
}
