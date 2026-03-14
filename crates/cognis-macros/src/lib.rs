//! Standalone derive macros for generating OpenAPI-compatible JSON schemas.
//!
//! This crate is **framework-independent** — it has zero runtime dependencies
//! beyond `syn`/`quote`/`proc-macro2` (compile-time only).
//!
//! ## Standalone schema generation
//!
//! `#[derive(JsonSchema)]` generates an implementation of the `JsonSchema` trait
//! that returns a `serde_json::Value` describing the type in OpenAPI format.
//! Works on structs and enums. No framework dependency required.
//!
//! ```ignore
//! use cognis_macros::JsonSchema;
//!
//! #[derive(JsonSchema, serde::Serialize, serde::Deserialize)]
//! struct SearchFilter {
//!     /// Minimum relevance score
//!     min_score: f64,
//!     /// Categories to include
//!     categories: Vec<String>,
//! }
//!
//! let schema = SearchFilter::json_schema();
//! // {"type":"object","properties":{"min_score":{"type":"number","description":"Minimum relevance score"},...},"required":["min_score","categories"]}
//! ```
//!
//! ## Framework integration
//!
//! `#[derive(Tool)]` generates both a `JsonSchema` impl AND a framework-specific
//! `BaseTool` impl. The framework crate path defaults to `cognis_core` but can be
//! overridden with `#[tool(crate_path = "my_framework")]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Expr, Fields, Lit, Meta, Type};

// =========================================================================
// The JsonSchema trait — defined here so the macro is fully standalone.
// Users only need `cognis-macros` + `serde_json` in scope.
// =========================================================================

// NOTE: The trait is not defined as Rust code in this proc-macro crate (proc-macro
// crates can only export procedural macros). Instead, the derive macro generates
// an impl for a trait `cognis_macros::JsonSchema` which is defined via a
// re-export trick: the trait definition lives in the generated code itself using
// a well-known path. We use a standalone trait path that can be configured.

// ---------------------------------------------------------------------------
// #[derive(JsonSchema)] — standalone, no framework dependency
// ---------------------------------------------------------------------------

/// Derive macro that generates a `json_schema()` associated function returning
/// an OpenAPI-compatible JSON Schema as `serde_json::Value`.
///
/// Works on **structs** (generates `"type": "object"` with properties) and
/// **enums** (generates `"type": "string"` with enum values).
///
/// This macro is **standalone** — it does not depend on any LLM framework.
/// The only runtime dependency is `serde_json`.
///
/// # Struct-level attributes
///
/// - `#[schema(description = "...")]` — Override the struct description.
///
/// # Field-level behaviour
///
/// - Doc comments (`///`) become `"description"` in the JSON schema.
/// - `Option<T>` fields are excluded from `"required"`.
/// - `#[serde(skip)]` fields are excluded from the schema entirely.
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

// ---------------------------------------------------------------------------
// #[derive(Tool)] — framework integration (generates BaseTool + JsonSchema)
// ---------------------------------------------------------------------------

/// Derive macro that generates both a `JsonSchema` impl and a framework-specific
/// `BaseTool` impl for the struct.
///
/// # Struct-level attributes
///
/// - `#[tool(name = "my_tool")]` — Override the tool name (defaults to snake_case).
/// - `#[tool(description = "...")]` — Override the description (defaults to doc comment).
/// - `#[tool(crate_path = "my_crate")]` — Override the framework crate path
///   (defaults to `cognis_core`).
///
/// The struct must implement `async fn execute(&self) -> Result<ToolOutput>`.
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_tool_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Kept for backward compatibility — alias for `#[derive(JsonSchema)]`.
#[proc_macro_derive(ToolSchema, attributes(tool, schema))]
pub fn derive_tool_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_json_schema_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// =========================================================================
// Implementation: #[derive(JsonSchema)]
// =========================================================================

fn derive_json_schema_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => generate_json_schema_impl(name, &f.named),
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
                    /// Returns the OpenAPI-compatible JSON Schema for this type.
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

fn generate_json_schema_impl(
    name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let schema_body = generate_schema_body(fields)?;
    Ok(quote! {
        impl #name {
            /// Returns the OpenAPI-compatible JSON Schema for this type.
            pub fn json_schema() -> serde_json::Value {
                #schema_body
            }
        }
    })
}

// =========================================================================
// Implementation: #[derive(Tool)]
// =========================================================================

fn derive_tool_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "Tool derive only supports structs with named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Tool derive only supports structs",
            ))
        }
    };

    let (tool_name, tool_desc, crate_path) = parse_tool_attrs(&input.attrs, name)?;
    let json_schema_impl = generate_json_schema_impl(name, fields)?;
    let schema_body = generate_schema_body(fields)?;

    Ok(quote! {
        // Standalone json_schema() — no framework dependency
        #json_schema_impl

        // Framework-specific BaseTool impl
        #[async_trait::async_trait]
        impl #crate_path::tools::BaseTool for #name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #tool_desc
            }

            fn args_schema(&self) -> Option<serde_json::Value> {
                Some(Self::json_schema())
            }

            async fn _run(&self, _input: #crate_path::tools::ToolInput) -> #crate_path::error::Result<#crate_path::tools::ToolOutput> {
                self.execute().await
            }
        }

        // Framework ToolJsonSchema bridge — delegates to standalone json_schema()
        impl #crate_path::tools::ToolJsonSchema for #name {
            fn json_schema() -> serde_json::Value {
                <#name>::json_schema()
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
// Type → JSON Schema mapping
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
// Helper: extract generic type arguments
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

// =========================================================================
// Attribute parsing helpers
// =========================================================================

fn parse_tool_attrs(
    attrs: &[Attribute],
    struct_name: &syn::Ident,
) -> syn::Result<(TokenStream2, TokenStream2, TokenStream2)> {
    let mut tool_name: Option<String> = None;
    let mut tool_desc: Option<String> = None;
    let mut crate_path_str: Option<String> = None;

    for attr in attrs {
        if attr.path().is_ident("tool") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let value = meta.value()?;
                    let s: Lit = value.parse()?;
                    if let Lit::Str(lit) = s {
                        tool_name = Some(lit.value());
                    }
                    Ok(())
                } else if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let s: Lit = value.parse()?;
                    if let Lit::Str(lit) = s {
                        tool_desc = Some(lit.value());
                    }
                    Ok(())
                } else if meta.path.is_ident("crate_path") {
                    let value = meta.value()?;
                    let s: Lit = value.parse()?;
                    if let Lit::Str(lit) = s {
                        crate_path_str = Some(lit.value());
                    }
                    Ok(())
                } else {
                    Err(meta.error("expected `name`, `description`, or `crate_path`"))
                }
            })?;
        }
    }

    let name_str = tool_name.unwrap_or_else(|| to_snake_case(&struct_name.to_string()));
    let desc_str = tool_desc
        .unwrap_or_else(|| get_doc_comment(attrs).unwrap_or_else(|| format!("Tool: {}", name_str)));

    let crate_path: TokenStream2 = if let Some(path) = crate_path_str {
        let ident = syn::parse_str::<syn::Path>(&path).map_err(|e| {
            syn::Error::new_spanned(struct_name, format!("invalid crate_path: {e}"))
        })?;
        quote! { #ident }
    } else {
        quote! { cognis_core }
    };

    Ok((quote! { #name_str }, quote! { #desc_str }, crate_path))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("CalculatorTool"), "calculator_tool");
        assert_eq!(to_snake_case("Search"), "search");
        assert_eq!(to_snake_case("MyAPITool"), "my_a_p_i_tool");
        assert_eq!(to_snake_case("simple"), "simple");
    }
}
