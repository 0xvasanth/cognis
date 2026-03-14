//! Derive macros for the Cognis LLM framework.
//!
//! Provides `#[derive(Tool)]` for auto-generating `BaseTool` implementations
//! with OpenAPI-compatible JSON schemas, and `#[derive(ToolSchema)]` for
//! generating `ToolJsonSchema` implementations for nested structs and enums.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Expr, Fields, Lit, Meta, Type};

// ---------------------------------------------------------------------------
// #[derive(Tool)]
// ---------------------------------------------------------------------------

/// Derive macro that generates `BaseTool` and `ToolJsonSchema` implementations
/// for a struct.
///
/// # Struct-level attributes
///
/// - `#[tool(name = "my_tool")]` — Override the tool name (defaults to snake_case
///   of the struct name).
/// - `#[tool(description = "...")]` — Override the description (defaults to the
///   struct's doc comment).
///
/// # Field-level behaviour
///
/// - Doc comments become the `"description"` in the JSON schema.
/// - `Option<T>` fields are excluded from `"required"`.
/// - `#[serde(skip)]` fields are excluded from the schema entirely.
/// - `#[serde(rename = "new_name")]` uses the renamed key.
/// - `#[serde(default)]` removes the field from `"required"`.
///
/// The struct must also implement an `async fn execute(&self) -> cognis_core::error::Result<cognis_core::tools::ToolOutput>`
/// method that contains the actual tool logic.
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_tool_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Derive macro that generates `ToolJsonSchema` for structs and enums.
///
/// For enums, generates `{"type": "string", "enum": [...]}` using variant names.
/// Respects `#[serde(rename = "...")]` on variants.
///
/// For structs, generates the same schema as `#[derive(Tool)]` but only the
/// `ToolJsonSchema` trait (not `BaseTool`).
#[proc_macro_derive(ToolSchema, attributes(tool))]
pub fn derive_tool_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_tool_schema_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ---------------------------------------------------------------------------
// Core implementation for #[derive(Tool)]
// ---------------------------------------------------------------------------

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

    let (tool_name, tool_desc) = parse_tool_attrs(&input.attrs, name)?;

    let schema_body = generate_schema_body(fields)?;
    let tool_schema_impl = generate_tool_json_schema_impl(name, fields)?;

    Ok(quote! {
        #[async_trait::async_trait]
        impl cognis_core::tools::BaseTool for #name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #tool_desc
            }

            fn args_schema(&self) -> Option<serde_json::Value> {
                Some(#schema_body)
            }

            async fn _run(&self, input: cognis_core::tools::ToolInput) -> cognis_core::error::Result<cognis_core::tools::ToolOutput> {
                self.execute().await
            }
        }

        #tool_schema_impl
    })
}

// ---------------------------------------------------------------------------
// Core implementation for #[derive(ToolSchema)]
// ---------------------------------------------------------------------------

fn derive_tool_schema_impl(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => generate_tool_json_schema_impl(name, &f.named),
            _ => Err(syn::Error::new_spanned(
                name,
                "ToolSchema derive for structs only supports named fields",
            )),
        },
        Data::Enum(data) => {
            let variants: Vec<String> = data
                .variants
                .iter()
                .map(|v| {
                    // Check for #[serde(rename = "...")]
                    if let Some(renamed) = get_serde_rename(&v.attrs) {
                        renamed
                    } else {
                        v.ident.to_string()
                    }
                })
                .collect();

            let variant_literals: Vec<_> = variants.iter().map(|v| quote! { #v }).collect();

            Ok(quote! {
                impl cognis_core::tools::ToolJsonSchema for #name {
                    fn json_schema() -> serde_json::Value {
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
            "ToolSchema derive only supports structs and enums",
        )),
    }
}

// ---------------------------------------------------------------------------
// Schema body generation (shared by Tool and ToolSchema for structs)
// ---------------------------------------------------------------------------

fn generate_tool_json_schema_impl(
    name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let schema_body = generate_schema_body(fields)?;
    Ok(quote! {
        impl cognis_core::tools::ToolJsonSchema for #name {
            fn json_schema() -> serde_json::Value {
                #schema_body
            }
        }
    })
}

fn generate_schema_body(
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let mut property_inserts = Vec::new();
    let mut required_inserts = Vec::new();

    for field in fields {
        // Skip fields with #[serde(skip)]
        if has_serde_skip(&field.attrs) {
            continue;
        }

        let field_ident = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "expected named field"))?;

        // Determine the JSON key name
        let json_key = if let Some(renamed) = get_serde_rename(&field.attrs) {
            renamed
        } else {
            field_ident.to_string()
        };

        // Extract doc comment
        let description = get_doc_comment(&field.attrs);

        // Determine if the field is optional or has a default
        let has_default = has_serde_default(&field.attrs);
        let (inner_ty, is_option) = unwrap_option_type(&field.ty);

        // Generate the schema for this field's type
        let schema_expr = type_to_schema(inner_ty);

        // Add description if present
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

        // Add to required if not Option and not serde(default)
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

// ---------------------------------------------------------------------------
// Type → JSON Schema mapping
// ---------------------------------------------------------------------------

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
                    // Extract the value type (second generic arg)
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
                    // Shouldn't normally reach here since we unwrap Option in the caller,
                    // but handle gracefully.
                    if let Some(inner) = extract_generic_arg(&last_segment.arguments) {
                        type_to_schema(inner)
                    } else {
                        quote! { serde_json::json!({}) }
                    }
                }
                // Any other type — assume it implements ToolJsonSchema
                _ => {
                    let ty_tokens = quote! { #ty };
                    quote! {
                        <#ty_tokens as cognis_core::tools::ToolJsonSchema>::json_schema()
                    }
                }
            }
        }
        Type::Reference(type_ref) => {
            // Handle &str etc.
            type_to_schema(&type_ref.elem)
        }
        _ => {
            // Fallback: delegate to ToolJsonSchema
            quote! {
                <#ty as cognis_core::tools::ToolJsonSchema>::json_schema()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract generic type arguments
// ---------------------------------------------------------------------------

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
            types.next(); // skip first (key type)
            types.next() // second (value type)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Attribute parsing helpers
// ---------------------------------------------------------------------------

/// Parse `#[tool(name = "...", description = "...")]` and doc comments on the struct.
fn parse_tool_attrs(
    attrs: &[Attribute],
    struct_name: &syn::Ident,
) -> syn::Result<(TokenStream2, TokenStream2)> {
    let mut tool_name: Option<String> = None;
    let mut tool_desc: Option<String> = None;

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
                } else {
                    Err(meta.error("expected `name` or `description`"))
                }
            })?;
        }
    }

    let name_str = tool_name.unwrap_or_else(|| to_snake_case(&struct_name.to_string()));
    let desc_str = tool_desc
        .unwrap_or_else(|| get_doc_comment(attrs).unwrap_or_else(|| format!("Tool: {}", name_str)));

    Ok((quote! { #name_str }, quote! { #desc_str }))
}

/// Extract doc comment text from attributes.
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

/// Check if a field has `#[serde(skip)]`.
fn has_serde_skip(attrs: &[Attribute]) -> bool {
    has_serde_attr(attrs, "skip")
}

/// Check if a field has `#[serde(default)]`.
fn has_serde_default(attrs: &[Attribute]) -> bool {
    has_serde_attr(attrs, "default")
}

/// Generic helper to check for a bare `#[serde(attr_name)]`.
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

/// Extract `#[serde(rename = "new_name")]` value.
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

/// Unwrap `Option<T>` → `(T, true)`, otherwise `(ty, false)`.
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

/// Convert PascalCase to snake_case.
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
