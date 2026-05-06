//! `#[cognis::tool]` attribute macro.
//!
//! Wraps an `async fn` (either standalone or inside an `impl` block) to
//! generate a `BaseTool` implementation with typed, schema-validated args.
//!
//! Two forms are supported:
//!
//! - **Standalone** — `#[tool]` on a free `async fn`. The macro emits a
//!   unit struct with the fn's PascalCase name; that struct is the
//!   `BaseTool` implementation.
//! - **Impl-block** — `#[tool]` on an `impl` block containing exactly one
//!   `async fn`. The macro preserves the original inherent method and
//!   emits a separate `BaseTool` impl whose `_run` dispatches into that
//!   method (letting tools hold state like HTTP clients or API keys).
//!
//! # Schema generation
//!
//! The generated args struct derives `schemars::JsonSchema`. The
//! `args_schema()` method calls `schemars::schema_for!(...)` and then
//! post-processes the resulting JSON to inject keywords driven by
//! `#[schema(...)]` validators (minimum/maximum/minLength/maxLength/
//! pattern/enum/format). This keeps the schemars derive vanilla while
//! preserving the rich validator surface the rest of the framework
//! depends on.
//!
//! Generated code references `cognis_core` and `schemars` by absolute
//! path, so callers only need `cognis_core` and `async_trait` in scope.
//! Regex compilation for `#[schema(pattern(...))]` goes through
//! `cognis_core::tools::validation::__regex` — users do not need to add
//! `regex` to their `Cargo.toml`.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    Expr, FnArg, ImplItem, ItemFn, ItemImpl, Lit, LitStr, Meta, Pat, Signature, Token, Type,
    Visibility,
};

use crate::schema_attr::{self, Validator};

/// Parsed arguments of the `#[tool(...)]` attribute.
#[derive(Default)]
pub(crate) struct ToolArgs {
    pub name: Option<String>,
    pub description: Option<String>,
    #[allow(dead_code)]
    pub return_direct: Option<bool>,
    /// Crate path used in generated code, e.g. `"cognis_core"` (default,
    /// for v1) or `"cognis2_core"` (for v2). The macro emits all framework
    /// type references as `::<crate_path>::tools::*` etc.
    pub crate_path: String,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ToolArgs {
            crate_path: "cognis_core".to_string(),
            ..ToolArgs::default()
        };
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "name" => args.name = Some(input.parse::<LitStr>()?.value()),
                "description" => args.description = Some(input.parse::<LitStr>()?.value()),
                "return_direct" => {
                    let b: syn::LitBool = input.parse()?;
                    args.return_direct = Some(b.value);
                }
                "crate_path" => args.crate_path = input.parse::<LitStr>()?.value(),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[tool] argument `{other}`; expected name, description, return_direct, or crate_path"
                        ),
                    ))
                }
            }
            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
            }
        }
        Ok(args)
    }
}

/// Entry point — dispatches to the correct form based on what `input`
/// parses as.
pub(crate) fn expand(args: ToolArgs, input: TokenStream2) -> syn::Result<TokenStream2> {
    if let Ok(item_fn) = syn::parse2::<ItemFn>(input.clone()) {
        return expand_fn(args, item_fn);
    }
    if let Ok(item_impl) = syn::parse2::<ItemImpl>(input) {
        return expand_impl(args, item_impl);
    }
    Err(syn::Error::new(
        Span::call_site(),
        "#[tool] can only be applied to an async fn or to an impl block containing exactly one async fn",
    ))
}

// ---------------------------------------------------------------------------
// Standalone `async fn` form
// ---------------------------------------------------------------------------

fn expand_fn(args: ToolArgs, item_fn: ItemFn) -> syn::Result<TokenStream2> {
    if item_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig,
            "#[tool] requires an `async fn` — tools are invoked asynchronously",
        ));
    }
    if item_fn.sig.generics.params.iter().next().is_some() {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.generics,
            "#[tool] does not support generic functions in v1",
        ));
    }

    let vis = &item_fn.vis;
    let fn_ident = &item_fn.sig.ident;
    let fn_doc = collect_doc_comment(&item_fn.attrs);

    let arg_specs = parse_typed_args(&item_fn.sig, /* allow_self = */ false)?;

    let struct_ident = pascal_case_ident(&fn_ident.to_string(), fn_ident.span());
    let args_struct_ident = format_ident!("__{}Args", pascal_case(&fn_ident.to_string()));
    let body_fn_ident = format_ident!("__{}_body", fn_ident);

    let tool_name = args.name.clone().unwrap_or_else(|| fn_ident.to_string());
    let description = args.description.clone().or(fn_doc).unwrap_or_default();
    let return_direct = args.return_direct.unwrap_or(false);

    let args_struct = emit_args_struct(&args_struct_ident, &arg_specs);
    let validate_impl = emit_validate_impl(&args_struct_ident, &arg_specs)?;
    let body_fn = emit_body_fn(&body_fn_ident, &item_fn)?;
    let base_tool_impl = emit_base_tool_impl_standalone(
        vis,
        &struct_ident,
        &args_struct_ident,
        &body_fn_ident,
        &tool_name,
        &description,
        return_direct,
        &arg_specs,
    );

    Ok(quote! {
        #args_struct
        #validate_impl
        #body_fn
        #vis struct #struct_ident;
        #base_tool_impl
    })
}

// ---------------------------------------------------------------------------
// Impl-block form
// ---------------------------------------------------------------------------

fn expand_impl(args: ToolArgs, item_impl: ItemImpl) -> syn::Result<TokenStream2> {
    if item_impl.generics.params.iter().next().is_some() {
        return Err(syn::Error::new_spanned(
            &item_impl.generics,
            "#[tool] does not support generic impl blocks in v1",
        ));
    }
    if let Some((_, path, _)) = &item_impl.trait_ {
        return Err(syn::Error::new_spanned(
            path,
            "#[tool] must be applied to an inherent impl block (not a trait impl)",
        ));
    }

    let mut async_methods = Vec::new();
    for item in &item_impl.items {
        if let ImplItem::Fn(m) = item {
            if m.sig.asyncness.is_some() {
                async_methods.push(m);
            }
        }
    }
    if async_methods.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_impl,
            "#[tool] impl block must contain exactly one `async fn` (the tool body); found none",
        ));
    }
    if async_methods.len() > 1 {
        return Err(syn::Error::new_spanned(
            async_methods[1],
            "#[tool] impl block must contain exactly one `async fn`; found multiple — \
             split the extra async methods into a separate `impl` block",
        ));
    }
    let method = async_methods[0];

    let self_ty = &*item_impl.self_ty;
    let struct_path = quote! { #self_ty };
    let struct_ident = match self_ty {
        Type::Path(tp) => tp
            .path
            .segments
            .last()
            .map(|s| s.ident.clone())
            .ok_or_else(|| syn::Error::new_spanned(self_ty, "cannot resolve impl target name"))?,
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "#[tool] impl target must be a named struct type",
            ))
        }
    };

    let receiver = method.sig.receiver().ok_or_else(|| {
        syn::Error::new_spanned(
            &method.sig,
            "#[tool] method must take `&self` as its first argument",
        )
    })?;
    if receiver.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            receiver,
            "#[tool] method receiver must be `&self` (not `&mut self`)",
        ));
    }
    if receiver.reference.is_none() {
        return Err(syn::Error::new_spanned(
            receiver,
            "#[tool] method receiver must be `&self` (consuming `self` is not supported)",
        ));
    }

    let method_ident = &method.sig.ident;
    let method_doc = collect_doc_comment(&method.attrs);
    let arg_specs = parse_typed_args(&method.sig, /* allow_self = */ true)?;

    let args_struct_ident = format_ident!(
        "__{}{}Args",
        pascal_case(&struct_ident.to_string()),
        pascal_case(&method_ident.to_string())
    );

    let tool_name = args
        .name
        .clone()
        .unwrap_or_else(|| method_ident.to_string());
    let description = args.description.clone().or(method_doc).unwrap_or_default();
    let return_direct = args.return_direct.unwrap_or(false);

    let args_struct = emit_args_struct(&args_struct_ident, &arg_specs);
    let validate_impl = emit_validate_impl(&args_struct_ident, &arg_specs)?;
    let base_tool_impl = emit_base_tool_impl_method(
        &struct_path,
        &args_struct_ident,
        method_ident,
        &tool_name,
        &description,
        return_direct,
        &arg_specs,
    );

    // Strip helper attributes from the preserved impl block's method
    // parameters. `#[schema(...)]` and `///` doc comments were already
    // captured into `arg_specs`; leaving them on the regenerated fn
    // parameters would be a hard error on stable Rust.
    let mut cleaned_impl = item_impl.clone();
    for item in cleaned_impl.items.iter_mut() {
        if let ImplItem::Fn(m) = item {
            if m.sig.asyncness.is_some() {
                for input in m.sig.inputs.iter_mut() {
                    if let FnArg::Typed(pt) = input {
                        pt.attrs
                            .retain(|a| !a.path().is_ident("schema") && !a.path().is_ident("doc"));
                    }
                }
            }
        }
    }

    Ok(quote! {
        #args_struct
        #validate_impl
        #cleaned_impl
        #base_tool_impl
    })
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

/// A parsed tool argument — one positional fn parameter that becomes a
/// field in the generated args struct.
struct ArgSpec {
    /// Original field identifier (preserved verbatim for serde).
    ident: syn::Ident,
    /// Full type path (`Option<T>` preserved).
    ty: Type,
    /// Inner type with `Option` stripped (used for type-dispatch in
    /// validator emission).
    inner_ty: Type,
    /// `true` iff the outer type was `Option<T>`.
    is_option: bool,
    /// Doc comment (rustdoc) attached to the argument, passed through to
    /// the generated struct so `schemars` picks it up as `description`.
    docs: Vec<syn::Attribute>,
    /// Parsed validators (accumulated across all `#[schema(...)]` attrs).
    validators: Vec<Validator>,
}

fn parse_typed_args(sig: &Signature, allow_self: bool) -> syn::Result<Vec<ArgSpec>> {
    let mut specs = Vec::new();
    for input in &sig.inputs {
        match input {
            FnArg::Receiver(r) => {
                if !allow_self {
                    return Err(syn::Error::new_spanned(
                        r,
                        "standalone #[tool] functions cannot take `self` — use the impl-block form instead",
                    ));
                }
            }
            FnArg::Typed(pat_type) => {
                let ident = match &*pat_type.pat {
                    Pat::Ident(pi) => pi.ident.clone(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "#[tool] args must be plain identifiers (no destructuring)",
                        ))
                    }
                };
                if let Type::Reference(tr) = &*pat_type.ty {
                    return Err(syn::Error::new_spanned(
                        tr,
                        "#[tool] args must be owned types (e.g. `String`, not `&str`) — \
                         the macro deserializes via `serde_json::from_value`",
                    ));
                }
                let (inner_ty, is_option) = unwrap_option(&pat_type.ty);
                let docs: Vec<syn::Attribute> = pat_type
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("doc"))
                    .cloned()
                    .collect();

                let mut validators = Vec::new();
                for attr in pat_type
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("schema"))
                {
                    let parsed = attr.parse_args::<schema_attr::SchemaAttr>()?;
                    validators.extend(parsed.validators);
                }

                specs.push(ArgSpec {
                    ident,
                    ty: (*pat_type.ty).clone(),
                    inner_ty: inner_ty.clone(),
                    is_option,
                    docs,
                    validators,
                });
            }
        }
    }
    Ok(specs)
}

fn unwrap_option(ty: &Type) -> (Type, bool) {
    if let Type::Path(tp) = ty {
        if let Some(last) = tp.path.segments.last() {
            if last.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(ab) = &last.arguments {
                    for arg in &ab.args {
                        if let syn::GenericArgument::Type(t) = arg {
                            return (t.clone(), true);
                        }
                    }
                }
            }
        }
    }
    (ty.clone(), false)
}

// ---------------------------------------------------------------------------
// Code emission
// ---------------------------------------------------------------------------

fn emit_args_struct(struct_ident: &syn::Ident, specs: &[ArgSpec]) -> TokenStream2 {
    let fields = specs.iter().map(|s| {
        let ident = &s.ident;
        let ty = &s.ty;
        let docs = &s.docs;
        quote! {
            #(#docs)*
            pub #ident: #ty,
        }
    });
    quote! {
        #[derive(::serde::Deserialize, ::cognis_core::schemars::JsonSchema)]
        #[schemars(crate = "::cognis_core::schemars")]
        #[allow(non_camel_case_types, non_snake_case, dead_code)]
        struct #struct_ident {
            #(#fields)*
        }
    }
}

fn emit_validate_impl(struct_ident: &syn::Ident, specs: &[ArgSpec]) -> syn::Result<TokenStream2> {
    let mut pattern_statics = Vec::new();
    let mut validator_stmts = Vec::new();

    for spec in specs {
        let ident = &spec.ident;
        let field_name = ident.to_string();
        let mut checks = Vec::new();
        let type_kind = classify_type(&spec.inner_ty);

        for (i, v) in spec.validators.iter().enumerate() {
            match v {
                Validator::Range { min, max } => {
                    let min_tok = option_f64(min);
                    let max_tok = option_f64(max);
                    checks.push(quote! {
                        ::cognis_core::tools::validation::check_range(
                            #field_name,
                            (*__v) as f64,
                            #min_tok,
                            #max_tok,
                        )?;
                    });
                }
                Validator::Length { min, max } => {
                    let min_tok = option_usize(min);
                    let max_tok = option_usize(max);
                    let len_expr = match type_kind {
                        TypeKind::String => {
                            quote! { ::core::primitive::str::chars(__v.as_str()).count() }
                        }
                        TypeKind::Vec | TypeKind::Other => quote! { __v.len() },
                    };
                    checks.push(quote! {
                        ::cognis_core::tools::validation::check_length(
                            #field_name,
                            #len_expr,
                            #min_tok,
                            #max_tok,
                        )?;
                    });
                }
                Validator::Pattern(p) => {
                    let static_ident = format_ident!(
                        "__{}_{}_PATTERN_{}",
                        struct_ident.to_string().to_uppercase(),
                        field_name.to_uppercase(),
                        i
                    );
                    let accessor_ident = format_ident!(
                        "__{}_{}_pattern_{}",
                        struct_ident.to_string().to_lowercase(),
                        field_name.to_lowercase(),
                        i
                    );
                    let pat_lit = p.as_str();
                    pattern_statics.push(quote! {
                        static #static_ident:
                            ::std::sync::OnceLock<
                                ::cognis_core::tools::validation::__regex::Regex,
                            > = ::std::sync::OnceLock::new();
                        #[allow(non_snake_case)]
                        fn #accessor_ident()
                            -> &'static ::cognis_core::tools::validation::__regex::Regex
                        {
                            #static_ident.get_or_init(|| {
                                ::cognis_core::tools::validation::__regex::Regex::new(#pat_lit)
                                    .expect("regex validated at macro time")
                            })
                        }
                    });
                    checks.push(quote! {
                        ::cognis_core::tools::validation::check_pattern(
                            #field_name,
                            __v.as_str(),
                            #accessor_ident(),
                        )?;
                    });
                }
                Validator::EnumValues(values) => {
                    let values_tok = values.iter().map(|s| quote! { #s });
                    checks.push(quote! {
                        ::cognis_core::tools::validation::check_enum(
                            #field_name,
                            __v.as_str(),
                            &[#(#values_tok),*],
                        )?;
                    });
                }
                Validator::Format(fmt) => {
                    let fmt_variant = match fmt {
                        schema_attr::FormatName::Email => quote! { Email },
                        schema_attr::FormatName::Uri => quote! { Uri },
                        schema_attr::FormatName::Uuid => quote! { Uuid },
                        schema_attr::FormatName::DateTime => quote! { DateTime },
                        schema_attr::FormatName::Ipv4 => quote! { Ipv4 },
                        schema_attr::FormatName::Ipv6 => quote! { Ipv6 },
                    };
                    checks.push(quote! {
                        ::cognis_core::tools::validation::check_format(
                            #field_name,
                            __v.as_str(),
                            ::cognis_core::tools::validation::Format::#fmt_variant,
                        )?;
                    });
                }
                Validator::Items(_) => {
                    // Items-nested validators only contribute to the JSON
                    // schema for now — runtime iteration over Vec items
                    // is not implemented in v1.
                }
            }
        }

        if checks.is_empty() {
            continue;
        }

        if spec.is_option {
            validator_stmts.push(quote! {
                if let Some(ref __v) = self.#ident {
                    #(#checks)*
                }
            });
        } else {
            validator_stmts.push(quote! {
                {
                    let __v = &self.#ident;
                    #(#checks)*
                }
            });
        }
    }

    Ok(quote! {
        #(#pattern_statics)*
        impl ::cognis_core::tools::validation::ValidateArgs for #struct_ident {
            fn validate(&self) -> ::cognis_core::error::Result<()> {
                #(#validator_stmts)*
                Ok(())
            }
        }
    })
}

fn emit_body_fn(body_fn_ident: &syn::Ident, item_fn: &ItemFn) -> syn::Result<TokenStream2> {
    let vis = &item_fn.vis;
    let mut sig = item_fn.sig.clone();
    sig.ident = body_fn_ident.clone();
    for input in sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = input {
            pt.attrs.clear();
        }
    }
    let block = &item_fn.block;
    Ok(quote! {
        #[allow(non_snake_case, dead_code)]
        #vis #sig #block
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_base_tool_impl_standalone(
    _vis: &Visibility,
    struct_ident: &syn::Ident,
    args_struct_ident: &syn::Ident,
    body_fn_ident: &syn::Ident,
    tool_name: &str,
    description: &str,
    return_direct: bool,
    specs: &[ArgSpec],
) -> TokenStream2 {
    let field_idents: Vec<_> = specs.iter().map(|s| &s.ident).collect();
    let return_direct_method = if return_direct {
        Some(quote! {
            fn return_direct(&self) -> bool { true }
        })
    } else {
        None
    };
    let schema_body = emit_args_schema_body(args_struct_ident, specs);

    quote! {
        #[::async_trait::async_trait]
        impl ::cognis_core::tools::BaseTool for #struct_ident {
            fn name(&self) -> &str { #tool_name }
            fn description(&self) -> &str { #description }
            fn args_schema(&self) -> ::core::option::Option<::serde_json::Value> {
                #schema_body
            }
            #return_direct_method
            async fn _run(
                &self,
                input: ::cognis_core::tools::ToolInput,
            ) -> ::cognis_core::error::Result<::cognis_core::tools::ToolOutput> {
                let __json = input.into_json();
                let __args: #args_struct_ident = ::serde_json::from_value(__json)
                    .map_err(|e| ::cognis_core::error::CognisError::ToolValidationError(
                        e.to_string(),
                    ))?;
                <#args_struct_ident as ::cognis_core::tools::validation::ValidateArgs>::validate(&__args)?;
                #body_fn_ident(#(__args.#field_idents),*).await
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_base_tool_impl_method(
    struct_path: &TokenStream2,
    args_struct_ident: &syn::Ident,
    method_ident: &syn::Ident,
    tool_name: &str,
    description: &str,
    return_direct: bool,
    specs: &[ArgSpec],
) -> TokenStream2 {
    let field_idents: Vec<_> = specs.iter().map(|s| &s.ident).collect();
    let return_direct_method = if return_direct {
        Some(quote! {
            fn return_direct(&self) -> bool { true }
        })
    } else {
        None
    };
    let schema_body = emit_args_schema_body(args_struct_ident, specs);

    quote! {
        #[::async_trait::async_trait]
        impl ::cognis_core::tools::BaseTool for #struct_path {
            fn name(&self) -> &str { #tool_name }
            fn description(&self) -> &str { #description }
            fn args_schema(&self) -> ::core::option::Option<::serde_json::Value> {
                #schema_body
            }
            #return_direct_method
            async fn _run(
                &self,
                input: ::cognis_core::tools::ToolInput,
            ) -> ::cognis_core::error::Result<::cognis_core::tools::ToolOutput> {
                let __json = input.into_json();
                let __args: #args_struct_ident = ::serde_json::from_value(__json)
                    .map_err(|e| ::cognis_core::error::CognisError::ToolValidationError(
                        e.to_string(),
                    ))?;
                <#args_struct_ident as ::cognis_core::tools::validation::ValidateArgs>::validate(&__args)?;
                self.#method_ident(#(__args.#field_idents),*).await
            }
        }
    }
}

// ---------------------------------------------------------------------------
// args_schema() body — schemars + per-field validator post-processing.
// ---------------------------------------------------------------------------

/// Build the body of `args_schema()`. Calls `schemars::schema_for!(Args)`,
/// converts to `serde_json::Value`, then inserts validator-derived keywords
/// (minimum/maximum/minLength/maxLength/pattern/enum/format) into the
/// matching `properties.<field>` objects.
fn emit_args_schema_body(args_struct_ident: &syn::Ident, specs: &[ArgSpec]) -> TokenStream2 {
    let mut field_mutations = Vec::new();
    for spec in specs {
        let field_name = spec.ident.to_string();
        let type_kind = classify_type(&spec.inner_ty);
        let mut inserts = Vec::new();

        for v in &spec.validators {
            match v {
                Validator::Range { min, max } => {
                    if let Some(m) = min {
                        let tok = number_token(*m);
                        inserts.push(quote! {
                            __field.insert("minimum".to_string(), ::serde_json::json!(#tok));
                        });
                    }
                    if let Some(m) = max {
                        let tok = number_token(*m);
                        inserts.push(quote! {
                            __field.insert("maximum".to_string(), ::serde_json::json!(#tok));
                        });
                    }
                }
                Validator::Length { min, max } => {
                    let (min_key, max_key) = match type_kind {
                        TypeKind::Vec => ("minItems", "maxItems"),
                        TypeKind::String | TypeKind::Other => ("minLength", "maxLength"),
                    };
                    if let Some(m) = min {
                        inserts.push(quote! {
                            __field.insert(#min_key.to_string(), ::serde_json::json!(#m));
                        });
                    }
                    if let Some(m) = max {
                        inserts.push(quote! {
                            __field.insert(#max_key.to_string(), ::serde_json::json!(#m));
                        });
                    }
                }
                Validator::Pattern(p) => {
                    let s = p.as_str();
                    inserts.push(quote! {
                        __field.insert("pattern".to_string(), ::serde_json::json!(#s));
                    });
                }
                Validator::EnumValues(values) => {
                    let list = values.iter().map(|v| quote! { #v }).collect::<Vec<_>>();
                    inserts.push(quote! {
                        __field.insert(
                            "enum".to_string(),
                            ::serde_json::json!([#(#list),*]),
                        );
                    });
                }
                Validator::Format(fmt) => {
                    let name = fmt.as_str();
                    inserts.push(quote! {
                        __field.insert("format".to_string(), ::serde_json::json!(#name));
                    });
                }
                Validator::Items(_) => {
                    // Items-nested validators are not yet propagated to the
                    // schema in the schemars-backed path. Runtime iteration
                    // for Vec items is also not implemented. Tracked as a
                    // follow-up.
                }
            }
        }

        if inserts.is_empty() {
            continue;
        }

        field_mutations.push(quote! {
            if let Some(__field) = __properties
                .get_mut(#field_name)
                .and_then(|v| v.as_object_mut())
            {
                #(#inserts)*
            }
        });
    }

    if field_mutations.is_empty() {
        return quote! {
            ::serde_json::to_value(::cognis_core::schemars::schema_for!(#args_struct_ident)).ok()
        };
    }

    quote! {
        let mut __schema = ::serde_json::to_value(
            ::cognis_core::schemars::schema_for!(#args_struct_ident)
        ).ok()?;
        if let Some(__properties) = __schema
            .get_mut("properties")
            .and_then(|v| v.as_object_mut())
        {
            #(#field_mutations)*
        }
        ::core::option::Option::Some(__schema)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum TypeKind {
    String,
    Vec,
    Other,
}

fn classify_type(ty: &Type) -> TypeKind {
    if let Type::Path(tp) = ty {
        if let Some(last) = tp.path.segments.last() {
            match last.ident.to_string().as_str() {
                "String" => return TypeKind::String,
                "Vec" => return TypeKind::Vec,
                _ => {}
            }
        }
    }
    TypeKind::Other
}

fn option_f64(v: &Option<f64>) -> TokenStream2 {
    match v {
        Some(x) => quote! { ::core::option::Option::Some(#x) },
        None => quote! { ::core::option::Option::None },
    }
}

fn option_usize(v: &Option<usize>) -> TokenStream2 {
    match v {
        Some(x) => quote! { ::core::option::Option::Some(#x) },
        None => quote! { ::core::option::Option::None },
    }
}

/// Emit a numeric literal as integer if it's a whole number within `i64`
/// range, otherwise as a float. Keeps `range(min = 1)` rendered as the
/// integer `1` in the schema, not `1.0`.
fn number_token(v: f64) -> TokenStream2 {
    if v.is_finite() && v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
        let as_i64 = v as i64;
        quote! { #as_i64 }
    } else {
        quote! { #v }
    }
}

fn pascal_case(s: &str) -> String {
    let mut out = String::new();
    let mut upper_next = true;
    for ch in s.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn pascal_case_ident(s: &str, span: Span) -> syn::Ident {
    syn::Ident::new(&pascal_case(s), span)
}

fn collect_doc_comment(attrs: &[syn::Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|a| {
            if !a.path().is_ident("doc") {
                return None;
            }
            if let Meta::NameValue(nv) = &a.meta {
                if let Expr::Lit(el) = &nv.value {
                    if let Lit::Str(s) = &el.lit {
                        let raw = s.value();
                        let trimmed = raw.strip_prefix(' ').unwrap_or(&raw).to_string();
                        return Some(trimmed);
                    }
                }
            }
            None
        })
        .collect();
    if lines.is_empty() {
        return None;
    }
    Some(lines.join(" ").trim().to_string())
}

#[allow(dead_code)]
fn _touch_spans(t: &dyn ToTokens) -> Span {
    t.span()
}
