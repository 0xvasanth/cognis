//! `#[tools_impl]` — outer attribute that scans an `impl` block for inner
//! `#[tool]`-marked async methods and generates one `BaseTool`-implementing
//! wrapper struct per method, plus an `into_tools()` collector on the user's
//! struct.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    Attribute, Expr, FnArg, ImplItem, ImplItemFn, ItemImpl, Lit, LitStr, Meta, Pat, Token, Type,
};

#[derive(Default)]
pub(crate) struct ToolsImplArgs {
    pub crate_path: String,
}

impl Parse for ToolsImplArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ToolsImplArgs {
            crate_path: "cognis_core".to_string(),
        };
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "crate_path" => args.crate_path = input.parse::<LitStr>()?.value(),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown #[tools_impl] argument `{other}`; expected crate_path"),
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

/// Per-#[tool] method args. A subset of ToolArgs from tool_attr.rs (we
/// don't share that type because tools_impl always inherits crate_path
/// from the outer #[tools_impl]).
struct InnerToolArgs {
    name: Option<String>,
    description: Option<String>,
}

fn parse_inner_tool_attr(attr: &Attribute) -> syn::Result<InnerToolArgs> {
    let mut name = None;
    let mut description = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            let v = meta.value()?;
            let lit: LitStr = v.parse()?;
            name = Some(lit.value());
            Ok(())
        } else if meta.path.is_ident("description") {
            let v = meta.value()?;
            let lit: LitStr = v.parse()?;
            description = Some(lit.value());
            Ok(())
        } else {
            Err(meta.error("inside #[tools_impl], inner #[tool] supports only name and description"))
        }
    })?;
    Ok(InnerToolArgs { name, description })
}

pub(crate) fn expand(args: ToolsImplArgs, input: TokenStream2) -> syn::Result<TokenStream2> {
    let item_impl: ItemImpl = syn::parse2(input)?;

    if item_impl.generics.params.iter().next().is_some() {
        return Err(syn::Error::new_spanned(
            &item_impl.generics,
            "#[tools_impl] does not support generic impl blocks",
        ));
    }
    if let Some((_, path, _)) = &item_impl.trait_ {
        return Err(syn::Error::new_spanned(
            path,
            "#[tools_impl] must be applied to an inherent impl block (not a trait impl)",
        ));
    }

    let self_ty = (*item_impl.self_ty).clone();
    let struct_ident = match &self_ty {
        Type::Path(tp) => tp
            .path
            .segments
            .last()
            .map(|s| s.ident.clone())
            .ok_or_else(|| syn::Error::new_spanned(&self_ty, "cannot resolve impl target name"))?,
        _ => {
            return Err(syn::Error::new_spanned(
                &self_ty,
                "#[tools_impl] target must be a named struct type",
            ))
        }
    };

    let root = root_path(&args.crate_path);

    // Collect (method, inner-tool-args) pairs and strip the #[tool] attr from each.
    let mut tool_methods: Vec<(ImplItemFn, InnerToolArgs)> = Vec::new();
    let mut cleaned_impl = item_impl.clone();
    for item in cleaned_impl.items.iter_mut() {
        if let ImplItem::Fn(m) = item {
            if let Some(idx) = m.attrs.iter().position(|a| a.path().is_ident("tool")) {
                let tool_attr = m.attrs.remove(idx);
                let parsed = parse_inner_tool_attr(&tool_attr)?;
                tool_methods.push((m.clone(), parsed));
                // Also strip arg-level helper attributes from the kept method.
                for input in m.sig.inputs.iter_mut() {
                    if let FnArg::Typed(pt) = input {
                        pt.attrs
                            .retain(|a| !a.path().is_ident("schema") && !a.path().is_ident("doc"));
                    }
                }
            }
        }
    }

    if tool_methods.is_empty() {
        return Err(syn::Error::new_spanned(
            &item_impl,
            "#[tools_impl] requires at least one method annotated with #[tool]",
        ));
    }

    let mut wrappers = Vec::new();
    let mut into_tools_pushes = Vec::new();

    for (method, inner) in &tool_methods {
        validate_receiver(method)?;

        let method_ident = &method.sig.ident;
        let tool_name = inner
            .name
            .clone()
            .unwrap_or_else(|| method_ident.to_string());
        let description = inner
            .description
            .clone()
            .or_else(|| collect_doc_comment(&method.attrs))
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    &method.sig,
                    "#[tools_impl]: inner #[tool] requires a description (either via attribute or `///` doc comment)",
                )
            })?;

        let wrapper_ident = format_ident!(
            "{}{}Tool",
            struct_ident,
            pascal_case(&method_ident.to_string())
        );

        let arg_specs = parse_typed_args(&method.sig)?;
        // v2 #[tools_impl] follows the rsllm "single params struct" pattern:
        // each tool method MUST take exactly one typed argument besides &self,
        // and that argument's type IS the tool's params type. Unwrapping it
        // means the LLM sees the params schema directly (no spurious `"p"` key).
        if arg_specs.len() != 1 {
            return Err(syn::Error::new_spanned(
                &method.sig,
                "#[tools_impl]: each #[tool] method must take exactly one params struct argument \
                 (besides &self). Multi-arg and zero-arg variants are not supported in slice 1.",
            ));
        }
        let arg_ty = &arg_specs[0].ty;

        wrappers.push(quote! {
            #[allow(non_camel_case_types)]
            pub struct #wrapper_ident {
                inner: ::std::sync::Arc<#self_ty>,
            }

            #[::async_trait::async_trait]
            impl #root::tools::BaseTool for #wrapper_ident {
                fn name(&self) -> &str { #tool_name }
                fn description(&self) -> &str { #description }
                fn args_schema(&self) -> ::core::option::Option<::serde_json::Value> {
                    ::core::option::Option::Some(
                        ::serde_json::to_value(
                            #root::schemars::schema_for!(#arg_ty)
                        ).expect("schemars output is always serializable")
                    )
                }
                async fn _run(
                    &self,
                    input: #root::tools::ToolInput,
                ) -> #root::error::Result<#root::tools::ToolOutput> {
                    let __json = input.into_json();
                    let __args: #arg_ty = ::serde_json::from_value(__json)
                        .map_err(|e| #root::error::CognisError::ToolValidationError(
                            e.to_string(),
                        ))?;
                    self.inner.#method_ident(__args).await
                }
            }
        });

        into_tools_pushes.push(quote! {
            ::std::sync::Arc::new(#wrapper_ident { inner: self.clone() })
                as ::std::sync::Arc<dyn #root::tools::BaseTool>,
        });
    }

    let collector = quote! {
        impl #self_ty {
            /// Collect every #[tool]-annotated method as a `Vec<Arc<dyn BaseTool>>`,
            /// each holding a clone of the shared `Arc<Self>` so all tools observe
            /// the same instance.
            pub fn into_tools(
                self: ::std::sync::Arc<Self>,
            ) -> ::std::vec::Vec<::std::sync::Arc<dyn #root::tools::BaseTool>> {
                vec![ #(#into_tools_pushes)* ]
            }
        }
    };

    Ok(quote! {
        #cleaned_impl
        #(#wrappers)*
        #collector
    })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn validate_receiver(method: &ImplItemFn) -> syn::Result<()> {
    if method.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[tools_impl]: #[tool] methods must be `async`",
        ));
    }
    let receiver = method.sig.receiver().ok_or_else(|| {
        syn::Error::new_spanned(
            &method.sig,
            "#[tools_impl]: #[tool] methods must take `&self`",
        )
    })?;
    if receiver.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            receiver,
            "#[tools_impl]: #[tool] methods must take `&self` (not `&mut self`)",
        ));
    }
    if receiver.reference.is_none() {
        return Err(syn::Error::new_spanned(
            receiver,
            "#[tools_impl]: #[tool] methods must take `&self` (consuming `self` is rejected)",
        ));
    }
    Ok(())
}

struct ArgSpec {
    #[allow(dead_code)]
    ident: syn::Ident,
    ty: Type,
    #[allow(dead_code)]
    docs: Vec<Attribute>,
}

fn parse_typed_args(sig: &syn::Signature) -> syn::Result<Vec<ArgSpec>> {
    let mut specs = Vec::new();
    for input in &sig.inputs {
        match input {
            FnArg::Receiver(_) => continue,
            FnArg::Typed(pat_type) => {
                let ident = match &*pat_type.pat {
                    Pat::Ident(pi) => pi.ident.clone(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "#[tools_impl]: tool args must be plain identifiers",
                        ))
                    }
                };
                if let Type::Reference(tr) = &*pat_type.ty {
                    return Err(syn::Error::new_spanned(
                        tr,
                        "#[tools_impl]: tool args must be owned types (e.g. `String`, not `&str`)",
                    ));
                }
                let docs = pat_type
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("doc"))
                    .cloned()
                    .collect();
                specs.push(ArgSpec {
                    ident,
                    ty: (*pat_type.ty).clone(),
                    docs,
                });
            }
        }
    }
    Ok(specs)
}

fn root_path(crate_path: &str) -> syn::Path {
    let segments: Vec<syn::Ident> = crate_path
        .split("::")
        .map(|seg| syn::Ident::new(seg, Span::call_site()))
        .collect();
    syn::parse_quote!(:: #(#segments)::*)
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

fn collect_doc_comment(attrs: &[Attribute]) -> Option<String> {
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
                        return Some(raw.strip_prefix(' ').unwrap_or(&raw).to_string());
                    }
                }
            }
            None
        })
        .collect();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join(" ").trim().to_string())
    }
}
