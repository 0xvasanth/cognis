//! `#[derive(GraphStateV2)]` — emits a typed sibling `XUpdate` struct with
//! the same field names as `X`, plus an `impl GraphState for X` whose
//! `apply()` dispatches per-field reducers.
//!
//! Fields opt into reducers via `#[reducer(append|add|last|merge|skip)]`
//! or `#[reducer(custom = "fn_path")]`. Fields named `extras` are auto-
//! skipped (treated as `#[reducer(skip)]`) — they're owned by middleware,
//! not the per-step graph reducer.
//!
//! The `crate_path` to the framework is configurable via
//! `#[graph_state(crate_path = "...")]`, defaulting to `cognis2_graph`
//! since this is a v2-only derive.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Attribute, Data, DeriveInput, Fields, LitStr};

#[derive(Clone)]
enum Reducer {
    Append,
    Add,
    Last,
    Merge,
    Skip,
    Custom(syn::Path),
}

struct ReducedField {
    ident: syn::Ident,
    ty: syn::Type,
    reducer: Reducer,
}

pub fn derive_graph_state_v2(input: DeriveInput) -> TokenStream {
    let crate_path = parse_crate_path(&input.attrs).unwrap_or_else(|| "cognis2_graph".to_string());
    let root = root_path(&crate_path);
    let name = &input.ident;
    let update_name = format_ident!("{}Update", name);
    let vis = &input.vis;

    let named = match &input.data {
        Data::Struct(d) => match &d.fields {
            Fields::Named(n) => &n.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "GraphStateV2 requires a struct with named fields",
                )
                .to_compile_error();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "GraphStateV2 requires a struct")
                .to_compile_error();
        }
    };

    let mut fields = Vec::new();
    for f in named {
        let ident = match f.ident.clone() {
            Some(i) => i,
            None => continue,
        };
        let ty = f.ty.clone();
        let reducer = match parse_reducer_attr(&f.attrs) {
            Ok(Some(r)) => r,
            Ok(None) => {
                if ident == "extras" {
                    Reducer::Skip
                } else {
                    Reducer::Last
                }
            }
            Err(e) => return e.to_compile_error(),
        };
        fields.push(ReducedField { ident, ty, reducer });
    }

    let update_fields = fields.iter().filter(|f| !matches!(f.reducer, Reducer::Skip)).map(|f| {
        let id = &f.ident;
        let ty = &f.ty;
        // Update type: Append / Custom keep the same type; Last wraps in Option;
        // Add takes the same numeric type; Merge takes serde_json::Value.
        let upd_ty = match &f.reducer {
            Reducer::Last => quote! { ::core::option::Option<#ty> },
            Reducer::Merge => quote! { ::core::option::Option<::serde_json::Value> },
            _ => quote! { #ty },
        };
        quote! { pub #id: #upd_ty, }
    });

    let apply_arms = fields.iter().filter(|f| !matches!(f.reducer, Reducer::Skip)).map(|f| {
        let id = &f.ident;
        match &f.reducer {
            Reducer::Append => quote! {
                self.#id.extend(update.#id);
            },
            Reducer::Add => quote! {
                self.#id = self.#id + update.#id;
            },
            Reducer::Last => quote! {
                if let ::core::option::Option::Some(v) = update.#id {
                    self.#id = v;
                }
            },
            Reducer::Merge => quote! {
                if let ::core::option::Option::Some(v) = update.#id {
                    #root::__merge_json(&mut self.#id, v);
                }
            },
            Reducer::Custom(path) => quote! {
                #path(&mut self.#id, update.#id);
            },
            Reducer::Skip => quote! {},
        }
    });

    quote! {
        #[derive(::core::default::Default, ::core::clone::Clone, ::core::fmt::Debug)]
        #[allow(non_camel_case_types, dead_code)]
        #vis struct #update_name {
            #(#update_fields)*
        }

        impl #root::GraphState for #name {
            type Update = #update_name;
            fn apply(&mut self, update: Self::Update) {
                #(#apply_arms)*
            }
        }
    }
}

fn parse_reducer_attr(attrs: &[Attribute]) -> syn::Result<Option<Reducer>> {
    for attr in attrs {
        if !attr.path().is_ident("reducer") {
            continue;
        }
        let mut found: Option<Reducer> = None;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("append") {
                found = Some(Reducer::Append);
            } else if meta.path.is_ident("add") {
                found = Some(Reducer::Add);
            } else if meta.path.is_ident("last") {
                found = Some(Reducer::Last);
            } else if meta.path.is_ident("merge") {
                found = Some(Reducer::Merge);
            } else if meta.path.is_ident("skip") {
                found = Some(Reducer::Skip);
            } else if meta.path.is_ident("custom") {
                let v = meta.value()?;
                let lit: LitStr = v.parse()?;
                let path: syn::Path = syn::parse_str(&lit.value())?;
                found = Some(Reducer::Custom(path));
            } else {
                return Err(meta.error(
                    "unknown reducer; expected append, add, last, merge, skip, or custom = \"path::fn\"",
                ));
            }
            Ok(())
        })?;
        return Ok(found);
    }
    Ok(None)
}

fn parse_crate_path(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("graph_state") {
            continue;
        }
        let mut path = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("crate_path") {
                let v = meta.value()?;
                let lit: LitStr = v.parse()?;
                path = Some(lit.value());
            }
            Ok(())
        });
        return path;
    }
    None
}

fn root_path(crate_path: &str) -> syn::Path {
    let segments: Vec<syn::Ident> = crate_path
        .split("::")
        .map(|seg| syn::Ident::new(seg, Span::call_site()))
        .collect();
    syn::parse_quote!(:: #(#segments)::*)
}
