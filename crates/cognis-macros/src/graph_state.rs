use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Expr, Fields, Lit, Meta, MetaNameValue};

/// Extract the `#[reducer(append)]` attribute value from a field.
fn get_reducer_attr(field: &syn::Field) -> Option<String> {
    for attr in &field.attrs {
        if attr.path().is_ident("reducer") {
            let mut reducer_name = None;
            let _ = attr.parse_nested_meta(|meta| {
                if let Some(ident) = meta.path.get_ident() {
                    reducer_name = Some(ident.to_string());
                }
                Ok(())
            });
            return reducer_name;
        }
    }
    None
}

/// Extract `///` doc comments from a field.
fn get_field_doc(field: &syn::Field) -> Option<String> {
    let mut docs = Vec::new();
    for attr in &field.attrs {
        if attr.path().is_ident("doc") {
            if let Meta::NameValue(MetaNameValue {
                value:
                    Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }),
                ..
            }) = &attr.meta
            {
                docs.push(s.value().trim().to_string());
            }
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.join(" "))
    }
}

pub fn derive_graph_state(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(name, "GraphState only supports named fields")
                    .to_compile_error();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "GraphState can only be derived for structs")
                .to_compile_error();
        }
    };

    let mut field_inserts = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap().to_string();
        let reducer_kind = get_reducer_attr(field).unwrap_or_else(|| "last_value".to_string());
        let description = get_field_doc(field);

        let reducer_fn = match reducer_kind.as_str() {
            "append" => quote! {
                Box::new(|current: &serde_json::Value, update: &serde_json::Value| -> serde_json::Value {
                    let mut result = match current.as_array() {
                        Some(arr) => arr.clone(),
                        None => vec![current.clone()],
                    };
                    match update.as_array() {
                        Some(arr) => result.extend(arr.iter().cloned()),
                        None => result.push(update.clone()),
                    }
                    serde_json::Value::Array(result)
                })
            },
            "merge" => quote! {
                Box::new(|current: &serde_json::Value, update: &serde_json::Value| -> serde_json::Value {
                    let mut result = current.clone();
                    if let (Some(cur), Some(upd)) = (result.as_object_mut(), update.as_object()) {
                        for (k, v) in upd {
                            cur.insert(k.clone(), v.clone());
                        }
                    }
                    result
                })
            },
            "add" => quote! {
                Box::new(|current: &serde_json::Value, update: &serde_json::Value| -> serde_json::Value {
                    let a = current.as_f64().unwrap_or(0.0);
                    let b = update.as_f64().unwrap_or(0.0);
                    let sum = a + b;
                    // Preserve integer type when both inputs are integers.
                    if current.is_i64() && update.is_i64() {
                        serde_json::json!(sum as i64)
                    } else {
                        serde_json::json!(sum)
                    }
                })
            },
            "last_value" => quote! {
                Box::new(|_current: &serde_json::Value, update: &serde_json::Value| -> serde_json::Value {
                    update.clone()
                })
            },
            unknown => {
                let msg = format!(
                    "unknown reducer '{}'. Expected one of: append, last_value, add, merge",
                    unknown
                );
                return syn::Error::new_spanned(field, msg).to_compile_error();
            }
        };

        let desc_token = match &description {
            Some(d) => quote! { Some(#d.to_string()) },
            None => quote! { None },
        };

        field_inserts.push(quote! {
            fields.insert(
                #field_name.to_string(),
                cognisgraph::GraphStateField {
                    reducer: #reducer_fn,
                    description: #desc_token,
                },
            );
        });
    }

    quote! {
        impl #name {
            /// Returns the graph state schema with per-field reducers.
            ///
            /// Generated by `#[derive(GraphState)]`.
            pub fn graph_state() -> cognisgraph::GraphStateSchema {
                let mut fields = std::collections::HashMap::new();
                #(#field_inserts)*
                cognisgraph::GraphStateSchema { fields }
            }
        }
    }
}
