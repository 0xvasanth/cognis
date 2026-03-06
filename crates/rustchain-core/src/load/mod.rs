//! Serialization and deserialization infrastructure.
//!
//! Mirrors Python `langchain_core.load`.
//!
//! Provides a `Serializable` trait for objects that can be serialized
//! to a constructor-based JSON format, plus `dumpd`/`dumps` for
//! serialization and `load`/`loads` for deserialization.

pub mod dump;
#[allow(clippy::module_inception)]
pub mod load;
pub mod mapping;
pub mod serializable;
pub mod validation;

pub use dump::{
    dumpd as dumpd_value, dumpd_serializable, dumps as dumps_value, dumps_serializable,
};
pub use load::{load as load_value, loads, Reviver};
pub use mapping::{get_default_mappings, is_allowed_class_path};
pub use serializable::{
    dumpd, dumps, BaseSerialized, Serializable, SerializedConstructor, SerializedNotImplemented,
    SerializedSecret,
};
pub use validation::{
    escape_dict, is_escaped_dict, is_lc_secret, needs_escaping, serialize_value, unescape_value,
    LC_ESCAPED_KEY,
};
