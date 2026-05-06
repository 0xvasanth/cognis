//! A typed key-value container, keyed by `TypeId`. Inspired by
//! `tower::Extensions` / `axum::Extension`. Use this when a struct needs
//! to carry arbitrary plugin-supplied data without baking those types into
//! its public surface.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;

/// A typed map indexed by `TypeId`. Each `T: Any + Send + Sync + 'static`
/// can have at most one value stored.
#[derive(Default)]
pub struct Extensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    /// Create an empty `Extensions`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value. Returns the previous value of the same type, if any.
    pub fn insert<T: Any + Send + Sync + 'static>(&mut self, value: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|prev| prev.downcast::<T>().ok().map(|b| *b))
    }

    /// Get a reference to a stored value of type `T`.
    pub fn get<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    /// Get a mutable reference to a stored value of type `T`.
    pub fn get_mut<T: Any + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    /// Remove and return the stored value of type `T`.
    pub fn remove<T: Any + Send + Sync + 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// True if a value of type `T` is stored.
    pub fn contains<T: Any + Send + Sync + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Number of distinct types stored.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if no values are stored.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear all stored values.
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

impl fmt::Debug for Extensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Extensions")
            .field("len", &self.map.len())
            .finish()
    }
}

impl Clone for Extensions {
    /// Clones the *map*, but NOT the values inside. Cloning here makes a
    /// fresh empty `Extensions` — there's no general way to clone arbitrary
    /// `dyn Any`. If you need cloned values, store `Arc<T>` instead of `T`.
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct UserId(u32);

    #[derive(Debug, PartialEq)]
    struct Tag(&'static str);

    #[test]
    fn insert_get_remove() {
        let mut ex = Extensions::new();
        assert!(ex.is_empty());

        assert!(ex.insert(UserId(42)).is_none());
        assert_eq!(ex.get::<UserId>(), Some(&UserId(42)));
        assert_eq!(ex.len(), 1);

        let prev = ex.insert(UserId(99));
        assert_eq!(prev, Some(UserId(42)));
        assert_eq!(ex.get::<UserId>(), Some(&UserId(99)));

        let removed = ex.remove::<UserId>();
        assert_eq!(removed, Some(UserId(99)));
        assert!(ex.get::<UserId>().is_none());
    }

    #[test]
    fn distinct_types_coexist() {
        let mut ex = Extensions::new();
        ex.insert(UserId(1));
        ex.insert(Tag("admin"));
        assert_eq!(ex.get::<UserId>(), Some(&UserId(1)));
        assert_eq!(ex.get::<Tag>(), Some(&Tag("admin")));
        assert_eq!(ex.len(), 2);
    }

    #[test]
    fn clone_yields_empty() {
        let mut ex = Extensions::new();
        ex.insert(UserId(1));
        let cloned = ex.clone();
        assert!(cloned.is_empty());
    }
}
