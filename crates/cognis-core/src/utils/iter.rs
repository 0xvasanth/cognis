//! Iteration utilities.

/// Batch an iterator into chunks of a given size.
pub fn batch_iterate<T>(size: usize, iterable: impl IntoIterator<Item = T>) -> Vec<Vec<T>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::with_capacity(size);
    for item in iterable {
        current_batch.push(item);
        if current_batch.len() == size {
            batches.push(current_batch);
            current_batch = Vec::with_capacity(size);
        }
    }
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }
    batches
}
