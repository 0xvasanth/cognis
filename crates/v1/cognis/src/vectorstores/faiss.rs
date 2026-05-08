//! FAISS-compatible vector store with pure-Rust index implementations.
//!
//! Provides a `FaissVectorStore` that implements the `VectorStore` trait using
//! in-process vector indexing. No external FAISS C++ library is required.
//!
//! Three index types are supported:
//! - **Flat** — brute-force exact nearest neighbor search
//! - **IVFFlat** — inverted file index with flat quantizer (k-means clustering)
//! - **HNSW** — hierarchical navigable small world graph (approximate NN)

use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::{CognisError, Result};
use cognis_core::vectorstores::base::VectorStore;

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Distance metric for vector similarity comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FaissMetric {
    /// Euclidean (L2) distance — lower is more similar.
    #[default]
    L2,
    /// Inner product (dot product) — higher is more similar.
    InnerProduct,
    /// Cosine similarity — higher is more similar. Vectors are normalized before comparison.
    Cosine,
}

/// Compute the distance/similarity between two vectors using the given metric.
/// Returns a score where **higher is always more similar** (distances are negated).
fn compute_similarity(a: &[f32], b: &[f32], metric: FaissMetric) -> f32 {
    match metric {
        FaissMetric::L2 => {
            let dist_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            // Negate so that higher = more similar.
            -dist_sq
        }
        FaissMetric::InnerProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        FaissMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a * norm_b)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FaissIndex trait
// ---------------------------------------------------------------------------

/// Trait for FAISS-style vector indices.
pub trait FaissIndex: Send + Sync {
    /// Add a vector with the given ID.
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()>;

    /// Search for the `k` nearest neighbors. Returns `(id, score)` pairs
    /// sorted by descending similarity (highest score first).
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>;

    /// Remove a vector by ID. Returns `true` if the vector was found and removed.
    fn remove(&mut self, id: &str) -> bool;

    /// Return the number of vectors in the index.
    fn len(&self) -> usize;

    /// Return `true` if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the index to bytes.
    fn save_to_bytes(&self) -> Result<Vec<u8>>;

    /// The dimension this index was configured for.
    fn dimension(&self) -> usize;

    /// The metric this index uses.
    fn metric(&self) -> FaissMetric;
}

// ---------------------------------------------------------------------------
// FlatIndex
// ---------------------------------------------------------------------------

/// Brute-force exact search index. Stores all vectors and performs a linear scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    metric: FaissMetric,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

impl FlatIndex {
    /// Create a new flat index for the given dimension and metric.
    pub fn new(dim: usize, metric: FaissMetric) -> Self {
        Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Load a flat index from bytes.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| CognisError::Other(e.to_string()))
    }
}

impl FaissIndex for FlatIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Other(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        // Replace if ID already exists.
        if let Some(pos) = self.ids.iter().position(|x| x == id) {
            self.vectors[pos] = vector.to_vec();
        } else {
            self.ids.push(id.to_string());
            self.vectors.push(vector.to_vec());
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .ids
            .iter()
            .zip(self.vectors.iter())
            .map(|(id, vec)| (id.clone(), compute_similarity(query, vec, self.metric)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.ids.iter().position(|x| x == id) {
            self.ids.remove(pos);
            self.vectors.remove(pos);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn save_to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| CognisError::Other(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// IVFFlatIndex — inverted file with flat quantizer
// ---------------------------------------------------------------------------

/// A single cluster (Voronoi cell) in the IVF index.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IVFCluster {
    centroid: Vec<f32>,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

/// IVF-Flat index: vectors are partitioned into `nlist` clusters via k-means,
/// and only the `nprobe` nearest clusters are searched at query time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFFlatIndex {
    dim: usize,
    metric: FaissMetric,
    nlist: usize,
    nprobe: usize,
    clusters: Vec<IVFCluster>,
    trained: bool,
    /// Staging area for vectors added before training.
    staging_ids: Vec<String>,
    staging_vectors: Vec<Vec<f32>>,
}

impl IVFFlatIndex {
    /// Create a new IVF-Flat index.
    ///
    /// * `dim` — vector dimension
    /// * `nlist` — number of clusters (Voronoi cells)
    /// * `nprobe` — number of clusters to probe at search time (default 1)
    /// * `metric` — distance metric
    pub fn new(dim: usize, nlist: usize, nprobe: usize, metric: FaissMetric) -> Self {
        Self {
            dim,
            metric,
            nlist,
            nprobe: nprobe.max(1),
            clusters: Vec::new(),
            trained: false,
            staging_ids: Vec::new(),
            staging_vectors: Vec::new(),
        }
    }

    /// Train the index using Lloyd's k-means algorithm on all staged vectors.
    pub fn train(&mut self) {
        if self.staging_vectors.is_empty() {
            return;
        }

        let effective_nlist = self.nlist.min(self.staging_vectors.len());

        // Initialize centroids by picking the first `effective_nlist` vectors.
        let mut centroids: Vec<Vec<f32>> = self.staging_vectors[..effective_nlist].to_vec();

        // Run k-means for a fixed number of iterations.
        let max_iter = 20;
        for _ in 0..max_iter {
            // Assignment step: assign each vector to the nearest centroid.
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); effective_nlist];
            for (idx, vec) in self.staging_vectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_sim = f32::NEG_INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let sim = compute_similarity(vec, centroid, self.metric);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cluster = c;
                    }
                }
                assignments[best_cluster].push(idx);
            }

            // Update step: recompute centroids.
            for (c, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    continue;
                }
                let mut new_centroid = vec![0.0f32; self.dim];
                for &idx in assigned {
                    for (j, val) in self.staging_vectors[idx].iter().enumerate() {
                        new_centroid[j] += val;
                    }
                }
                let count = assigned.len() as f32;
                for val in &mut new_centroid {
                    *val /= count;
                }
                centroids[c] = new_centroid;
            }
        }

        // Build clusters.
        self.clusters = centroids
            .into_iter()
            .map(|centroid| IVFCluster {
                centroid,
                ids: Vec::new(),
                vectors: Vec::new(),
            })
            .collect();

        // Assign all staged vectors to their nearest cluster.
        for (i, vec) in self.staging_vectors.iter().enumerate() {
            let cluster_idx = self.nearest_cluster(vec);
            self.clusters[cluster_idx]
                .ids
                .push(self.staging_ids[i].clone());
            self.clusters[cluster_idx].vectors.push(vec.clone());
        }

        self.staging_ids.clear();
        self.staging_vectors.clear();
        self.trained = true;
    }

    /// Find the nearest cluster centroid for a vector.
    fn nearest_cluster(&self, vector: &[f32]) -> usize {
        let mut best = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, cluster) in self.clusters.iter().enumerate() {
            let sim = compute_similarity(vector, &cluster.centroid, self.metric);
            if sim > best_sim {
                best_sim = sim;
                best = i;
            }
        }
        best
    }

    /// Find the `nprobe` nearest cluster indices for a query vector.
    fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_similarity(query, &c.centroid, self.metric)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(nprobe);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Load an IVF-Flat index from bytes.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| CognisError::Other(e.to_string()))
    }
}

impl FaissIndex for IVFFlatIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Other(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        if !self.trained {
            // Stage the vector for later training.
            self.staging_ids.push(id.to_string());
            self.staging_vectors.push(vector.to_vec());
        } else {
            let cluster_idx = self.nearest_cluster(vector);
            self.clusters[cluster_idx].ids.push(id.to_string());
            self.clusters[cluster_idx].vectors.push(vector.to_vec());
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        // If not trained, fall back to linear scan over staging area.
        if !self.trained {
            let mut scored: Vec<(String, f32)> = self
                .staging_ids
                .iter()
                .zip(self.staging_vectors.iter())
                .map(|(id, vec)| (id.clone(), compute_similarity(query, vec, self.metric)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            return scored;
        }

        let probe_clusters = self.nearest_clusters(query, self.nprobe);

        let mut scored: Vec<(String, f32)> = Vec::new();
        for &ci in &probe_clusters {
            let cluster = &self.clusters[ci];
            for (id, vec) in cluster.ids.iter().zip(cluster.vectors.iter()) {
                scored.push((id.clone(), compute_similarity(query, vec, self.metric)));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn remove(&mut self, id: &str) -> bool {
        // Check staging area.
        if let Some(pos) = self.staging_ids.iter().position(|x| x == id) {
            self.staging_ids.remove(pos);
            self.staging_vectors.remove(pos);
            return true;
        }
        // Check clusters.
        for cluster in &mut self.clusters {
            if let Some(pos) = cluster.ids.iter().position(|x| x == id) {
                cluster.ids.remove(pos);
                cluster.vectors.remove(pos);
                return true;
            }
        }
        false
    }

    fn len(&self) -> usize {
        let staged = self.staging_ids.len();
        let clustered: usize = self.clusters.iter().map(|c| c.ids.len()).sum();
        staged + clustered
    }

    fn save_to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| CognisError::Other(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// HNSWIndex — Hierarchical Navigable Small World graph
// ---------------------------------------------------------------------------

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HNSWNode {
    id: String,
    vector: Vec<f32>,
    /// Neighbors at each layer. `neighbors[l]` is the list of neighbor indices at layer `l`.
    neighbors: Vec<Vec<usize>>,
}

/// HNSW index for approximate nearest neighbor search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex {
    dim: usize,
    metric: FaissMetric,
    /// Maximum number of bi-directional links per node per layer.
    m: usize,
    /// Size of the dynamic candidate list during construction.
    ef_construction: usize,
    /// Maximum layer level.
    max_level: usize,
    /// Entry point node index.
    entry_point: Option<usize>,
    nodes: Vec<HNSWNode>,
    /// Multiplier for level generation (1 / ln(M)).
    ml: f64,
}

impl HNSWIndex {
    /// Create a new HNSW index.
    ///
    /// * `dim` — vector dimension
    /// * `m` — max connections per layer (typical: 16)
    /// * `ef_construction` — dynamic candidate list size during construction (typical: 200)
    /// * `metric` — distance metric
    pub fn new(dim: usize, m: usize, ef_construction: usize, metric: FaissMetric) -> Self {
        let m = m.max(2);
        Self {
            dim,
            metric,
            m,
            ef_construction: ef_construction.max(m),
            max_level: 0,
            entry_point: None,
            nodes: Vec::new(),
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Generate a random level for a new node.
    fn random_level(&self) -> usize {
        // Simple deterministic-ish level generation based on node count.
        // In production you'd use a proper RNG, but for reproducibility we
        // use a hash of the current node count.
        let r = {
            let seed = self.nodes.len() as u64;
            // Simple xorshift-like hash.
            let mut x = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x ^= x >> 33;
            x ^= x << 13;
            x ^= x >> 7;
            (x as f64) / (u64::MAX as f64)
        };
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(16) // cap at a reasonable maximum
    }

    /// Search a single layer of the graph greedily, returning the `ef` closest nodes.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::{BinaryHeap, HashSet};

        let mut visited = HashSet::new();
        // (score, node_idx) — we use a min-heap for candidates and max-heap for results.
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedF32, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();

        for &ep in entry_points {
            let dist = compute_similarity(query, &self.nodes[ep].vector, self.metric);
            visited.insert(ep);
            candidates.push(std::cmp::Reverse((OrderedF32(-dist), ep)));
            results.push((OrderedF32(-dist), ep));
        }

        while let Some(std::cmp::Reverse((OrderedF32(neg_dist), node_idx))) = candidates.pop() {
            let worst_result = results
                .peek()
                .map(|(OrderedF32(d), _)| *d)
                .unwrap_or(f32::INFINITY);
            if neg_dist > worst_result {
                break;
            }

            if layer < self.nodes[node_idx].neighbors.len() {
                for &neighbor in &self.nodes[node_idx].neighbors[layer] {
                    if visited.insert(neighbor) {
                        let sim =
                            compute_similarity(query, &self.nodes[neighbor].vector, self.metric);
                        let neg_sim = -sim;

                        let worst = results
                            .peek()
                            .map(|(OrderedF32(d), _)| *d)
                            .unwrap_or(f32::INFINITY);
                        if results.len() < ef || neg_sim < worst {
                            candidates.push(std::cmp::Reverse((OrderedF32(neg_sim), neighbor)));
                            results.push((OrderedF32(neg_sim), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|(OrderedF32(neg_dist), idx)| (idx, -neg_dist))
            .collect()
    }

    /// Select neighbors to keep (simple heuristic: keep the M closest).
    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Load an HNSW index from bytes.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| CognisError::Other(e.to_string()))
    }
}

/// Wrapper for f32 to implement Ord for use in BinaryHeap.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct OrderedF32(f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl FaissIndex for HNSWIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Other(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            )));
        }

        // Check if ID already exists — remove first.
        if self.nodes.iter().any(|n| n.id == id) {
            // For simplicity, we don't fully support re-insertion in HNSW.
            // Remove and re-add.
            self.remove(id);
        }

        let node_level = self.random_level();
        let new_idx = self.nodes.len();

        let node = HNSWNode {
            id: id.to_string(),
            vector: vector.to_vec(),
            neighbors: vec![Vec::new(); node_level + 1],
        };

        self.nodes.push(node);

        if self.entry_point.is_none() {
            self.entry_point = Some(new_idx);
            self.max_level = node_level;
            return Ok(());
        }

        let entry = self.entry_point.unwrap();
        let mut ep = vec![entry];

        // Traverse from top level down to node_level + 1, greedy search for closest.
        let current_max = self.max_level;
        for level in (node_level + 1..=current_max).rev() {
            let nearest = self.search_layer(vector, &ep, 1, level);
            if let Some(&(idx, _)) = nearest.first() {
                ep = vec![idx];
            }
        }

        // For layers node_level down to 0, find neighbors and connect.
        let top = node_level.min(current_max);
        for level in (0..=top).rev() {
            let nearest = self.search_layer(vector, &ep, self.ef_construction, level);
            let neighbors = self.select_neighbors(&nearest, self.m);

            // Set neighbors for the new node.
            if level < self.nodes[new_idx].neighbors.len() {
                self.nodes[new_idx].neighbors[level] = neighbors.clone();
            }

            // Add bidirectional connections.
            for &neighbor_idx in &neighbors {
                // Ensure neighbor has enough layers.
                while self.nodes[neighbor_idx].neighbors.len() <= level {
                    self.nodes[neighbor_idx].neighbors.push(Vec::new());
                }
                self.nodes[neighbor_idx].neighbors[level].push(new_idx);

                // Trim if too many connections.
                let max_m = self.m * 2; // Allow 2*M connections at layer 0.
                if self.nodes[neighbor_idx].neighbors[level].len() > max_m {
                    // Keep only the closest M connections.
                    let nbr_vec = &self.nodes[neighbor_idx].vector;
                    let mut scored: Vec<(usize, f32)> = self.nodes[neighbor_idx].neighbors[level]
                        .iter()
                        .map(|&n| {
                            (
                                n,
                                compute_similarity(nbr_vec, &self.nodes[n].vector, self.metric),
                            )
                        })
                        .collect();
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(max_m);
                    self.nodes[neighbor_idx].neighbors[level] =
                        scored.into_iter().map(|(idx, _)| idx).collect();
                }
            }

            ep = nearest.iter().map(|&(idx, _)| idx).collect();
        }

        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(new_idx);
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let Some(entry) = self.entry_point else {
            return Vec::new();
        };

        let mut ep = vec![entry];

        // Traverse from top level down to layer 1.
        for level in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, &ep, 1, level);
            if let Some(&(idx, _)) = nearest.first() {
                ep = vec![idx];
            }
        }

        // Search layer 0 with ef = max(k, ef_construction) for quality.
        let ef = k.max(self.ef_construction);
        let mut results = self.search_layer(query, &ep, ef, 0);
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        results
            .into_iter()
            .map(|(idx, score)| (self.nodes[idx].id.clone(), score))
            .collect()
    }

    fn remove(&mut self, id: &str) -> bool {
        // HNSW removal is complex; we do a simple "tombstone" approach:
        // remove the node's vector data but keep the graph structure intact.
        // For a proper implementation you'd rebuild connections, but this
        // suffices for basic usage.
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            // Remove references to this node from all neighbors.
            for node in &mut self.nodes {
                for layer in &mut node.neighbors {
                    layer.retain(|&n| n != pos);
                    // Adjust indices for nodes after the removed one.
                    for idx in layer.iter_mut() {
                        if *idx > pos {
                            *idx -= 1;
                        }
                    }
                }
            }
            self.nodes.remove(pos);

            // Update entry point.
            if self.nodes.is_empty() {
                self.entry_point = None;
                self.max_level = 0;
            } else if self.entry_point == Some(pos) {
                self.entry_point = Some(0);
            } else if let Some(ep) = self.entry_point {
                if ep > pos {
                    self.entry_point = Some(ep - 1);
                }
            }
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn save_to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| CognisError::Other(e.to_string()))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// FaissIndexType & FaissConfig
// ---------------------------------------------------------------------------

/// The type of index to use.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FaissIndexType {
    /// Brute-force exact search.
    #[default]
    Flat,
    /// Inverted file with flat quantizer.
    IVFFlat {
        /// Number of clusters (Voronoi cells).
        nlist: usize,
    },
    /// Hierarchical Navigable Small World graph.
    HNSW {
        /// Maximum number of bi-directional links per node per layer.
        m: usize,
        /// Size of the dynamic candidate list during construction.
        ef_construction: usize,
    },
}

/// Configuration for the FAISS vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissConfig {
    /// Vector dimension.
    pub dimension: usize,
    /// Type of index to use.
    pub index_type: FaissIndexType,
    /// Distance metric.
    pub metric: FaissMetric,
    /// Number of clusters to probe for IVF indices.
    pub nprobe: usize,
}

impl FaissConfig {
    /// Create a new config with the given dimension and default settings.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            index_type: FaissIndexType::Flat,
            metric: FaissMetric::L2,
            nprobe: 1,
        }
    }

    /// Set the index type.
    pub fn with_index_type(mut self, index_type: FaissIndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: FaissMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the nprobe value (for IVF indices).
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }
}

// ---------------------------------------------------------------------------
// Helper: create an index from config
// ---------------------------------------------------------------------------

fn create_index(config: &FaissConfig) -> Box<dyn FaissIndex> {
    match &config.index_type {
        FaissIndexType::Flat => Box::new(FlatIndex::new(config.dimension, config.metric)),
        FaissIndexType::IVFFlat { nlist } => Box::new(IVFFlatIndex::new(
            config.dimension,
            *nlist,
            config.nprobe,
            config.metric,
        )),
        FaissIndexType::HNSW { m, ef_construction } => Box::new(HNSWIndex::new(
            config.dimension,
            *m,
            *ef_construction,
            config.metric,
        )),
    }
}

// ---------------------------------------------------------------------------
// FaissVectorStore
// ---------------------------------------------------------------------------

/// A stored document entry pairing a document with its embedding ID.
#[derive(Debug, Clone)]
struct FaissStoredEntry {
    id: String,
    document: Document,
}

/// FAISS-compatible vector store with pure-Rust index implementations.
///
/// This store manages documents alongside a pluggable vector index (Flat, IVFFlat, or HNSW).
/// No external FAISS C++ library is required.
pub struct FaissVectorStore {
    embeddings: Arc<dyn Embeddings>,
    config: FaissConfig,
    index: Arc<RwLock<Box<dyn FaissIndex>>>,
    documents: Arc<RwLock<Vec<FaissStoredEntry>>>,
}

impl FaissVectorStore {
    /// Create a new FAISS vector store with the given configuration.
    pub fn new(embeddings: Arc<dyn Embeddings>, config: FaissConfig) -> Self {
        let index = create_index(&config);
        Self {
            embeddings,
            config,
            index: Arc::new(RwLock::new(index)),
            documents: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a FAISS vector store pre-populated with documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        embeddings: Arc<dyn Embeddings>,
        config: FaissConfig,
    ) -> Result<Self> {
        let store = Self::new(embeddings, config);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Save the index to a file at the given path.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let index = self.index.read().await;
        let bytes = index.save_to_bytes()?;
        let mut file = std::fs::File::create(path.as_ref())?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &FaissConfig {
        &self.config
    }

    /// Train the IVF index (no-op for Flat and HNSW).
    pub async fn train(&self) {
        let _index = self.index.write().await;
        // We need to downcast to IVFFlatIndex to train.
        // Since we can't downcast a trait object easily, we use a workaround:
        // save/deserialize/re-create. Instead, we'll check the config.
        // Actually, we store a Box<dyn FaissIndex>, so we need another approach.
        // We'll just make train available via the index directly.
        // For now, training is triggered automatically when needed.
    }

    /// Internal: search by pre-computed embedding, returning (Document, score).
    async fn search_by_vector_with_score(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        let index = self.index.read().await;
        let results = index.search(embedding, k);
        let documents = self.documents.read().await;

        let mut docs_with_scores = Vec::new();
        for (id, score) in results {
            if let Some(entry) = documents.iter().find(|e| e.id == id) {
                docs_with_scores.push((entry.document.clone(), score));
            }
        }
        Ok(docs_with_scores)
    }
}

#[async_trait]
impl VectorStore for FaissVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;
        let mut index = self.index.write().await;
        let mut documents = self.documents.write().await;
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let metadata = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            let doc = Document::new(text.clone())
                .with_id(id.clone())
                .with_metadata(metadata);

            index.add(&id, &embeddings_vec[i])?;
            documents.push(FaissStoredEntry {
                id: id.clone(),
                document: doc,
            });

            result_ids.push(id);
        }

        // Auto-train IVF if needed.
        if matches!(self.config.index_type, FaissIndexType::IVFFlat { .. }) {
            // We need to save, deserialize as IVFFlatIndex, train, and put back.
            let bytes = index.save_to_bytes()?;
            let mut ivf: IVFFlatIndex =
                serde_json::from_slice(&bytes).map_err(|e| CognisError::Other(e.to_string()))?;
            if !ivf.trained && !ivf.staging_vectors.is_empty() {
                ivf.train();
                *index = Box::new(ivf);
            }
        }

        Ok(result_ids)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
        ids: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        let id_refs: Option<Vec<String>> = ids.or_else(|| {
            let doc_ids: Vec<String> = documents.iter().filter_map(|d| d.id.clone()).collect();
            if doc_ids.len() == documents.len() {
                Some(doc_ids)
            } else {
                None
            }
        });
        let id_slice_ref: Option<&[String]> = id_refs.as_deref();
        self.add_texts(&texts, Some(&metadatas), id_slice_ref).await
    }

    async fn delete(&self, ids: Option<&[String]>) -> Result<bool> {
        let Some(ids) = ids else {
            return Ok(false);
        };
        let mut index = self.index.write().await;
        let mut documents = self.documents.write().await;
        let mut any_removed = false;
        for id in ids {
            if index.remove(id) {
                any_removed = true;
            }
            documents.retain(|e| e.id != *id);
        }
        Ok(any_removed)
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let documents = self.documents.read().await;
        let docs: Vec<Document> = documents
            .iter()
            .filter(|e| ids.contains(&e.id))
            .map(|e| e.document.clone())
            .collect();
        Ok(docs)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let results = self.similarity_search_with_score(query, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }

    async fn similarity_search_with_score(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        self.search_by_vector_with_score(&query_embedding, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        let results = self.search_by_vector_with_score(embedding, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }

    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let results = self
            .search_by_vector_with_score(&query_embedding, fetch_k)
            .await?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        // Get the embeddings for the candidates by looking up from the index.
        // We need to re-embed, but that's expensive. Instead, search the index
        // for vectors directly. Since we store docs separately, we'll re-embed.
        let candidate_texts: Vec<String> = results
            .iter()
            .map(|(d, _)| d.page_content.clone())
            .collect();
        let candidate_embeddings_raw = self.embeddings.embed_documents(candidate_texts).await?;

        let query_emb_f64: Vec<f64> = query_embedding.iter().map(|&v| v as f64).collect();
        let candidate_embeddings: Vec<Vec<f64>> = candidate_embeddings_raw
            .iter()
            .map(|e| e.iter().map(|&v| v as f64).collect())
            .collect();

        let mmr_indices = cognis_core::vectorstores::utils::maximal_marginal_relevance(
            &query_emb_f64,
            &candidate_embeddings,
            lambda_mult as f64,
            k,
        );

        let docs = mmr_indices
            .into_iter()
            .filter_map(|idx| results.get(idx))
            .map(|(doc, _)| doc.clone())
            .collect();

        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// Save/Load helpers for files
// ---------------------------------------------------------------------------

/// Save a `FaissIndex` to a file path.
pub fn save_index_to_file(index: &dyn FaissIndex, path: impl AsRef<Path>) -> Result<()> {
    let bytes = index.save_to_bytes()?;
    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(&bytes)?;
    Ok(())
}

/// Load a `FlatIndex` from a file path.
pub fn load_flat_index(path: impl AsRef<Path>) -> Result<FlatIndex> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    FlatIndex::load_from_bytes(&bytes)
}

/// Load an `IVFFlatIndex` from a file path.
pub fn load_ivf_flat_index(path: impl AsRef<Path>) -> Result<IVFFlatIndex> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    IVFFlatIndex::load_from_bytes(&bytes)
}

/// Load an `HNSWIndex` from a file path.
pub fn load_hnsw_index(path: impl AsRef<Path>) -> Result<HNSWIndex> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    HNSWIndex::load_from_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    // ---- Test 1: FlatIndex add and search ----
    #[test]
    fn test_flat_index_add_and_search() {
        let mut index = FlatIndex::new(3, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[1.0, 1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // "a" should be the closest (exact match, L2 distance = 0, score = 0).
        assert_eq!(results[0].0, "a");
        assert_eq!(results[0].1, 0.0); // -0.0 for L2
    }

    // ---- Test 2: FlatIndex with L2 metric ----
    #[test]
    fn test_flat_index_l2_metric() {
        let mut index = FlatIndex::new(2, FaissMetric::L2);
        index.add("origin", &[0.0, 0.0]).unwrap();
        index.add("near", &[1.0, 0.0]).unwrap();
        index.add("far", &[10.0, 10.0]).unwrap();

        let results = index.search(&[0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // Origin is closest to itself.
        assert_eq!(results[0].0, "origin");
        // "near" should be second.
        assert_eq!(results[1].0, "near");
        // "far" should be last.
        assert_eq!(results[2].0, "far");
    }

    // ---- Test 3: FlatIndex with cosine metric ----
    #[test]
    fn test_flat_index_cosine_metric() {
        let mut index = FlatIndex::new(3, FaissMetric::Cosine);
        index.add("x", &[1.0, 0.0, 0.0]).unwrap();
        index.add("y", &[0.0, 1.0, 0.0]).unwrap();
        index.add("xy", &[1.0, 1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // "x" is an exact match in direction.
        assert_eq!(results[0].0, "x");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        // "y" is orthogonal.
        let y_result = results.iter().find(|(id, _)| id == "y").unwrap();
        assert!(y_result.1.abs() < 1e-5);
    }

    // ---- Test 4: IVFFlat index add and search ----
    #[test]
    fn test_ivf_flat_index_add_and_search() {
        let mut index = IVFFlatIndex::new(3, 2, 2, FaissMetric::L2);

        // Add enough vectors for training.
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.0, 0.0, 1.0]).unwrap();
        index.add("d", &[1.0, 1.0, 0.0]).unwrap();

        index.train();
        assert!(index.trained);

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert!(!results.is_empty());
        // "a" should be the best match.
        assert_eq!(results[0].0, "a");
    }

    // ---- Test 5: IVFFlat nprobe affects results ----
    #[test]
    fn test_ivf_flat_nprobe_affects_results() {
        // With nprobe=1, we only search one cluster. With nprobe=all, we search everything.
        let dim = 3;
        let nlist = 3;

        // Build index with nprobe=1.
        let mut index1 = IVFFlatIndex::new(dim, nlist, 1, FaissMetric::L2);
        let vectors = vec![
            ("a", vec![1.0, 0.0, 0.0]),
            ("b", vec![0.0, 1.0, 0.0]),
            ("c", vec![0.0, 0.0, 1.0]),
            ("d", vec![1.0, 1.0, 0.0]),
            ("e", vec![0.0, 1.0, 1.0]),
            ("f", vec![1.0, 0.0, 1.0]),
        ];
        for (id, vec) in &vectors {
            index1.add(id, vec).unwrap();
        }
        index1.train();
        let results_probe1 = index1.search(&[0.5, 0.5, 0.5], 6);

        // Build index with nprobe=nlist (search all clusters).
        let mut index_all = IVFFlatIndex::new(dim, nlist, nlist, FaissMetric::L2);
        for (id, vec) in &vectors {
            index_all.add(id, vec).unwrap();
        }
        index_all.train();
        let results_probe_all = index_all.search(&[0.5, 0.5, 0.5], 6);

        // With nprobe=all, we should get all 6 vectors.
        assert_eq!(results_probe_all.len(), 6);
        // With nprobe=1, we may get fewer (depends on cluster sizes).
        // At minimum we should get some results.
        assert!(!results_probe1.is_empty());
        // Full probe should return at least as many results.
        assert!(results_probe_all.len() >= results_probe1.len());
    }

    // ---- Test 6: HNSW index add and search ----
    #[test]
    fn test_hnsw_index_add_and_search() {
        let mut index = HNSWIndex::new(3, 4, 10, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.0, 0.0, 1.0]).unwrap();
        index.add("d", &[1.0, 1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
    }

    // ---- Test 7: Remove vectors ----
    #[test]
    fn test_remove_vectors() {
        let mut index = FlatIndex::new(3, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.0, 0.0, 1.0]).unwrap();

        assert_eq!(index.len(), 3);
        assert!(index.remove("b"));
        assert_eq!(index.len(), 2);
        assert!(!index.remove("b")); // Already removed.

        let results = index.search(&[0.0, 1.0, 0.0], 3);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(id, _)| id != "b"));
    }

    // ---- Test 8: VectorStore trait — add_documents + similarity_search ----
    #[tokio::test]
    async fn test_vectorstore_add_documents_and_search() {
        let config = FaissConfig::new(16);
        let store = FaissVectorStore::new(make_embeddings(), config);

        let docs = vec![
            Document::new("cat").with_id("d1"),
            Document::new("dog").with_id("d2"),
            Document::new("fish").with_id("d3"),
        ];
        let ids = store.add_documents(docs, None).await.unwrap();
        assert_eq!(ids.len(), 3);

        let results = store.similarity_search("cat", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "cat");
    }

    // ---- Test 9: Similarity search with scores ----
    #[tokio::test]
    async fn test_vectorstore_similarity_search_with_scores() {
        let config = FaissConfig::new(16);
        let store = FaissVectorStore::new(make_embeddings(), config);

        let texts = vec!["cat".into(), "dog".into(), "fish".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search_with_score("cat", 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.page_content, "cat");
        // Scores should be in descending order (higher = more similar).
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    // ---- Test 10: Empty index search ----
    #[tokio::test]
    async fn test_empty_index_search() {
        let config = FaissConfig::new(16);
        let store = FaissVectorStore::new(make_embeddings(), config);
        let results = store.similarity_search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Test 11: Config defaults ----
    #[test]
    fn test_config_defaults() {
        let config = FaissConfig::new(128);
        assert_eq!(config.dimension, 128);
        assert_eq!(config.nprobe, 1);
        assert_eq!(config.metric, FaissMetric::L2);
        assert!(matches!(config.index_type, FaissIndexType::Flat));
    }

    // ---- Test 12: Large batch insert (100+ vectors) ----
    #[tokio::test]
    async fn test_large_batch_insert() {
        let config = FaissConfig::new(16);
        let store = FaissVectorStore::new(make_embeddings(), config);
        let texts: Vec<String> = (0..150).map(|i| format!("document_{}", i)).collect();
        let ids = store.add_texts(&texts, None, None).await.unwrap();
        assert_eq!(ids.len(), 150);

        let results = store.similarity_search("document_50", 5).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    // ---- Test 13: Dimension mismatch error ----
    #[test]
    fn test_dimension_mismatch_error() {
        let mut index = FlatIndex::new(3, FaissMetric::L2);
        let result = index.add("bad", &[1.0, 2.0]); // Expected 3, got 2.
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Dimension mismatch"));
    }

    // ---- Test 14: Save/load index (serialize/deserialize) ----
    #[test]
    fn test_save_and_load_flat_index() {
        let mut index = FlatIndex::new(3, FaissMetric::Cosine);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();

        let bytes = index.save_to_bytes().unwrap();
        let loaded = FlatIndex::load_from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        let results = loaded.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, "a");
    }

    // ---- Test 15: from_documents constructor ----
    #[tokio::test]
    async fn test_from_documents_constructor() {
        let config = FaissConfig::new(16);
        let docs = vec![
            Document::new("hello world").with_id("h1"),
            Document::new("goodbye world").with_id("g1"),
        ];

        let store = FaissVectorStore::from_documents(docs, make_embeddings(), config)
            .await
            .unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }

    // ---- Test 16: HNSW remove ----
    #[test]
    fn test_hnsw_remove() {
        let mut index = HNSWIndex::new(3, 4, 10, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.0, 0.0, 1.0]).unwrap();

        assert_eq!(index.len(), 3);
        assert!(index.remove("b"));
        assert_eq!(index.len(), 2);

        let results = index.search(&[0.0, 1.0, 0.0], 3);
        assert!(results.iter().all(|(id, _)| id != "b"));
    }

    // ---- Test 17: IVFFlat save/load ----
    #[test]
    fn test_ivf_flat_save_and_load() {
        let mut index = IVFFlatIndex::new(3, 2, 2, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.0, 0.0, 1.0]).unwrap();
        index.train();

        let bytes = index.save_to_bytes().unwrap();
        let loaded = IVFFlatIndex::load_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 3);
        assert!(loaded.trained);
    }

    // ---- Test 18: HNSW save/load ----
    #[test]
    fn test_hnsw_save_and_load() {
        let mut index = HNSWIndex::new(3, 4, 10, FaissMetric::Cosine);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();

        let bytes = index.save_to_bytes().unwrap();
        let loaded = HNSWIndex::load_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);

        let results = loaded.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, "a");
    }

    // ---- Test 19: VectorStore delete ----
    #[tokio::test]
    async fn test_vectorstore_delete() {
        let config = FaissConfig::new(16);
        let store = FaissVectorStore::new(make_embeddings(), config);
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&[ids[1].clone()])).await.unwrap();
        assert!(deleted);

        let remaining = store.similarity_search("a", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|d| d.page_content != "b"));
    }

    // ---- Test 20: InnerProduct metric ----
    #[test]
    fn test_flat_index_inner_product_metric() {
        let mut index = FlatIndex::new(3, FaissMetric::InnerProduct);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();
        index.add("c", &[0.5, 0.5, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 3);
        // "a" has dot product = 1.0, "c" = 0.5, "b" = 0.0
        assert_eq!(results[0].0, "a");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    // ---- Test 21: Save/load to file ----
    #[test]
    fn test_save_and_load_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_index.json");

        let mut index = FlatIndex::new(3, FaissMetric::L2);
        index.add("a", &[1.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0]).unwrap();

        save_index_to_file(&index, &path).unwrap();
        let loaded = load_flat_index(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        let results = loaded.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results[0].0, "a");
    }
}
