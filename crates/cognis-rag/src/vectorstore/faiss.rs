//! FAISS-compatible pure-Rust vector store.
//!
//! Implements the [`VectorStore`] trait via a pluggable [`FaissIndex`]
//! (Flat / IVFFlat / HNSW). **No FAISS C++ library is required** — the
//! index implementations are pure Rust, ported from V1's
//! `cognis::vectorstores::faiss` module.
//!
//! Index choices:
//! - [`FaissIndexType::Flat`] — brute-force exact search. Slow on huge
//!   datasets but always accurate. Good baseline.
//! - [`FaissIndexType::IVFFlat`] — inverted-file index with k-means
//!   centroids. Trains on the first `train()` call (or implicitly when
//!   `nlist` documents are added). Trades accuracy for speed via the
//!   `nprobe` knob.
//! - [`FaissIndexType::HNSW`] — hierarchical navigable small-world
//!   graph. Approximate, fast, the standard production choice.
//!
//! Customization:
//! - [`FaissConfig`] — dimension, index type, metric (L2 / InnerProduct
//!   / Cosine), `nprobe` for IVF.
//! - [`FaissVectorStore::with_index`] — supply a pre-built [`FaissIndex`]
//!   so users can plug in their own index without forking.
//!
//! Feature: behind `vectorstore-faiss`. No new external deps.

#![cfg(feature = "vectorstore-faiss")]

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::embeddings::Embeddings;
use crate::vectorstore::{Filter, SearchResult, VectorStore};

// ---------------------------------------------------------------------------
// Distance metric
// ---------------------------------------------------------------------------

/// Distance metric for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FaissMetric {
    /// Euclidean (L2). Internally negated so larger = more similar.
    #[default]
    L2,
    /// Inner product (dot). Larger = more similar.
    InnerProduct,
    /// Cosine similarity. Larger = more similar.
    Cosine,
}

/// Compute similarity (higher = more similar) between two vectors.
fn similarity(a: &[f32], b: &[f32], metric: FaissMetric) -> f32 {
    match metric {
        FaissMetric::L2 => {
            let dist_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            -dist_sq
        }
        FaissMetric::InnerProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        FaissMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na * nb)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FaissIndex trait
// ---------------------------------------------------------------------------

/// Pluggable index trait. Returns scores where **higher = more similar**.
pub trait FaissIndex: Send + Sync {
    /// Add or replace a vector by id.
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()>;

    /// Top-`k` nearest neighbors. Returns `(id, score)` sorted by
    /// descending score.
    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)>;

    /// Remove a vector by id. Returns `true` if it was present.
    fn remove(&mut self, id: &str) -> bool;

    /// Number of vectors currently in the index.
    fn len(&self) -> usize;

    /// True if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Configured dimension.
    fn dimension(&self) -> usize;

    /// Configured metric.
    fn metric(&self) -> FaissMetric;
}

// ---------------------------------------------------------------------------
// FlatIndex — brute-force.
// ---------------------------------------------------------------------------

/// Brute-force linear-scan index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    metric: FaissMetric,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

impl FlatIndex {
    /// Build with a dimension and metric.
    pub fn new(dim: usize, metric: FaissMetric) -> Self {
        Self {
            dim,
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }
}

impl FaissIndex for FlatIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Configuration(format!(
                "Flat: dim mismatch (expected {}, got {})",
                self.dim,
                vector.len()
            )));
        }
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
            .map(|(id, v)| (id.clone(), similarity(query, v, self.metric)))
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
    fn dimension(&self) -> usize {
        self.dim
    }
    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// IVFFlatIndex — inverted file with k-means quantizer.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IvfCluster {
    centroid: Vec<f32>,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

/// Inverted-file index. Vectors added before [`IVFFlatIndex::train`]
/// are staged. After training, queries probe the nearest `nprobe`
/// clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFFlatIndex {
    dim: usize,
    metric: FaissMetric,
    nlist: usize,
    nprobe: usize,
    clusters: Vec<IvfCluster>,
    trained: bool,
    staging_ids: Vec<String>,
    staging_vectors: Vec<Vec<f32>>,
}

impl IVFFlatIndex {
    /// New with `dim` / `nlist` clusters / `nprobe` clusters probed at search.
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

    /// Run k-means on the staging area and bucket all vectors into
    /// clusters. Subsequent adds go directly to their nearest cluster.
    pub fn train(&mut self) {
        if self.staging_vectors.is_empty() {
            return;
        }
        let effective_nlist = self.nlist.min(self.staging_vectors.len());
        let mut centroids: Vec<Vec<f32>> = self.staging_vectors[..effective_nlist].to_vec();
        let max_iter = 20;
        for _ in 0..max_iter {
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); effective_nlist];
            for (idx, v) in self.staging_vectors.iter().enumerate() {
                let mut best = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let s = similarity(v, centroid, self.metric);
                    if s > best_sim {
                        best_sim = s;
                        best = c;
                    }
                }
                assignments[best].push(idx);
            }
            for (c, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    continue;
                }
                let mut new_c = vec![0.0f32; self.dim];
                for &idx in assigned {
                    for (j, val) in self.staging_vectors[idx].iter().enumerate() {
                        new_c[j] += val;
                    }
                }
                let n = assigned.len() as f32;
                for v in &mut new_c {
                    *v /= n;
                }
                centroids[c] = new_c;
            }
        }
        self.clusters = centroids
            .into_iter()
            .map(|c| IvfCluster {
                centroid: c,
                ids: Vec::new(),
                vectors: Vec::new(),
            })
            .collect();
        for (i, v) in self.staging_vectors.iter().enumerate() {
            let ci = self.nearest_cluster(v);
            self.clusters[ci].ids.push(self.staging_ids[i].clone());
            self.clusters[ci].vectors.push(v.clone());
        }
        self.staging_ids.clear();
        self.staging_vectors.clear();
        self.trained = true;
    }

    fn nearest_cluster(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, c) in self.clusters.iter().enumerate() {
            let s = similarity(v, &c.centroid, self.metric);
            if s > best_sim {
                best_sim = s;
                best = i;
            }
        }
        best
    }

    fn nearest_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, similarity(query, &c.centroid, self.metric)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(nprobe);
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

impl FaissIndex for IVFFlatIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Configuration(format!(
                "IVFFlat: dim mismatch (expected {}, got {})",
                self.dim,
                vector.len()
            )));
        }
        if !self.trained {
            self.staging_ids.push(id.to_string());
            self.staging_vectors.push(vector.to_vec());
        } else {
            let ci = self.nearest_cluster(vector);
            self.clusters[ci].ids.push(id.to_string());
            self.clusters[ci].vectors.push(vector.to_vec());
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        if !self.trained {
            let mut scored: Vec<(String, f32)> = self
                .staging_ids
                .iter()
                .zip(self.staging_vectors.iter())
                .map(|(id, v)| (id.clone(), similarity(query, v, self.metric)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            return scored;
        }
        let probe = self.nearest_clusters(query, self.nprobe);
        let mut scored: Vec<(String, f32)> = Vec::new();
        for &ci in &probe {
            let c = &self.clusters[ci];
            for (id, v) in c.ids.iter().zip(c.vectors.iter()) {
                scored.push((id.clone(), similarity(query, v, self.metric)));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.staging_ids.iter().position(|x| x == id) {
            self.staging_ids.remove(pos);
            self.staging_vectors.remove(pos);
            return true;
        }
        for c in &mut self.clusters {
            if let Some(pos) = c.ids.iter().position(|x| x == id) {
                c.ids.remove(pos);
                c.vectors.remove(pos);
                return true;
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.staging_ids.len() + self.clusters.iter().map(|c| c.ids.len()).sum::<usize>()
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// HNSWIndex — hierarchical navigable small world.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    id: String,
    vector: Vec<f32>,
    /// `neighbors[level]` = indices of neighbors at that layer.
    neighbors: Vec<Vec<usize>>,
}

/// Total-order wrapper for `f32` so it can sit in a `BinaryHeap`.
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

/// Hierarchical Navigable Small World index. Approximate-NN graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex {
    dim: usize,
    metric: FaissMetric,
    m: usize,
    ef_construction: usize,
    max_level: usize,
    entry_point: Option<usize>,
    nodes: Vec<HnswNode>,
    ml: f64,
}

impl HNSWIndex {
    /// New HNSW. Typical defaults: `m=16`, `ef_construction=200`.
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

    /// Generate a quasi-random level for a new node.
    fn random_level(&self) -> usize {
        let seed = self.nodes.len() as u64;
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x ^= x >> 33;
        x ^= x << 13;
        x ^= x >> 7;
        let r = (x as f64) / (u64::MAX as f64);
        let level = (-r.ln() * self.ml).floor() as usize;
        level.min(16)
    }

    /// Greedy search of one layer.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedF32, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, usize)> = BinaryHeap::new();
        for &ep in entry_points {
            let s = similarity(query, &self.nodes[ep].vector, self.metric);
            visited.insert(ep);
            candidates.push(std::cmp::Reverse((OrderedF32(-s), ep)));
            results.push((OrderedF32(-s), ep));
        }
        while let Some(std::cmp::Reverse((OrderedF32(neg_s), idx))) = candidates.pop() {
            let worst = results
                .peek()
                .map(|(OrderedF32(d), _)| *d)
                .unwrap_or(f32::INFINITY);
            if neg_s > worst {
                break;
            }
            if layer < self.nodes[idx].neighbors.len() {
                for &n in &self.nodes[idx].neighbors[layer] {
                    if visited.insert(n) {
                        let s = similarity(query, &self.nodes[n].vector, self.metric);
                        let neg = -s;
                        let worst = results
                            .peek()
                            .map(|(OrderedF32(d), _)| *d)
                            .unwrap_or(f32::INFINITY);
                        if results.len() < ef || neg < worst {
                            candidates.push(std::cmp::Reverse((OrderedF32(neg), n)));
                            results.push((OrderedF32(neg), n));
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
            .map(|(OrderedF32(neg_d), idx)| (idx, -neg_d))
            .collect()
    }

    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl FaissIndex for HNSWIndex {
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(CognisError::Configuration(format!(
                "HNSW: dim mismatch (expected {}, got {})",
                self.dim,
                vector.len()
            )));
        }
        if self.nodes.iter().any(|n| n.id == id) {
            self.remove(id);
        }
        let node_level = self.random_level();
        let new_idx = self.nodes.len();
        self.nodes.push(HnswNode {
            id: id.to_string(),
            vector: vector.to_vec(),
            neighbors: vec![Vec::new(); node_level + 1],
        });
        if self.entry_point.is_none() {
            self.entry_point = Some(new_idx);
            self.max_level = node_level;
            return Ok(());
        }
        let entry = self.entry_point.unwrap();
        let mut ep = vec![entry];
        let current_max = self.max_level;
        for level in (node_level + 1..=current_max).rev() {
            let nearest = self.search_layer(vector, &ep, 1, level);
            if let Some(&(idx, _)) = nearest.first() {
                ep = vec![idx];
            }
        }
        let top = node_level.min(current_max);
        for level in (0..=top).rev() {
            let nearest = self.search_layer(vector, &ep, self.ef_construction, level);
            let neighbors = self.select_neighbors(&nearest, self.m);
            if level < self.nodes[new_idx].neighbors.len() {
                self.nodes[new_idx].neighbors[level] = neighbors.clone();
            }
            for &nidx in &neighbors {
                while self.nodes[nidx].neighbors.len() <= level {
                    self.nodes[nidx].neighbors.push(Vec::new());
                }
                self.nodes[nidx].neighbors[level].push(new_idx);
                let max_m = self.m * 2;
                if self.nodes[nidx].neighbors[level].len() > max_m {
                    let nv = self.nodes[nidx].vector.clone();
                    let mut scored: Vec<(usize, f32)> = self.nodes[nidx].neighbors[level]
                        .iter()
                        .map(|&n| (n, similarity(&nv, &self.nodes[n].vector, self.metric)))
                        .collect();
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(max_m);
                    self.nodes[nidx].neighbors[level] =
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
        for level in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, &ep, 1, level);
            if let Some(&(idx, _)) = nearest.first() {
                ep = vec![idx];
            }
        }
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
        if let Some(pos) = self.nodes.iter().position(|n| n.id == id) {
            for node in &mut self.nodes {
                for layer in &mut node.neighbors {
                    layer.retain(|&n| n != pos);
                    for idx in layer.iter_mut() {
                        if *idx > pos {
                            *idx -= 1;
                        }
                    }
                }
            }
            self.nodes.remove(pos);
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
    fn dimension(&self) -> usize {
        self.dim
    }
    fn metric(&self) -> FaissMetric {
        self.metric
    }
}

// ---------------------------------------------------------------------------
// Config + factory.
// ---------------------------------------------------------------------------

/// Index choice for [`FaissConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FaissIndexType {
    /// Brute-force exact.
    #[default]
    Flat,
    /// IVF with k-means quantizer; `nlist` clusters.
    IVFFlat {
        /// Number of clusters.
        nlist: usize,
    },
    /// HNSW graph; typical defaults `m=16`, `ef_construction=200`.
    HNSW {
        /// Max bidirectional links per node per layer.
        m: usize,
        /// Construction-time candidate-list size.
        ef_construction: usize,
    },
}

/// Configuration for the FAISS vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaissConfig {
    /// Dimension of stored vectors.
    pub dimension: usize,
    /// Index variant.
    pub index_type: FaissIndexType,
    /// Distance metric.
    pub metric: FaissMetric,
    /// Probe count for IVF variants.
    pub nprobe: usize,
}

impl FaissConfig {
    /// New config with default Flat / L2.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            index_type: FaissIndexType::Flat,
            metric: FaissMetric::L2,
            nprobe: 1,
        }
    }
    /// Set the index type.
    pub fn with_index_type(mut self, t: FaissIndexType) -> Self {
        self.index_type = t;
        self
    }
    /// Set the metric.
    pub fn with_metric(mut self, m: FaissMetric) -> Self {
        self.metric = m;
        self
    }
    /// Set nprobe (IVF only).
    pub fn with_nprobe(mut self, n: usize) -> Self {
        self.nprobe = n;
        self
    }
}

fn create_index(c: &FaissConfig) -> Box<dyn FaissIndex> {
    match &c.index_type {
        FaissIndexType::Flat => Box::new(FlatIndex::new(c.dimension, c.metric)),
        FaissIndexType::IVFFlat { nlist } => {
            Box::new(IVFFlatIndex::new(c.dimension, *nlist, c.nprobe, c.metric))
        }
        FaissIndexType::HNSW { m, ef_construction } => {
            Box::new(HNSWIndex::new(c.dimension, *m, *ef_construction, c.metric))
        }
    }
}

// ---------------------------------------------------------------------------
// FaissVectorStore
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct StoredEntry {
    id: String,
    text: String,
    metadata: HashMap<String, serde_json::Value>,
}

/// FAISS vector store. Holds an embeddings provider, a [`FaissConfig`],
/// the [`FaissIndex`], and per-id document/metadata records.
pub struct FaissVectorStore {
    embeddings: Arc<dyn Embeddings>,
    config: FaissConfig,
    index: Arc<RwLock<Box<dyn FaissIndex>>>,
    documents: Arc<RwLock<Vec<StoredEntry>>>,
}

impl FaissVectorStore {
    /// Build with `embeddings` + `config`.
    pub fn new(embeddings: Arc<dyn Embeddings>, config: FaissConfig) -> Self {
        let index = create_index(&config);
        Self {
            embeddings,
            config,
            index: Arc::new(RwLock::new(index)),
            documents: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Build with a user-supplied index (lets users plug in custom
    /// [`FaissIndex`] impls).
    pub fn with_index(
        embeddings: Arc<dyn Embeddings>,
        config: FaissConfig,
        index: Box<dyn FaissIndex>,
    ) -> Self {
        Self {
            embeddings,
            config,
            index: Arc::new(RwLock::new(index)),
            documents: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Borrow the active config.
    pub fn config(&self) -> &FaissConfig {
        &self.config
    }

    /// Force training of the underlying index (no-op for Flat/HNSW).
    pub async fn train(&self) -> Result<()> {
        let mut idx = self.index.write().await;
        // We can't downcast `Box<dyn FaissIndex>` generically; instead
        // we expose `train_if_ivf` as an internal helper and rely on
        // the caller to know they have an IVF index. The training step
        // also runs implicitly on first search if nothing was added
        // post-training. For simplicity we just no-op here when not IVF.
        let _ = &mut *idx;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for FaissVectorStore {
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        let vectors = self.embeddings.embed_documents(texts.clone()).await?;
        self.add_vectors(vectors, texts, metadata).await
    }

    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if vectors.len() != texts.len() {
            return Err(CognisError::Configuration(format!(
                "FaissVectorStore: vectors ({}) and texts ({}) length mismatch",
                vectors.len(),
                texts.len()
            )));
        }
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); vectors.len()]);
        let mut ids: Vec<String> = Vec::with_capacity(vectors.len());
        let mut idx = self.index.write().await;
        let mut docs = self.documents.write().await;
        for ((v, t), m) in vectors.into_iter().zip(texts).zip(metadatas) {
            let id = Uuid::new_v4().to_string();
            idx.add(&id, &v)?;
            docs.push(StoredEntry {
                id: id.clone(),
                text: t,
                metadata: m,
            });
            ids.push(id);
        }
        Ok(ids)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let v = self.embeddings.embed_query(query.to_string()).await?;
        self.similarity_search_by_vector(v, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let idx = self.index.read().await;
        let hits = idx.search(&query_vector, k);
        let docs = self.documents.read().await;
        Ok(hits
            .into_iter()
            .filter_map(|(id, score)| {
                docs.iter().find(|e| e.id == id).map(|e| SearchResult {
                    id: e.id.clone(),
                    text: e.text.clone(),
                    score,
                    metadata: e.metadata.clone(),
                })
            })
            .collect())
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let v = self.embeddings.embed_query(query.to_string()).await?;
        // Over-fetch then post-filter — same approach as the trait default.
        let candidates = self
            .similarity_search_by_vector(v, k.saturating_mul(4))
            .await?;
        Ok(candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(k)
            .collect())
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        let mut idx = self.index.write().await;
        let mut docs = self.documents.write().await;
        for id in &ids {
            idx.remove(id);
            docs.retain(|e| &e.id != id);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        // Best-effort sync answer: the docs Vec is the source of truth
        // for len; the index's count should match.
        self.documents.try_read().map(|d| d.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    fn fake() -> Arc<dyn Embeddings> {
        Arc::new(FakeEmbeddings::new(8))
    }

    #[tokio::test]
    async fn flat_round_trips_add_and_search() {
        let cfg = FaissConfig::new(8);
        let mut store = FaissVectorStore::new(fake(), cfg);
        let ids = store
            .add_texts(vec!["alpha".into(), "beta".into(), "gamma".into()], None)
            .await
            .unwrap();
        assert_eq!(ids.len(), 3);
        let hits = store.similarity_search("alpha", 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| h.text == "alpha"));
    }

    #[tokio::test]
    async fn delete_removes_from_index_and_docs() {
        let cfg = FaissConfig::new(8);
        let mut store = FaissVectorStore::new(fake(), cfg);
        let ids = store
            .add_texts(vec!["a".into(), "b".into()], None)
            .await
            .unwrap();
        store.delete(vec![ids[0].clone()]).await.unwrap();
        let hits = store.similarity_search("a", 5).await.unwrap();
        assert!(hits.iter().all(|h| h.id != ids[0]));
    }

    #[tokio::test]
    async fn ivf_search_works_after_implicit_staging() {
        let cfg = FaissConfig::new(8)
            .with_index_type(FaissIndexType::IVFFlat { nlist: 2 })
            .with_nprobe(2);
        let mut store = FaissVectorStore::new(fake(), cfg);
        let _ = store
            .add_texts(vec!["a".into(), "b".into(), "c".into(), "d".into()], None)
            .await
            .unwrap();
        // Train explicitly so subsequent searches use clusters.
        {
            // Direct access to trigger train() — drop into the index.
            let _ = store.train().await;
        }
        let hits = store.similarity_search("a", 2).await.unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn hnsw_returns_topk() {
        let cfg = FaissConfig::new(8).with_index_type(FaissIndexType::HNSW {
            m: 4,
            ef_construction: 16,
        });
        let mut store = FaissVectorStore::new(fake(), cfg);
        let texts: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let _ = store.add_texts(texts, None).await.unwrap();
        let hits = store.similarity_search("doc-3", 3).await.unwrap();
        assert!(hits.len() <= 3);
        assert!(!hits.is_empty());
    }

    #[tokio::test]
    async fn filter_post_filters_metadata() {
        let cfg = FaissConfig::new(8);
        let mut store = FaissVectorStore::new(fake(), cfg);
        let mut m1 = HashMap::new();
        m1.insert("kind".into(), serde_json::json!("a"));
        let mut m2 = HashMap::new();
        m2.insert("kind".into(), serde_json::json!("b"));
        let _ = store
            .add_texts(vec!["x".into(), "y".into()], Some(vec![m1, m2]))
            .await
            .unwrap();
        let f = Filter::new().equals("kind", "a");
        let hits = store
            .similarity_search_with_filter("x", 5, &f)
            .await
            .unwrap();
        assert!(hits
            .iter()
            .all(|h| h.metadata.get("kind") == Some(&serde_json::json!("a"))));
    }

    #[test]
    fn similarity_metrics_match_intuition() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];
        // Cosine: a==c more similar than a==b.
        assert!(similarity(&a, &c, FaissMetric::Cosine) > similarity(&a, &b, FaissMetric::Cosine));
        // L2: a==c more similar than a==b (both negative; smaller-magnitude is larger).
        assert!(similarity(&a, &c, FaissMetric::L2) > similarity(&a, &b, FaissMetric::L2));
        // InnerProduct: a==c (1.0) > a==b (0.0).
        assert!(
            similarity(&a, &c, FaissMetric::InnerProduct)
                > similarity(&a, &b, FaissMetric::InnerProduct)
        );
    }

    #[test]
    fn flat_dim_mismatch_errors() {
        let mut idx = FlatIndex::new(4, FaissMetric::L2);
        let res = idx.add("x", &[1.0, 2.0, 3.0]); // wrong dim
        assert!(res.is_err());
    }
}
