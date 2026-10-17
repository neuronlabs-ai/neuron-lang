/// NEURON Memory allocator — arena-based, tensor-aware.
///
/// Allocates tensor data in arenas for cache-friendly access.
/// Gradient buffers are allocated adjacent to their forward tensors.

use std::alloc::{alloc, dealloc, Layout};

/// Arena allocator for tensor data.
pub struct Arena {
    blocks: Vec<ArenaBlock>,
    current_block: usize,
    offset: usize,
    block_size: usize,
}

struct ArenaBlock {
    ptr: *mut u8,
    layout: Layout,
    capacity: usize,
}

impl Arena {
    /// Create a new arena with the given block size (default 4MB).
    pub fn new(block_size: usize) -> Self {
        let mut arena = Self {
            blocks: Vec::new(),
            current_block: 0,
            offset: 0,
            block_size,
        };
        arena.alloc_block();
        arena
    }

    pub fn default() -> Self {
        Self::new(4 * 1024 * 1024) // 4MB blocks
    }

    fn alloc_block(&mut self) {
        let layout = Layout::from_size_align(self.block_size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Arena: out of memory (failed to allocate {} bytes)", self.block_size);
        }
        self.blocks.push(ArenaBlock { ptr, layout, capacity: self.block_size });
        self.current_block = self.blocks.len() - 1;
        self.offset = 0;
    }

    /// Allocate `n` f64 values from the arena. Returns a mutable slice.
    pub fn alloc_f64(&mut self, n: usize) -> &mut [f64] {
        let byte_size = n * std::mem::size_of::<f64>();
        let align = std::mem::align_of::<f64>();

        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + byte_size > self.block_size {
            // Need a new block
            if byte_size > self.block_size {
                // Oversized allocation — create a custom block
                let layout = Layout::from_size_align(byte_size, 64).unwrap();
                let ptr = unsafe { alloc(layout) };
                if ptr.is_null() { panic!("Arena: out of memory"); }
                self.blocks.push(ArenaBlock { ptr, layout, capacity: byte_size });
                return unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, n) };
            }
            self.alloc_block();
            return self.alloc_f64(n);
        }

        let block = &self.blocks[self.current_block];
        let ptr = unsafe { block.ptr.add(aligned_offset) as *mut f64 };
        self.offset = aligned_offset + byte_size;
        unsafe { std::slice::from_raw_parts_mut(ptr, n) }
    }

    /// Total bytes allocated across all blocks.
    pub fn total_allocated(&self) -> usize {
        self.blocks.iter().map(|b| b.capacity).sum()
    }

    /// Reset the arena — all previous allocations are invalidated.
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.offset = 0;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for block in &self.blocks {
            unsafe { dealloc(block.ptr, block.layout); }
        }
    }
}

use crate::tensor::Tensor;
use crate::vm::Value;

/// Differentiable Working Memory (Neural Turing Machine / DNC architecture).
/// Maintains a fixed set of memory slots with content-based addressing and gated erase/write.
#[derive(Clone, Debug)]
pub struct WorkingMemoryData {
    pub dim: usize,
    pub slots: usize,
    /// Memory matrix: shape [slots, dim]
    pub matrix: Tensor,
    pub write_head: usize,
}

impl WorkingMemoryData {
    pub fn new(dim: usize, slots: usize) -> Self {
        Self {
            dim,
            slots,
            matrix: Tensor::zeros(&[slots, dim]),
            write_head: 0,
        }
    }

    /// Content-based read via scaled dot-product attention over memory slots.
    /// Returns attended memory vector of shape [1, dim].
    pub fn read(&self, query: &Tensor) -> Tensor {
        let q_len = query.data.len();
        if q_len != self.dim {
            return Tensor::zeros(&[1, self.dim]);
        }
        let scale = 1.0 / (self.dim as f64).sqrt().max(1.0);
        let mut scores = Vec::with_capacity(self.slots);
        let mut max_score = f64::NEG_INFINITY;

        for s in 0..self.slots {
            let offset = s * self.dim;
            let mut dot = 0.0;
            for j in 0..self.dim {
                dot += query.data[j] * self.matrix.data[offset + j];
            }
            let scaled = dot * scale;
            if scaled > max_score {
                max_score = scaled;
            }
            scores.push(scaled);
        }

        // Softmax over slot similarities
        let mut sum_exp = 0.0;
        let mut attn = Vec::with_capacity(self.slots);
        for &sc in &scores {
            let e = (sc - max_score).exp();
            attn.push(e);
            sum_exp += e;
        }
        let inv_sum = if sum_exp > 1e-12 { 1.0 / sum_exp } else { 1.0 / (self.slots as f64) };
        for a in attn.iter_mut() {
            *a *= inv_sum;
        }

        // Weighted read vector
        let mut read_data = vec![0.0; self.dim];
        for s in 0..self.slots {
            let weight = attn[s];
            let offset = s * self.dim;
            for j in 0..self.dim {
                read_data[j] += weight * self.matrix.data[offset + j];
            }
        }

        Tensor::new(crate::buffer::Buffer::from(read_data), vec![1, self.dim])
    }

    /// Simple write to the current write head position.
    pub fn write(&mut self, content: &Tensor) {
        if content.data.len() != self.dim {
            return;
        }
        let s = self.write_head % self.slots;
        let offset = s * self.dim;
        for j in 0..self.dim {
            self.matrix.data[offset + j] = content.data[j];
        }
        self.write_head = (self.write_head + 1) % self.slots;
    }

    /// DNC-style gated write:
    /// M[s, j] = M[s, j] * (1 - write_attn[s] * erase[j]) + write_attn[s] * content[j]
    pub fn write_gated(&mut self, content: &Tensor, write_attn: &Tensor, erase_gate: &Tensor) {
        let content_data = &content.data;
        let erase_data = &erase_gate.data;
        let write_data = &write_attn.data;

        for s in 0..self.slots {
            let w = if s < write_data.len() { write_data[s] } else { 1.0 / (self.slots as f64) };
            let offset = s * self.dim;
            for j in 0..self.dim {
                let e = if j < erase_data.len() { erase_data[j] } else { 0.0 };
                let c = if j < content_data.len() { content_data[j] } else { 0.0 };
                let cur = self.matrix.data[offset + j];
                self.matrix.data[offset + j] = cur * (1.0 - w * e) + w * c;
            }
        }
    }

    pub fn clear(&mut self) {
        for x in self.matrix.data.iter_mut() {
            *x = 0.0;
        }
        self.write_head = 0;
    }

    pub fn get_slots(&self) -> Tensor {
        self.matrix.clone()
    }
}

/// A stored episodic memory entry.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub key: Tensor,
    pub value: Value,
    pub timestamp: usize,
    pub access_count: f64,
}

/// Episodic Memory with similarity-based recall and Ebbinghaus forgetting curves.
#[derive(Clone, Debug)]
pub struct EpisodicMemoryData {
    pub dim: usize,
    pub capacity: usize,
    pub decay_rate: f64,
    pub current_time: usize,
    pub entries: Vec<MemoryEntry>,
}

impl EpisodicMemoryData {
    pub fn new(dim: usize, capacity: usize, decay_rate: Option<f64>) -> Self {
        Self {
            dim,
            capacity,
            decay_rate: decay_rate.unwrap_or(0.01),
            current_time: 0,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn store(&mut self, key: &Tensor, value: Value) {
        self.current_time += 1;
        let entry = MemoryEntry {
            key: key.clone(),
            value,
            timestamp: self.current_time,
            access_count: 1.0,
        };

        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            // Find least important memory by Ebbinghaus curve:
            // importance = access_count * exp(-decay * (now - t))
            let mut min_idx = 0;
            let mut min_imp = f64::INFINITY;
            for (idx, e) in self.entries.iter().enumerate() {
                let dt = (self.current_time - e.timestamp) as f64;
                let imp = e.access_count * (-self.decay_rate * dt).exp();
                if imp < min_imp {
                    min_imp = imp;
                    min_idx = idx;
                }
            }
            self.entries[min_idx] = entry;
        }
    }

    pub fn recall(&mut self, query: &Tensor, k: usize) -> Vec<Value> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }
        self.current_time += 1;

        let q_norm = query.norm();
        let mut scored_indices: Vec<(usize, f64)> = self.entries.iter().enumerate().map(|(idx, e)| {
            let dot: f64 = query.data.iter().zip(e.key.data.iter()).map(|(a, b)| a * b).sum();
            let k_norm = e.key.norm();
            let cos_sim = if q_norm > 1e-12 && k_norm > 1e-12 {
                dot / (q_norm * k_norm)
            } else {
                0.0
            };
            let dt = (self.current_time - e.timestamp) as f64;
            let recency = (-self.decay_rate * dt).exp();
            let score = cos_sim * recency;
            (idx, score)
        }).collect();

        // Sort descending by score
        scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_indices: Vec<usize> = scored_indices.iter().take(k).map(|(idx, _)| *idx).collect();
        let mut results = Vec::new();

        for &idx in &top_k_indices {
            // Memory reinforcement upon recall
            self.entries[idx].access_count += 1.0;
            self.entries[idx].timestamp = self.current_time;
            results.push(self.entries[idx].value.clone());
        }

        results
    }

    pub fn recall_weighted(&mut self, query: &Tensor, k: usize) -> Result<Tensor, String> {
        let values = self.recall(query, k);
        if values.is_empty() {
            return Ok(Tensor::zeros(&[1, self.dim]));
        }
        let mut sum = vec![0.0; self.dim];
        let mut count = 0.0;
        for v in values {
            if let Value::Tensor(t) = v {
                for (s, &x) in sum.iter_mut().zip(t.data.iter()) {
                    *s += x;
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for s in sum.iter_mut() {
                *s /= count;
            }
        }
        Ok(Tensor::new(crate::buffer::Buffer::from(sum), vec![1, self.dim]))
    }

    pub fn consolidate(&mut self) {
        let now = self.current_time;
        let decay = self.decay_rate;
        // Retain memories whose importance >= 0.05
        self.entries.retain(|e| {
            let dt = (now - e.timestamp) as f64;
            let imp = e.access_count * (-decay * dt).exp();
            imp >= 0.05
        });
    }

    pub fn forget(&mut self, max_age: usize) {
        let now = self.current_time;
        self.entries.retain(|e| (now - e.timestamp) <= max_age);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let mut arena = Arena::new(1024);
        let slice = arena.alloc_f64(10);
        assert_eq!(slice.len(), 10);
        for i in 0..10 { slice[i] = i as f64; }
        assert_eq!(slice[5], 5.0);
    }

    #[test]
    fn test_arena_multiple() {
        let mut arena = Arena::new(256);
        let a = arena.alloc_f64(8);
        let a_ptr = a.as_mut_ptr();
        let a_len = a.len();
        let b = arena.alloc_f64(8);
        assert_eq!(a_len, 8);
        assert_eq!(b.len(), 8);
        // Verify they don't overlap by writing through raw pointer
        unsafe {
            *a_ptr = 1.0;
            b[0] = 2.0;
            assert_eq!(*a_ptr, 1.0);
            assert_eq!(b[0], 2.0);
        }
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = Arena::new(128);
        // Should trigger new block allocation
        let a = arena.alloc_f64(100);
        assert_eq!(a.len(), 100);
    }
}
