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
use num_bigint::BigInt;
use std::str::FromStr;

const MAGIC_WMEM: &[u8; 9] = b"NRN_WMEM\x01";
const MAGIC_EMEM: &[u8; 9] = b"NRN_EMEM\x01";
const MAGIC_SMEM: &[u8; 9] = b"NRN_SMEM\x01";
const MAGIC_UMEM: &[u8; 9] = b"NRN_UMEM\x01";

// Binary serialization format helpers
fn write_u8(buf: &mut Vec<u8>, val: u8) {
    buf.push(val);
}

fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_f64(buf: &mut Vec<u8>, val: f64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, val: &str) {
    let bytes = val.as_bytes();
    write_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

fn write_tensor(buf: &mut Vec<u8>, t: &Tensor) {
    write_u32(buf, t.shape.len() as u32);
    for &d in &t.shape {
        write_u64(buf, d as u64);
    }
    write_u64(buf, t.data.len() as u64);
    for &x in t.data.iter() {
        write_f64(buf, x);
    }
}

fn write_value(buf: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Tensor(t) => {
            write_u8(buf, 1);
            write_tensor(buf, t);
        }
        Value::Int(i) => {
            write_u8(buf, 2);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            write_u8(buf, 3);
            write_f64(buf, *f);
        }
        Value::Bool(b) => {
            write_u8(buf, 4);
            write_u8(buf, if *b { 1 } else { 0 });
        }
        Value::Str(s) => {
            write_u8(buf, 5);
            write_str(buf, s);
        }
        Value::List(items) => {
            write_u8(buf, 6);
            write_u32(buf, items.len() as u32);
            for item in items {
                write_value(buf, item);
            }
        }
        Value::Tuple(items) => {
            write_u8(buf, 7);
            write_u32(buf, items.len() as u32);
            for item in items {
                write_value(buf, item);
            }
        }
        Value::Uncertain { value, std, confidence } => {
            write_u8(buf, 8);
            write_f64(buf, *value);
            write_f64(buf, *std);
            write_f64(buf, *confidence);
        }
        Value::Random { mean, variance } => {
            write_u8(buf, 9);
            write_f64(buf, *mean);
            write_f64(buf, *variance);
        }
        Value::Void => {
            write_u8(buf, 10);
        }
        Value::None => {
            write_u8(buf, 11);
        }
        Value::Err(e) => {
            write_u8(buf, 12);
            write_str(buf, e);
        }
        Value::BigInt(bi) => {
            write_u8(buf, 13);
            write_str(buf, &bi.to_string());
        }
        Value::WorkingMemory(wm) => {
            write_u8(buf, 14);
            let bytes = wm.borrow().to_bytes();
            write_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(&bytes);
        }
        Value::EpisodicMemory(em) => {
            write_u8(buf, 15);
            let bytes = em.borrow().to_bytes();
            write_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(&bytes);
        }
        Value::SemanticMemory(sm) => {
            write_u8(buf, 16);
            let bytes = sm.borrow().to_bytes();
            write_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(&bytes);
        }
        _ => {
            write_u8(buf, 255);
            write_str(buf, &v.display());
        }
    }
}

fn read_u8(cursor: &mut &[u8]) -> Result<u8, String> {
    if cursor.is_empty() {
        return Err("Unexpected EOF reading u8".into());
    }
    let val = cursor[0];
    *cursor = &cursor[1..];
    Ok(val)
}

fn read_u32(cursor: &mut &[u8]) -> Result<u32, String> {
    if cursor.len() < 4 {
        return Err("Unexpected EOF reading u32".into());
    }
    let val = u32::from_le_bytes(cursor[..4].try_into().unwrap());
    *cursor = &cursor[4..];
    Ok(val)
}

fn read_u64(cursor: &mut &[u8]) -> Result<u64, String> {
    if cursor.len() < 8 {
        return Err("Unexpected EOF reading u64".into());
    }
    let val = u64::from_le_bytes(cursor[..8].try_into().unwrap());
    *cursor = &cursor[8..];
    Ok(val)
}

fn read_f64(cursor: &mut &[u8]) -> Result<f64, String> {
    if cursor.len() < 8 {
        return Err("Unexpected EOF reading f64".into());
    }
    let val = f64::from_le_bytes(cursor[..8].try_into().unwrap());
    *cursor = &cursor[8..];
    Ok(val)
}

fn read_str(cursor: &mut &[u8]) -> Result<String, String> {
    let len = read_u32(cursor)? as usize;
    if cursor.len() < len {
        return Err(format!("Unexpected EOF reading string of length {}", len));
    }
    let s = std::str::from_utf8(&cursor[..len]).map_err(|e| format!("Invalid UTF-8: {}", e))?.to_string();
    *cursor = &cursor[len..];
    Ok(s)
}

fn read_tensor(cursor: &mut &[u8]) -> Result<Tensor, String> {
    let rank = read_u32(cursor)? as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(read_u64(cursor)? as usize);
    }
    let total_len = read_u64(cursor)? as usize;
    if cursor.len() < total_len * 8 {
        return Err("Unexpected EOF reading tensor data".into());
    }
    let mut data = Vec::with_capacity(total_len);
    for _ in 0..total_len {
        data.push(read_f64(cursor)?);
    }
    Ok(Tensor::new(crate::buffer::Buffer::from(data), shape))
}

fn read_value(cursor: &mut &[u8]) -> Result<Value, String> {
    let tag = read_u8(cursor)?;
    match tag {
        1 => Ok(Value::Tensor(read_tensor(cursor)?)),
        2 => {
            if cursor.len() < 8 { return Err("Unexpected EOF reading Int".into()); }
            let val = i64::from_le_bytes(cursor[..8].try_into().unwrap());
            *cursor = &cursor[8..];
            Ok(Value::Int(val))
        }
        3 => Ok(Value::Float(read_f64(cursor)?)),
        4 => Ok(Value::Bool(read_u8(cursor)? != 0)),
        5 => Ok(Value::Str(read_str(cursor)?)),
        6 => {
            let count = read_u32(cursor)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value(cursor)?);
            }
            Ok(Value::List(items))
        }
        7 => {
            let count = read_u32(cursor)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value(cursor)?);
            }
            Ok(Value::Tuple(items))
        }
        8 => {
            let value = read_f64(cursor)?;
            let std = read_f64(cursor)?;
            let confidence = read_f64(cursor)?;
            Ok(Value::Uncertain { value, std, confidence })
        }
        9 => {
            let mean = read_f64(cursor)?;
            let variance = read_f64(cursor)?;
            Ok(Value::Random { mean, variance })
        }
        10 => Ok(Value::Void),
        11 => Ok(Value::None),
        12 => Ok(Value::Err(read_str(cursor)?)),
        13 => {
            let s = read_str(cursor)?;
            let bi = BigInt::from_str(&s).map_err(|e| format!("Invalid BigInt: {}", e))?;
            Ok(Value::BigInt(bi))
        }
        14 => {
            let len = read_u32(cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading WorkingMemory".into()); }
            let bytes = &cursor[..len];
            *cursor = &cursor[len..];
            let wm = WorkingMemoryData::from_bytes(bytes)?;
            Ok(Value::WorkingMemory(std::rc::Rc::new(std::cell::RefCell::new(wm))))
        }
        15 => {
            let len = read_u32(cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading EpisodicMemory".into()); }
            let bytes = &cursor[..len];
            *cursor = &cursor[len..];
            let em = EpisodicMemoryData::from_bytes(bytes)?;
            Ok(Value::EpisodicMemory(std::rc::Rc::new(std::cell::RefCell::new(em))))
        }
        16 => {
            let len = read_u32(cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading SemanticMemory".into()); }
            let bytes = &cursor[..len];
            *cursor = &cursor[len..];
            let sm = SemanticMemoryData::from_bytes(bytes)?;
            Ok(Value::SemanticMemory(std::rc::Rc::new(std::cell::RefCell::new(sm))))
        }
        255 => Ok(Value::Str(read_str(cursor)?)),
        _ => Err(format!("Unknown Value tag in deserializer: {}", tag)),
    }
}


/// Differentiable Working Memory (Neural Turing Machine / DNC architecture).
/// Maintains a fixed set of memory slots with content-based addressing and gated erase/write.
#[derive(Clone, Debug)]
pub struct WorkingMemoryData {
    pub dim: usize,
    pub slots: usize,
    /// Memory matrix: shape [slots, dim]
    pub matrix: Tensor,
    pub write_head: usize,
    pub persist_path: Option<String>,
}

impl WorkingMemoryData {
    pub fn new(dim: usize, slots: usize) -> Self {
        Self {
            dim,
            slots,
            matrix: Tensor::zeros(&[slots, dim]),
            write_head: 0,
            persist_path: None,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + self.matrix.data.len() * 8);
        buf.extend_from_slice(MAGIC_WMEM);
        write_u64(&mut buf, self.dim as u64);
        write_u64(&mut buf, self.slots as u64);
        write_u64(&mut buf, self.write_head as u64);
        write_tensor(&mut buf, &self.matrix);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 || &data[..9] != MAGIC_WMEM {
            return Err("Invalid or corrupted WorkingMemory binary header".into());
        }
        let mut cursor = &data[9..];
        let dim = read_u64(&mut cursor)? as usize;
        let slots = read_u64(&mut cursor)? as usize;
        let write_head = read_u64(&mut cursor)? as usize;
        let matrix = read_tensor(&mut cursor)?;
        Ok(Self {
            dim,
            slots,
            matrix,
            write_head,
            persist_path: None,
        })
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        let mut wm = Self::from_bytes(&bytes)?;
        wm.persist_path = Some(path.to_string());
        Ok(wm)
    }

    pub fn with_path(dim: usize, slots: usize, path: &str) -> Result<Self, String> {
        if std::path::Path::new(path).exists() {
            Self::load_from_file(path)
        } else {
            let mut wm = Self::new(dim, slots);
            wm.persist_path = Some(path.to_string());
            Ok(wm)
        }
    }

    pub fn sync(&self) -> std::io::Result<()> {
        if let Some(ref path) = self.persist_path {
            self.save_to_file(path)
        } else {
            Ok(())
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
    pub persist_path: Option<String>,
}

impl EpisodicMemoryData {
    pub fn new(dim: usize, capacity: usize, decay_rate: Option<f64>) -> Self {
        Self {
            dim,
            capacity,
            decay_rate: decay_rate.unwrap_or(0.01),
            current_time: 0,
            entries: Vec::new(),
            persist_path: None,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_EMEM);
        write_u64(&mut buf, self.dim as u64);
        write_u64(&mut buf, self.capacity as u64);
        write_f64(&mut buf, self.decay_rate);
        write_u64(&mut buf, self.current_time as u64);
        write_u64(&mut buf, self.entries.len() as u64);
        for entry in &self.entries {
            write_u64(&mut buf, entry.timestamp as u64);
            write_f64(&mut buf, entry.access_count);
            write_tensor(&mut buf, &entry.key);
            write_value(&mut buf, &entry.value);
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 || &data[..9] != MAGIC_EMEM {
            return Err("Invalid or corrupted EpisodicMemory binary header".into());
        }
        let mut cursor = &data[9..];
        let dim = read_u64(&mut cursor)? as usize;
        let capacity = read_u64(&mut cursor)? as usize;
        let decay_rate = read_f64(&mut cursor)?;
        let current_time = read_u64(&mut cursor)? as usize;
        let count = read_u64(&mut cursor)? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let timestamp = read_u64(&mut cursor)? as usize;
            let access_count = read_f64(&mut cursor)?;
            let key = read_tensor(&mut cursor)?;
            let value = read_value(&mut cursor)?;
            entries.push(MemoryEntry {
                key,
                value,
                timestamp,
                access_count,
            });
        }
        Ok(Self {
            dim,
            capacity,
            decay_rate,
            current_time,
            entries,
            persist_path: None,
        })
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        let mut em = Self::from_bytes(&bytes)?;
        em.persist_path = Some(path.to_string());
        Ok(em)
    }

    pub fn with_path(dim: usize, capacity: usize, decay_rate: Option<f64>, path: &str) -> Result<Self, String> {
        if std::path::Path::new(path).exists() {
            Self::load_from_file(path)
        } else {
            let mut em = Self::new(dim, capacity, decay_rate);
            em.persist_path = Some(path.to_string());
            Ok(em)
        }
    }

    pub fn sync(&self) -> std::io::Result<()> {
        if let Some(ref path) = self.persist_path {
            self.save_to_file(path)
        } else {
            Ok(())
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

/// Relational fact entry stored in Semantic Memory.
#[derive(Clone, Debug)]
pub struct SemanticEntry {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub value: Value,
    pub confidence: f64,
    pub embedding: Option<Tensor>,
}

/// Semantic Memory — Timeless relational knowledge graph with neural vector embedding recall.
#[derive(Clone, Debug)]
pub struct SemanticMemoryData {
    pub dim: usize,
    pub capacity: usize,
    pub persist_path: Option<String>,
    pub entries: Vec<SemanticEntry>,
}

impl SemanticMemoryData {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            capacity,
            persist_path: None,
            entries: Vec::new(),
        }
    }

    pub fn with_path(dim: usize, capacity: usize, path: &str) -> Result<Self, String> {
        if std::path::Path::new(path).exists() {
            Self::load_from_file(path)
        } else {
            let mut sm = Self::new(dim, capacity);
            sm.persist_path = Some(path.to_string());
            Ok(sm)
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn store_fact(
        &mut self,
        subject: &str,
        relation: &str,
        object: &str,
        value: Value,
        confidence: f64,
        embedding: Option<Tensor>,
    ) {
        if let Some(existing) = self.entries.iter_mut().find(|e| {
            e.subject.eq_ignore_ascii_case(subject)
                && e.relation.eq_ignore_ascii_case(relation)
                && e.object.eq_ignore_ascii_case(object)
        }) {
            existing.value = value;
            existing.confidence = confidence.clamp(0.0, 1.0);
            if embedding.is_some() {
                existing.embedding = embedding;
            }
            return;
        }

        let entry = SemanticEntry {
            subject: subject.to_string(),
            relation: relation.to_string(),
            object: object.to_string(),
            value,
            confidence: confidence.clamp(0.0, 1.0),
            embedding,
        };

        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            let mut min_idx = 0;
            let mut min_conf = f64::INFINITY;
            for (idx, e) in self.entries.iter().enumerate() {
                if e.confidence < min_conf {
                    min_conf = e.confidence;
                    min_idx = idx;
                }
            }
            self.entries[min_idx] = entry;
        }
    }

    pub fn query(
        &self,
        subject: Option<&str>,
        relation: Option<&str>,
        object: Option<&str>,
    ) -> Vec<Value> {
        self.entries
            .iter()
            .filter(|e| {
                if let Some(s) = subject {
                    if !e.subject.eq_ignore_ascii_case(s) {
                        return false;
                    }
                }
                if let Some(r) = relation {
                    if !e.relation.eq_ignore_ascii_case(r) {
                        return false;
                    }
                }
                if let Some(o) = object {
                    if !e.object.eq_ignore_ascii_case(o) {
                        return false;
                    }
                }
                true
            })
            .map(|e| {
                Value::Tuple(vec![
                    Value::Str(e.subject.clone()),
                    Value::Str(e.relation.clone()),
                    Value::Str(e.object.clone()),
                    e.value.clone(),
                    Value::Float(e.confidence),
                ])
            })
            .collect()
    }

    pub fn query_value(&self, subject: Option<&str>, relation: Option<&str>) -> Value {
        for e in &self.entries {
            let s_match = subject.map(|s| e.subject.eq_ignore_ascii_case(s)).unwrap_or(true);
            let r_match = relation.map(|r| e.relation.eq_ignore_ascii_case(r)).unwrap_or(true);
            if s_match && r_match {
                return e.value.clone();
            }
        }
        Value::None
    }

    pub fn query_vector(&self, query: &Tensor, k: usize) -> Vec<Value> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }
        let q_norm = query.norm();
        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, e)| {
                e.embedding.as_ref().map(|emb| {
                    let dot: f64 = query.data.iter().zip(emb.data.iter()).map(|(a, b)| a * b).sum();
                    let emb_norm = emb.norm();
                    let cos_sim = if q_norm > 1e-12 && emb_norm > 1e-12 {
                        dot / (q_norm * emb_norm)
                    } else {
                        0.0
                    };
                    (idx, cos_sim * e.confidence)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(idx, score)| {
                let e = &self.entries[idx];
                Value::Tuple(vec![
                    Value::Str(e.subject.clone()),
                    Value::Str(e.relation.clone()),
                    Value::Str(e.object.clone()),
                    e.value.clone(),
                    Value::Float(score),
                ])
            })
            .collect()
    }

    pub fn forget_fact(&mut self, subject: &str, relation: Option<&str>) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| {
            if !e.subject.eq_ignore_ascii_case(subject) {
                return true;
            }
            if let Some(r) = relation {
                if !e.relation.eq_ignore_ascii_case(r) {
                    return true;
                }
            }
            false
        });
        before - self.entries.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_SMEM);
        write_u64(&mut buf, self.dim as u64);
        write_u64(&mut buf, self.capacity as u64);
        write_u64(&mut buf, self.entries.len() as u64);
        for entry in &self.entries {
            write_str(&mut buf, &entry.subject);
            write_str(&mut buf, &entry.relation);
            write_str(&mut buf, &entry.object);
            write_f64(&mut buf, entry.confidence);
            if let Some(ref emb) = entry.embedding {
                write_u8(&mut buf, 1);
                write_tensor(&mut buf, emb);
            } else {
                write_u8(&mut buf, 0);
            }
            write_value(&mut buf, &entry.value);
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 || &data[..9] != MAGIC_SMEM {
            return Err("Invalid or corrupted SemanticMemory binary header".into());
        }
        let mut cursor = &data[9..];
        let dim = read_u64(&mut cursor)? as usize;
        let capacity = read_u64(&mut cursor)? as usize;
        let count = read_u64(&mut cursor)? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let subject = read_str(&mut cursor)?;
            let relation = read_str(&mut cursor)?;
            let object = read_str(&mut cursor)?;
            let confidence = read_f64(&mut cursor)?;
            let has_emb = read_u8(&mut cursor)?;
            let embedding = if has_emb == 1 {
                Some(read_tensor(&mut cursor)?)
            } else {
                None
            };
            let value = read_value(&mut cursor)?;
            entries.push(SemanticEntry {
                subject,
                relation,
                object,
                value,
                confidence,
                embedding,
            });
        }
        Ok(Self {
            dim,
            capacity,
            persist_path: None,
            entries,
        })
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        let mut sm = Self::from_bytes(&bytes)?;
        sm.persist_path = Some(path.to_string());
        Ok(sm)
    }

    pub fn sync(&self) -> std::io::Result<()> {
        if let Some(ref path) = self.persist_path {
            self.save_to_file(path)
        } else {
            Ok(())
        }
    }
}

/// Unified cognitive memory container combining Working, Episodic, and Semantic memories.
#[derive(Clone, Debug, Default)]
pub struct UnifiedMemoryData {
    pub working: Option<WorkingMemoryData>,
    pub episodic: Option<EpisodicMemoryData>,
    pub semantic: Option<SemanticMemoryData>,
    pub persist_path: Option<String>,
}

impl UnifiedMemoryData {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC_UMEM);
        
        if let Some(ref wm) = self.working {
            write_u8(&mut buf, 1);
            let b = wm.to_bytes();
            write_u32(&mut buf, b.len() as u32);
            buf.extend_from_slice(&b);
        } else {
            write_u8(&mut buf, 0);
        }

        if let Some(ref em) = self.episodic {
            write_u8(&mut buf, 1);
            let b = em.to_bytes();
            write_u32(&mut buf, b.len() as u32);
            buf.extend_from_slice(&b);
        } else {
            write_u8(&mut buf, 0);
        }

        if let Some(ref sm) = self.semantic {
            write_u8(&mut buf, 1);
            let b = sm.to_bytes();
            write_u32(&mut buf, b.len() as u32);
            buf.extend_from_slice(&b);
        } else {
            write_u8(&mut buf, 0);
        }

        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 || &data[..9] != MAGIC_UMEM {
            return Err("Invalid or corrupted UnifiedMemory binary header".into());
        }
        let mut cursor = &data[9..];
        
        let has_wm = read_u8(&mut cursor)?;
        let working = if has_wm == 1 {
            let len = read_u32(&mut cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading Unified WorkingMemory".into()); }
            let b = &cursor[..len];
            cursor = &cursor[len..];
            Some(WorkingMemoryData::from_bytes(b)?)
        } else {
            None
        };

        let has_em = read_u8(&mut cursor)?;
        let episodic = if has_em == 1 {
            let len = read_u32(&mut cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading Unified EpisodicMemory".into()); }
            let b = &cursor[..len];
            cursor = &cursor[len..];
            Some(EpisodicMemoryData::from_bytes(b)?)
        } else {
            None
        };

        let has_sm = read_u8(&mut cursor)?;
        let semantic = if has_sm == 1 {
            let len = read_u32(&mut cursor)? as usize;
            if cursor.len() < len { return Err("Unexpected EOF reading Unified SemanticMemory".into()); }
            let b = &cursor[..len];
            let _ = &cursor[len..];
            Some(SemanticMemoryData::from_bytes(b)?)
        } else {
            None
        };

        Ok(Self {
            working,
            episodic,
            semantic,
            persist_path: None,
        })
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let bytes = self.to_bytes();
        std::fs::write(path, bytes)
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file '{}': {}", path, e))?;
        let mut um = Self::from_bytes(&bytes)?;
        um.persist_path = Some(path.to_string());
        Ok(um)
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

    #[test]
    fn test_working_memory_persistence() {
        let mut wm = WorkingMemoryData::new(4, 3);
        let t = Tensor::new(crate::buffer::Buffer::from(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 4]);
        wm.write(&t);
        let bytes = wm.to_bytes();
        let loaded = WorkingMemoryData::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.dim, 4);
        assert_eq!(loaded.slots, 3);
        assert_eq!(loaded.write_head, 1);
        let q = Tensor::new(crate::buffer::Buffer::from(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 4]);
        let read_orig = wm.read(&q);
        let read_loaded = loaded.read(&q);
        assert_eq!(read_orig.data.len(), read_loaded.data.len());
        for (a, b) in read_orig.data.iter().zip(read_loaded.data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_episodic_memory_persistence() {
        let mut em = EpisodicMemoryData::new(3, 10, Some(0.05));
        let k1 = Tensor::new(crate::buffer::Buffer::from(vec![1.0, 0.0, 0.0]), vec![1, 3]);
        let k2 = Tensor::new(crate::buffer::Buffer::from(vec![0.0, 1.0, 0.0]), vec![1, 3]);
        em.store(&k1, Value::Str("event_one".into()));
        em.store(&k2, Value::Str("event_two".into()));
        
        let bytes = em.to_bytes();
        let mut loaded = EpisodicMemoryData::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        
        let rec = loaded.recall(&k1, 1);
        assert_eq!(rec.len(), 1);
        if let Value::Str(ref s) = rec[0] {
            assert_eq!(s, "event_one");
        } else {
            panic!("Expected string value");
        }
    }

    #[test]
    fn test_semantic_memory_facts_and_query() {
        let mut sm = SemanticMemoryData::new(4, 100);
        let emb = Tensor::new(crate::buffer::Buffer::from(vec![0.9, 0.1, 0.0, 0.0]), vec![1, 4]);
        sm.store_fact("Paris", "is_capital_of", "France", Value::Str("France".into()), 0.99, Some(emb.clone()));
        sm.store_fact("Tokyo", "is_capital_of", "Japan", Value::Str("Japan".into()), 0.98, None);

        // Query by subject
        let res = sm.query(Some("Paris"), None, None);
        assert_eq!(res.len(), 1);

        // Query value
        let val = sm.query_value(Some("Tokyo"), Some("is_capital_of"));
        if let Value::Str(ref s) = val {
            assert_eq!(s, "Japan");
        } else {
            panic!("Expected Japan");
        }

        // Test vector query
        let q_emb = Tensor::new(crate::buffer::Buffer::from(vec![1.0, 0.0, 0.0, 0.0]), vec![1, 4]);
        let vec_res = sm.query_vector(&q_emb, 1);
        assert_eq!(vec_res.len(), 1);

        // Test persistence
        let bytes = sm.to_bytes();
        let loaded = SemanticMemoryData::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        let val_loaded = loaded.query_value(Some("Paris"), Some("is_capital_of"));
        if let Value::Str(ref s) = val_loaded {
            assert_eq!(s, "France");
        } else {
            panic!("Expected France");
        }
    }

    #[test]
    fn test_unified_memory_bundle() {
        let mut um = UnifiedMemoryData::new();
        let mut wm = WorkingMemoryData::new(2, 2);
        wm.write(&Tensor::new(crate::buffer::Buffer::from(vec![0.5, 0.5]), vec![1, 2]));
        um.working = Some(wm);

        let mut sm = SemanticMemoryData::new(2, 10);
        sm.store_fact("Neuron", "type", "Language", Value::Str("AGI-first".into()), 1.0, None);
        um.semantic = Some(sm);

        let bytes = um.to_bytes();
        let loaded = UnifiedMemoryData::from_bytes(&bytes).unwrap();
        assert!(loaded.working.is_some());
        assert!(loaded.semantic.is_some());
        assert!(loaded.episodic.is_none());
        assert_eq!(loaded.semantic.unwrap().len(), 1);
    }
}
