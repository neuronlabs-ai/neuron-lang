/// GGUF file parser and quantized weight storage for NEURON LLM inference.
///
/// Reads GGUF files directly (from disk or byte buffer), parses tensor metadata,
/// and provides on-the-fly dequantization during matrix-vector multiplication.
///
/// Supports: F32 (type 0), F16 (type 1), Q4_0 (type 2), Q8_0 (type 8), Q6_K (type 14)

use std::collections::HashMap;

// ─────────────────────────────────────────────
//  GGUF Constants
// ─────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32

// Tensor types
const GGUF_TYPE_F32: u32 = 0;
const GGUF_TYPE_F16: u32 = 1;
const GGUF_TYPE_Q4_0: u32 = 2;
const GGUF_TYPE_Q4_1: u32 = 3;
const GGUF_TYPE_Q8_0: u32 = 8;
const GGUF_TYPE_Q6_K: u32 = 14;

// Metadata value types
const GGUF_META_UINT8: u32 = 0;
const GGUF_META_INT8: u32 = 1;
const GGUF_META_UINT16: u32 = 2;
const GGUF_META_INT16: u32 = 3;
const GGUF_META_UINT32: u32 = 4;
const GGUF_META_INT32: u32 = 5;
const GGUF_META_FLOAT32: u32 = 6;
const GGUF_META_BOOL: u32 = 7;
const GGUF_META_STRING: u32 = 8;
const GGUF_META_ARRAY: u32 = 9;
const GGUF_META_UINT64: u32 = 10;
const GGUF_META_INT64: u32 = 11;
const GGUF_META_FLOAT64: u32 = 12;

// ─────────────────────────────────────────────
//  Data Structures
// ─────────────────────────────────────────────

/// A single tensor stored in GGUF format
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub tensor_type: u32,
    pub offset: usize,  // byte offset into the data section
}

impl GgufTensor {
    /// Total number of elements in this tensor
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Size in bytes of this tensor's data
    pub fn byte_size(&self) -> usize {
        let n = self.n_elements();
        match self.tensor_type {
            GGUF_TYPE_F32 => n * 4,
            GGUF_TYPE_F16 => n * 2,
            GGUF_TYPE_Q4_0 => {
                // 32 values per block, 18 bytes per block (2 scale + 16 nibbles)
                let n_blocks = (n + 31) / 32;
                n_blocks * 18
            }
            GGUF_TYPE_Q4_1 => {
                // 32 values per block, 20 bytes per block (2 scale + 2 min + 16 nibbles)
                let n_blocks = (n + 31) / 32;
                n_blocks * 20
            }
            GGUF_TYPE_Q8_0 => {
                // 32 values per block, 34 bytes per block (2 scale + 32 int8s)
                let n_blocks = (n + 31) / 32;
                n_blocks * 34
            }
            GGUF_TYPE_Q6_K => {
                // 256 values per super-block, 210 bytes per super-block
                let n_blocks = (n + 255) / 256;
                n_blocks * 210
            }
            _ => n * 4, // fallback assume f32
        }
    }
}

/// Parsed GGUF file: metadata + tensor descriptors + raw data buffer
pub struct GgufModel {
    pub metadata: HashMap<String, GgufMetaValue>,
    pub tensors: HashMap<String, GgufTensor>,
    pub data: Vec<u8>,  // raw tensor data
    pub data_offset: usize, // offset where tensor data starts in the file
}

#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufMetaValue>),
}

impl GgufMetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufMetaValue::U32(v) => Some(*v),
            GgufMetaValue::I32(v) => Some(*v as u32),
            GgufMetaValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufMetaValue::F32(v) => Some(*v),
            GgufMetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufMetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────
//  GGUF Parser
// ─────────────────────────────────────────────

struct GgufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    fn read_u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos+1]]);
        self.pos += 2;
        v
    }

    fn read_u32(&mut self) -> u32 {
        let v = u32::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
        ]);
        self.pos += 4;
        v
    }

    fn read_i32(&mut self) -> i32 {
        let v = i32::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
        ]);
        self.pos += 4;
        v
    }

    fn read_u64(&mut self) -> u64 {
        let v = u64::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
            self.data[self.pos+4], self.data[self.pos+5],
            self.data[self.pos+6], self.data[self.pos+7],
        ]);
        self.pos += 8;
        v
    }

    fn read_i64(&mut self) -> i64 {
        let v = i64::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
            self.data[self.pos+4], self.data[self.pos+5],
            self.data[self.pos+6], self.data[self.pos+7],
        ]);
        self.pos += 8;
        v
    }

    fn read_f32(&mut self) -> f32 {
        let v = f32::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
        ]);
        self.pos += 4;
        v
    }

    fn read_f64(&mut self) -> f64 {
        let v = f64::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1],
            self.data[self.pos+2], self.data[self.pos+3],
            self.data[self.pos+4], self.data[self.pos+5],
            self.data[self.pos+6], self.data[self.pos+7],
        ]);
        self.pos += 8;
        v
    }

    fn read_string(&mut self) -> String {
        let len = self.read_u64() as usize;
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos+len]).to_string();
        self.pos += len;
        s
    }

    fn read_meta_value(&mut self, vtype: u32) -> GgufMetaValue {
        match vtype {
            GGUF_META_UINT8 => GgufMetaValue::U8(self.read_u8()),
            GGUF_META_INT8 => GgufMetaValue::I8(self.read_u8() as i8),
            GGUF_META_UINT16 => GgufMetaValue::U16(self.read_u16()),
            GGUF_META_INT16 => GgufMetaValue::I16(self.read_u16() as i16),
            GGUF_META_UINT32 => GgufMetaValue::U32(self.read_u32()),
            GGUF_META_INT32 => GgufMetaValue::I32(self.read_i32()),
            GGUF_META_FLOAT32 => GgufMetaValue::F32(self.read_f32()),
            GGUF_META_BOOL => GgufMetaValue::Bool(self.read_u8() != 0),
            GGUF_META_STRING => GgufMetaValue::Str(self.read_string()),
            GGUF_META_UINT64 => GgufMetaValue::U64(self.read_u64()),
            GGUF_META_INT64 => GgufMetaValue::I64(self.read_i64()),
            GGUF_META_FLOAT64 => GgufMetaValue::F64(self.read_f64()),
            GGUF_META_ARRAY => {
                let elem_type = self.read_u32();
                let count = self.read_u64() as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_meta_value(elem_type));
                }
                GgufMetaValue::Array(arr)
            }
            _ => {
                // Unknown type, skip 4 bytes
                self.pos += 4;
                GgufMetaValue::U32(0)
            }
        }
    }
}

/// Parse a GGUF file from a byte buffer.
/// Returns the parsed model with metadata, tensor descriptors, and raw data.
pub fn parse_gguf(file_data: &[u8]) -> Result<GgufModel, String> {
    let mut r = GgufReader::new(file_data);

    // Magic number
    let magic = r.read_u32();
    if magic != GGUF_MAGIC {
        return Err(format!("Not a GGUF file: magic={:#x}, expected={:#x}", magic, GGUF_MAGIC));
    }

    // Version
    let version = r.read_u32();
    if version < 2 || version > 3 {
        return Err(format!("Unsupported GGUF version: {}", version));
    }

    // Counts
    let n_tensors = r.read_u64() as usize;
    let n_metadata = r.read_u64() as usize;

    eprintln!("[GGUF] Version {}, {} tensors, {} metadata entries", version, n_tensors, n_metadata);

    // Parse metadata
    let mut metadata = HashMap::new();
    for _ in 0..n_metadata {
        let key = r.read_string();
        let vtype = r.read_u32();
        let value = r.read_meta_value(vtype);
        metadata.insert(key, value);
    }

    // Parse tensor info
    let mut tensor_infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.read_string();
        let n_dims = r.read_u32() as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(r.read_u64());
        }
        let tensor_type = r.read_u32();
        let offset = r.read_u64() as usize;
        tensor_infos.push(GgufTensor { name, shape, tensor_type, offset });
    }

    // Data starts at alignment boundary after header
    // GGUF v2/v3 aligns to 32 bytes by default
    let alignment = metadata.get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as usize;
    let data_offset = (r.pos + alignment - 1) / alignment * alignment;

    eprintln!("[GGUF] Data starts at offset {}, alignment={}", data_offset, alignment);

    // Build tensor map with absolute offsets into file_data
    let mut tensors = HashMap::new();
    for mut t in tensor_infos {
        t.offset += data_offset; // convert relative offset to absolute
        tensors.insert(t.name.clone(), t);
    }

    Ok(GgufModel {
        metadata,
        tensors,
        data: Vec::new(), // We reference file_data directly, not copy it
        data_offset,
    })
}

// ─────────────────────────────────────────────
//  Dequantization Functions
// ─────────────────────────────────────────────

/// Convert f16 bytes to f32
#[inline]
fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16
        let mut m = mantissa;
        let mut e = 0u32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = 127 - 15 - e;
        let f32_mantissa = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa);
    }
    if exp == 31 {
        // Inf or NaN
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    let f32_exp = exp + 127 - 15;
    let f32_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
}

/// Dequantize Q4_0 block: 18 bytes → 32 f32 values
/// Layout: [f16 scale (2 bytes)] [16 bytes of packed nibbles]
/// GGML layout: low nibble is elements 0..15, high nibble is elements 16..31
#[inline]
pub fn dequantize_q4_0_block(block: &[u8]) -> [f32; 32] {
    let scale = f16_to_f32([block[0], block[1]]);
    let mut out = [0.0f32; 32];
    for i in 0..16 {
        let byte = block[2 + i];
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out[i] = lo as f32 * scale;
        out[i + 16] = hi as f32 * scale;
    }
    out
}

/// Dequantize Q8_0 block: 34 bytes → 32 f32 values
/// Layout: [f16 scale (2 bytes)] [32 int8 values]
#[inline]
pub fn dequantize_q8_0_block(block: &[u8]) -> [f32; 32] {
    let scale = f16_to_f32([block[0], block[1]]);
    let mut out = [0.0f32; 32];
    for i in 0..32 {
        out[i] = block[2 + i] as i8 as f32 * scale;
    }
    out
}

/// Dequantize Q6_K super-block: 210 bytes → 256 f32 values
/// Layout: GGML standard block_q6_K format
#[inline]
pub fn dequantize_q6_k_block(block: &[u8]) -> [f32; 256] {
    let mut out = [0.0f32; 256];
    let d = f16_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];

    let mut ql_ptr = 0;
    let mut qh_ptr = 0;
    let mut sc_ptr = 0;

    for half in 0..2 {
        let y_base = half * 128;
        for l in 0..32 {
            let is_val = l / 16;
            let sc0 = sc[sc_ptr + is_val + 0] as i8 as f32;
            let sc2 = sc[sc_ptr + is_val + 2] as i8 as f32;
            let sc4 = sc[sc_ptr + is_val + 4] as i8 as f32;
            let sc6 = sc[sc_ptr + is_val + 6] as i8 as f32;

            let q1 = ((ql[ql_ptr + l + 0] & 0x0F) | ((qh[qh_ptr + l] & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[ql_ptr + l + 32] & 0x0F) | ((qh[qh_ptr + l] & 0x0C) << 2)) as i32 - 32;
            let q3 = ((ql[ql_ptr + l + 0] >> 4) | ((qh[qh_ptr + l] & 0x30) >> 0)) as i32 - 32;
            let q4 = ((ql[ql_ptr + l + 32] >> 4) | ((qh[qh_ptr + l] & 0xC0) >> 2)) as i32 - 32;

            out[y_base + l + 0]  = d * sc0 * q1 as f32;
            out[y_base + l + 32] = d * sc2 * q2 as f32;
            out[y_base + l + 64] = d * sc4 * q3 as f32;
            out[y_base + l + 96] = d * sc6 * q4 as f32;
        }
        ql_ptr += 64;
        qh_ptr += 32;
        sc_ptr += 8;
    }
    out
}

// ─────────────────────────────────────────────
//  Quantized Matrix-Vector Multiplication
// ─────────────────────────────────────────────

/// Quantized mat-vec: out[i] = sum_j(x[j] * dequant(W[i,j]))
/// W is stored in quantized format, row-major [n_out, n_in]
/// This dequantizes on-the-fly, block by block.
pub fn qmv_mul(
    out: &mut [f32],
    x: &[f32],
    w_data: &[u8],
    n_out: usize,
    n_in: usize,
    tensor_type: u32,
) {
    match tensor_type {
        GGUF_TYPE_F32 => {
            // Direct f32
            for row in 0..n_out {
                let offset = row * n_in * 4;
                let mut sum = 0.0f32;
                for col in 0..n_in {
                    let b = offset + col * 4;
                    let w = f32::from_le_bytes([w_data[b], w_data[b+1], w_data[b+2], w_data[b+3]]);
                    sum += x[col] * w;
                }
                out[row] = sum;
            }
        }
        GGUF_TYPE_F16 => {
            // f16 weights
            for row in 0..n_out {
                let offset = row * n_in * 2;
                let mut sum = 0.0f32;
                for col in 0..n_in {
                    let b = offset + col * 2;
                    let w = f16_to_f32([w_data[b], w_data[b+1]]);
                    sum += x[col] * w;
                }
                out[row] = sum;
            }
        }
        GGUF_TYPE_Q4_0 => {
            let block_size = 32;
            let bytes_per_block = 18;
            let blocks_per_row = (n_in + block_size - 1) / block_size;
            let row_bytes = blocks_per_row * bytes_per_block;

            for row in 0..n_out {
                let row_offset = row * row_bytes;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let block_offset = row_offset + b * bytes_per_block;
                    let block = &w_data[block_offset..block_offset + bytes_per_block];
                    let vals = dequantize_q4_0_block(block);
                    let col_start = b * block_size;
                    let col_end = (col_start + block_size).min(n_in);
                    for c in col_start..col_end {
                        sum += x[c] * vals[c - col_start];
                    }
                }
                out[row] = sum;
            }
        }
        GGUF_TYPE_Q8_0 => {
            let block_size = 32;
            let bytes_per_block = 34;
            let blocks_per_row = (n_in + block_size - 1) / block_size;
            let row_bytes = blocks_per_row * bytes_per_block;

            for row in 0..n_out {
                let row_offset = row * row_bytes;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let block_offset = row_offset + b * bytes_per_block;
                    let block = &w_data[block_offset..block_offset + bytes_per_block];
                    let vals = dequantize_q8_0_block(block);
                    let col_start = b * block_size;
                    let col_end = (col_start + block_size).min(n_in);
                    for c in col_start..col_end {
                        sum += x[c] * vals[c - col_start];
                    }
                }
                out[row] = sum;
            }
        }
        GGUF_TYPE_Q6_K => {
            let block_size = 256;
            let bytes_per_block = 210;
            let blocks_per_row = (n_in + block_size - 1) / block_size;
            let row_bytes = blocks_per_row * bytes_per_block;

            for row in 0..n_out {
                let row_offset = row * row_bytes;
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let block_offset = row_offset + b * bytes_per_block;
                    let block = &w_data[block_offset..block_offset + bytes_per_block];
                    let vals = dequantize_q6_k_block(block);
                    let col_start = b * block_size;
                    let col_end = (col_start + block_size).min(n_in);
                    for c in col_start..col_end {
                        sum += x[c] * vals[c - col_start];
                    }
                }
                out[row] = sum;
            }
        }
        _ => {
            eprintln!("[GGUF] Unsupported tensor type {} — falling back to zeros", tensor_type);
            for v in out.iter_mut() { *v = 0.0; }
        }
    }
}

/// Dequantize an entire tensor to f32 (used for embeddings, norms)
pub fn dequantize_tensor(data: &[u8], n_elements: usize, tensor_type: u32) -> Vec<f32> {
    match tensor_type {
        GGUF_TYPE_F32 => {
            let mut out = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let offset = i * 4;
                out.push(f32::from_le_bytes([
                    data[offset], data[offset+1], data[offset+2], data[offset+3],
                ]));
            }
            out
        }
        GGUF_TYPE_F16 => {
            let mut out = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let offset = i * 2;
                out.push(f16_to_f32([data[offset], data[offset+1]]));
            }
            out
        }
        GGUF_TYPE_Q4_0 => {
            let block_size = 32;
            let bytes_per_block = 18;
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = Vec::with_capacity(n_elements);
            for b in 0..n_blocks {
                let offset = b * bytes_per_block;
                let block = &data[offset..offset + bytes_per_block];
                let vals = dequantize_q4_0_block(block);
                let end = ((b + 1) * block_size).min(n_elements);
                let count = end - b * block_size;
                out.extend_from_slice(&vals[..count]);
            }
            out
        }
        GGUF_TYPE_Q8_0 => {
            let block_size = 32;
            let bytes_per_block = 34;
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = Vec::with_capacity(n_elements);
            for b in 0..n_blocks {
                let offset = b * bytes_per_block;
                let block = &data[offset..offset + bytes_per_block];
                let vals = dequantize_q8_0_block(block);
                let end = ((b + 1) * block_size).min(n_elements);
                let count = end - b * block_size;
                out.extend_from_slice(&vals[..count]);
            }
            out
        }
        GGUF_TYPE_Q6_K => {
            let block_size = 256;
            let bytes_per_block = 210;
            let n_blocks = (n_elements + block_size - 1) / block_size;
            let mut out = Vec::with_capacity(n_elements);
            for b in 0..n_blocks {
                let offset = b * bytes_per_block;
                let block = &data[offset..offset + bytes_per_block];
                let vals = dequantize_q6_k_block(block);
                let end = ((b + 1) * block_size).min(n_elements);
                let count = end - b * block_size;
                out.extend_from_slice(&vals[..count]);
            }
            out
        }
        _ => {
            eprintln!("[GGUF] Unsupported type {} for dequantize, returning zeros", tensor_type);
            vec![0.0; n_elements]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_basic() {
        // f16 1.0 = 0x3C00
        let val = f16_to_f32([0x00, 0x3C]);
        assert!((val - 1.0).abs() < 1e-6, "Expected 1.0, got {}", val);

        // f16 0.0 = 0x0000
        let val = f16_to_f32([0x00, 0x00]);
        assert!((val - 0.0).abs() < 1e-6, "Expected 0.0, got {}", val);

        // f16 -1.0 = 0xBC00
        let val = f16_to_f32([0x00, 0xBC]);
        assert!((val + 1.0).abs() < 1e-6, "Expected -1.0, got {}", val);
    }

    #[test]
    fn test_q4_0_dequantize() {
        // Create a simple Q4_0 block
        // scale = 1.0 (f16: 0x3C00)
        // nibbles: all 8 (which means value 0 after subtracting 8)
        let mut block = [0u8; 18];
        block[0] = 0x00; // f16 1.0 low byte
        block[1] = 0x3C; // f16 1.0 high byte
        // All nibbles = 0x88 → lo=8-8=0, hi=8-8=0
        for i in 2..18 {
            block[i] = 0x88;
        }
        let vals = dequantize_q4_0_block(&block);
        for v in vals.iter() {
            assert!((v - 0.0).abs() < 1e-6, "Expected 0.0, got {}", v);
        }
    }
}
