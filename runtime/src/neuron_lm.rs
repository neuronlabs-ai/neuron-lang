/// NeuronLM — A native Transformer model running inside the NEURON runtime.
///
/// Implements tokenization, embedding lookup, multi-head self-attention with GQA,
/// SwiGLU feed-forward layers, and autoregressive generation.
///
/// Supports three modes:
///   1. Small model (128 vocab, 8-dim) for demos — weights in memory
///   2. Llama 8B (128256 vocab, 4096-dim, 32 layers, GQA) — f32 weights from disk
///   3. GGUF direct loading — reads quantized weights directly, dequantizes on-the-fly

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
#[cfg(feature = "native")]
use rayon::prelude::*;
use crate::gguf::{self, GgufTensor};

#[cfg(not(target_arch = "wasm32"))]
static WEIGHT_CACHE_F32: std::sync::LazyLock<Mutex<HashMap<String, Vec<f32>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Dense layer weights for native token-free Transformer layers.
#[derive(Clone, Debug, Default)]
pub struct NativeLayerWeights {
    pub attn_q: Vec<f32>,     // [d * d]
    pub attn_k: Vec<f32>,     // [kv_dim * d]
    pub attn_v: Vec<f32>,     // [kv_dim * d]
    pub attn_out: Vec<f32>,   // [d * d]
    pub attn_norm: Vec<f32>,  // [d]
    pub ffn_gate: Vec<f32>,   // [d_ff * d]
    pub ffn_up: Vec<f32>,     // [d_ff * d]
    pub ffn_down: Vec<f32>,   // [d * d_ff]
    pub ffn_norm: Vec<f32>,   // [d]
}

#[derive(Clone, Debug)]
pub struct NeuronLM {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub w_te: Vec<Vec<f32>>,     // Token embeddings [vocab, d]
    pub w_pe: Vec<Vec<f64>>,     // Position embeddings (small model only)
    pub w_q: Vec<Vec<f64>>,      // Small model only
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
    pub w_out: Vec<Vec<f64>>,
    pub w_ff1: Vec<Vec<f64>>,
    pub w_ff2: Vec<Vec<f64>>,
    pub w_ln_g: Vec<f64>,
    pub w_ln_b: Vec<f64>,
    pub output_norm: Vec<f32>,   // Final RMSNorm weights
    pub output_weight: Vec<f32>, // LM head [vocab * d] flat
    pub is_trained: bool,
    pub is_qwen: bool,
    pub weight_dir: String,
    pub vocab: Vec<String>,
    pub attn_blocks: Vec<bool>,
    // Native token-free execution
    pub is_byte_native: bool,
    pub native_layers: Vec<NativeLayerWeights>,
    // GGUF direct loading — quantized weights dequantized on-the-fly (external adapter)
    pub gguf_data: Option<Arc<Vec<u8>>>,
    pub gguf_tensors: HashMap<String, GgufTensor>,
}

/// Load a flat f32 binary file
#[cfg(not(target_arch = "wasm32"))]
fn load_f32_file(path: &str) -> Option<Vec<f32>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    Some(bytes_to_f32(&buf))
}

/// Convert raw bytes to f32 vector
fn bytes_to_f32(buf: &[u8]) -> Vec<f32> {
    let n = buf.len() / 4;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        let bytes = [buf[offset], buf[offset+1], buf[offset+2], buf[offset+3]];
        data.push(f32::from_le_bytes(bytes));
    }
    data
}

/// Load a flat f64 binary file (for legacy small model)
#[allow(dead_code)]
#[cfg(not(target_arch = "wasm32"))]
fn load_f64_file(path: &str) -> Option<Vec<f64>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    let n = buf.len() / 8;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 8;
        let bytes = [
            buf[offset], buf[offset+1], buf[offset+2], buf[offset+3],
            buf[offset+4], buf[offset+5], buf[offset+6], buf[offset+7],
        ];
        data.push(f64::from_le_bytes(bytes));
    }
    Some(data)
}

/// Load f32 weight with caching
#[cfg(not(target_arch = "wasm32"))]
fn load_weight_f32(dir: &str, name: &str) -> Option<Vec<f32>> {
    let key = format!("{}/{}", dir, name);
    {
        let cache = WEIGHT_CACHE_F32.lock().unwrap();
        if let Some(data) = cache.get(&key) {
            return Some(data.clone());
        }
    }
    let path = format!("{}/{}.f32", dir, name);
    if let Some(data) = load_f32_file(&path) {
        let mut cache = WEIGHT_CACHE_F32.lock().unwrap();
        cache.insert(key, data.clone());
        Some(data)
    } else {
        None
    }
}

const SEED_CORPUS: &str = "\
The quick brown fox jumps over the lazy dog. \
NEURON is an agentic programming language with native causal semantics and temporal types. \
Token-free architecture executes raw UTF-8 byte streams without subword fragmentation. \
2 + 2 = 4. 10 * 10 = 100. \
Instant learning updates episodic memory in O(1) time without gradient descent. \
Machine unlearning certifies parameter scrub with empirical verification certificates. \
Autonomous systems deliberate with active inference and confidence-gated vetoes. \
";

fn xavier_init(len: usize, fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let scale = (2.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    let mut rng_state = seed;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((rng_state >> 32) as u32 as f32 + 1.0) / 4294967296.0;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((rng_state >> 32) as u32 as f32 + 1.0) / 4294967296.0;
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        vec.push(normal * scale);
    }
    vec
}

fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp());
    s * (1.0 + x * (1.0 - s))
}

fn rms_norm_backward(dy: &[f32], x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x { sum_sq += v * v; }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let mut dot = 0.0f32;
    for i in 0..n {
        let w = if i < weight.len() { weight[i] } else { 1.0 };
        dot += dy[i] * w * x[i];
    }
    let factor = dot / (n as f32 * rms * rms * rms);
    let mut dx = vec![0.0f32; n];
    for i in 0..n {
        let w = if i < weight.len() { weight[i] } else { 1.0 };
        dx[i] = (dy[i] * w) / rms - x[i] * factor;
    }
    dx
}

fn apply_inverse_rope_f32(x: &mut [f32], pos: usize, base: f32) {
    let d = x.len();
    for i in (0..d).step_by(2) {
        if i + 1 >= d { break; }
        let freq = 1.0 / base.powf(i as f32 / d as f32);
        let angle = pos as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let x0 = x[i];
        let x1 = x[i + 1];
        x[i]     = x0 * cos_a + x1 * sin_a;
        x[i + 1] = -x0 * sin_a + x1 * cos_a;
    }
}

fn mv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        let offset = r * cols;
        for c in 0..cols {
            sum += w[offset + c] * x[c];
        }
        y[r] = sum;
    }
    y
}

fn mtv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; cols];
    for r in 0..rows {
        let offset = r * cols;
        let xr = x[r];
        for c in 0..cols {
            y[c] += w[offset + c] * xr;
        }
    }
    y
}

fn outer_acc(dw: &mut [f32], dy: &[f32], dx: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let yr = dy[r];
        let offset = r * cols;
        for c in 0..cols {
            dw[offset + c] += yr * dx[c];
        }
    }
}

fn adam_step(param: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, beta1: f32, beta2: f32, eps: f32, step: usize) {
    let bias_corr1 = (1.0 - beta1.powi(step as i32)).max(1e-8);
    let bias_corr2 = (1.0 - beta2.powi(step as i32)).max(1e-8);
    for i in 0..param.len() {
        let g = grad[i].clamp(-10.0, 10.0);
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bias_corr1;
        let v_hat = v[i] / bias_corr2;
        param[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

impl NeuronLM {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let byte_model_path = "data/native_byte_model.bin";
            if std::path::Path::new(byte_model_path).exists() {
                if let Ok(m) = Self::load_byte_model(byte_model_path) {
                    eprintln!("[NeuronLM] Loaded native token-free byte model from {}", byte_model_path);
                    return m;
                }
            }
        }

        // Initialize and calibrate native token-free byte substrate
        eprintln!("[NeuronLM] Initializing native token-free byte substrate (vocab=256, d=64, layers=2)...");
        let mut model = Self::new_byte_model(64, 4, 2, 128);
        let final_loss = model.train_byte_model(SEED_CORPUS, 100, 0.005);
        eprintln!("[NeuronLM] Native byte model calibrated (cross-entropy loss: {:.4})", final_loss);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = std::fs::create_dir_all("data");
            if let Err(e) = model.save_byte_model("data/native_byte_model.bin") {
                eprintln!("[NeuronLM] Note: failed to cache byte model: {}", e);
            } else {
                eprintln!("[NeuronLM] Cached trained native byte model to data/native_byte_model.bin");
            }
        }

        model
    }

    /// Construct a fresh token-free byte transformer model.
    pub fn new_byte_model(embed_dim: usize, num_heads: usize, num_layers: usize, d_ff: usize) -> Self {
        let vocab_size = 256;
        let mut rng_seed = 42u64;

        let mut w_te = vec![vec![0.0f32; embed_dim]; vocab_size];
        for i in 0..vocab_size {
            w_te[i] = xavier_init(embed_dim, vocab_size, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(101);
        }

        let mut native_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let attn_q = xavier_init(embed_dim * embed_dim, embed_dim, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(102);
            let attn_k = xavier_init(embed_dim * embed_dim, embed_dim, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(103);
            let attn_v = xavier_init(embed_dim * embed_dim, embed_dim, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(104);
            let attn_out = xavier_init(embed_dim * embed_dim, embed_dim, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(105);
            let attn_norm = vec![1.0f32; embed_dim];

            let ffn_gate = xavier_init(d_ff * embed_dim, embed_dim, d_ff, rng_seed);
            rng_seed = rng_seed.wrapping_add(106);
            let ffn_up = xavier_init(d_ff * embed_dim, embed_dim, d_ff, rng_seed);
            rng_seed = rng_seed.wrapping_add(107);
            let ffn_down = xavier_init(embed_dim * d_ff, d_ff, embed_dim, rng_seed);
            rng_seed = rng_seed.wrapping_add(108);
            let ffn_norm = vec![1.0f32; embed_dim];

            native_layers.push(NativeLayerWeights {
                attn_q, attn_k, attn_v, attn_out, attn_norm,
                ffn_gate, ffn_up, ffn_down, ffn_norm,
            });
        }

        let output_norm = vec![1.0f32; embed_dim];
        let output_weight = xavier_init(vocab_size * embed_dim, embed_dim, vocab_size, rng_seed);

        let vocab: Vec<String> = (0..256).map(|b| (b as u8 as char).to_string()).collect();

        Self {
            embed_dim,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size,
            num_layers,
            d_ff,
            rope_base: 10000.0,
            rms_eps: 1e-5,
            w_te,
            w_pe: vec![vec![0.0; 1]; 1],
            w_q: vec![vec![0.0; 1]; 1],
            w_k: vec![vec![0.0; 1]; 1],
            w_v: vec![vec![0.0; 1]; 1],
            w_out: vec![vec![0.0; 1]; 1],
            w_ff1: vec![vec![0.0; 1]; 1],
            w_ff2: vec![vec![0.0; 1]; 1],
            w_ln_g: vec![1.0; embed_dim],
            w_ln_b: vec![0.0; embed_dim],
            output_norm,
            output_weight,
            is_trained: true,
            is_qwen: false,
            weight_dir: String::new(),
            vocab,
            attn_blocks: vec![true; num_layers],
            gguf_data: None,
            gguf_tensors: HashMap::new(),
            is_byte_native: true,
            native_layers,
        }
    }

    /// Save native byte model weights to disk in binary format.
    pub fn save_byte_model(&self, path: &str) -> std::io::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"NRN_BYTE\x01");
        buf.extend_from_slice(&(self.embed_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_heads as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_layers as u32).to_le_bytes());
        buf.extend_from_slice(&(self.d_ff as u32).to_le_bytes());

        for row in &self.w_te {
            for &val in row {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        for layer in &self.native_layers {
            for slice in &[
                &layer.attn_q, &layer.attn_k, &layer.attn_v, &layer.attn_out,
                &layer.attn_norm, &layer.ffn_gate, &layer.ffn_up, &layer.ffn_down,
                &layer.ffn_norm,
            ] {
                for &val in *slice {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        for &val in &self.output_norm {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        for &val in &self.output_weight {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        std::fs::write(path, buf)
    }

    /// Load native byte model weights from disk.
    pub fn load_byte_model(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        if data.len() < 25 || &data[0..9] != b"NRN_BYTE\x01" {
            return Err("Invalid NRN_BYTE model header".into());
        }
        let d = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let num_heads = u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize;
        let num_layers = u32::from_le_bytes(data[17..21].try_into().unwrap()) as usize;
        let d_ff = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;

        let mut offset = 25;
        let mut read_f32_slice = |len: usize| -> Result<Vec<f32>, String> {
            let byte_len = len * 4;
            if offset + byte_len > data.len() {
                return Err("Unexpected EOF while reading model weights".into());
            }
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                let start = offset + i * 4;
                let bytes: [u8; 4] = data[start..start + 4].try_into().unwrap();
                v.push(f32::from_le_bytes(bytes));
            }
            offset += byte_len;
            Ok(v)
        };

        let mut w_te = vec![vec![0.0f32; d]; 256];
        for v in 0..256 {
            w_te[v] = read_f32_slice(d)?;
        }

        let mut native_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let attn_q = read_f32_slice(d * d)?;
            let attn_k = read_f32_slice(d * d)?;
            let attn_v = read_f32_slice(d * d)?;
            let attn_out = read_f32_slice(d * d)?;
            let attn_norm = read_f32_slice(d)?;
            let ffn_gate = read_f32_slice(d_ff * d)?;
            let ffn_up = read_f32_slice(d_ff * d)?;
            let ffn_down = read_f32_slice(d * d_ff)?;
            let ffn_norm = read_f32_slice(d)?;

            native_layers.push(NativeLayerWeights {
                attn_q, attn_k, attn_v, attn_out, attn_norm,
                ffn_gate, ffn_up, ffn_down, ffn_norm,
            });
        }

        let output_norm = read_f32_slice(d)?;
        let output_weight = read_f32_slice(256 * d)?;

        let vocab: Vec<String> = (0..256).map(|b| (b as u8 as char).to_string()).collect();

        Ok(Self {
            embed_dim: d,
            num_heads,
            num_kv_heads: num_heads,
            vocab_size: 256,
            num_layers,
            d_ff,
            rope_base: 10000.0,
            rms_eps: 1e-5,
            w_te,
            w_pe: vec![vec![0.0; 1]; 1],
            w_q: vec![vec![0.0; 1]; 1],
            w_k: vec![vec![0.0; 1]; 1],
            w_v: vec![vec![0.0; 1]; 1],
            w_out: vec![vec![0.0; 1]; 1],
            w_ff1: vec![vec![0.0; 1]; 1],
            w_ff2: vec![vec![0.0; 1]; 1],
            w_ln_g: vec![1.0; d],
            w_ln_b: vec![0.0; d],
            output_norm,
            output_weight,
            is_trained: true,
            is_qwen: false,
            is_byte_native: true,
            native_layers,
            weight_dir: String::new(),
            vocab,
            attn_blocks: vec![true; num_layers],
            gguf_data: None,
            gguf_tensors: HashMap::new(),
        })
    }

    /// Train native byte model weights on a text corpus using pure backpropagation with Adam.
    pub fn train_byte_model(&mut self, corpus: &str, steps: usize, lr: f32) -> f32 {
        let corpus_bytes = corpus.as_bytes();
        let max_len = corpus_bytes.len().saturating_sub(1);
        if max_len == 0 { return 0.0; }
        let seq_len = 64.min(max_len).max(1);
        let max_offset = corpus_bytes.len().saturating_sub(seq_len);
        if max_offset == 0 { return 0.0; }

        let d = self.embed_dim;
        let num_heads = self.num_heads;
        let head_dim = d / num_heads;
        let d_ff = self.d_ff;
        let scale = (head_dim as f32).sqrt();

        let mut m_te = vec![vec![0.0f32; d]; 256];
        let mut v_te = vec![vec![0.0f32; d]; 256];

        struct LayerAdam {
            m_q: Vec<f32>, v_q: Vec<f32>,
            m_k: Vec<f32>, v_k: Vec<f32>,
            m_v: Vec<f32>, v_v: Vec<f32>,
            m_out: Vec<f32>, v_out: Vec<f32>,
            m_an: Vec<f32>, v_an: Vec<f32>,
            m_gate: Vec<f32>, v_gate: Vec<f32>,
            m_up: Vec<f32>, v_up: Vec<f32>,
            m_down: Vec<f32>, v_down: Vec<f32>,
            m_fn: Vec<f32>, v_fn: Vec<f32>,
        }

        let mut layer_adams: Vec<LayerAdam> = (0..self.num_layers).map(|_| LayerAdam {
            m_q: vec![0.0; d * d], v_q: vec![0.0; d * d],
            m_k: vec![0.0; d * d], v_k: vec![0.0; d * d],
            m_v: vec![0.0; d * d], v_v: vec![0.0; d * d],
            m_out: vec![0.0; d * d], v_out: vec![0.0; d * d],
            m_an: vec![0.0; d], v_an: vec![0.0; d],
            m_gate: vec![0.0; d_ff * d], v_gate: vec![0.0; d_ff * d],
            m_up: vec![0.0; d_ff * d], v_up: vec![0.0; d_ff * d],
            m_down: vec![0.0; d * d_ff], v_down: vec![0.0; d * d_ff],
            m_fn: vec![0.0; d], v_fn: vec![0.0; d],
        }).collect();

        let mut m_out_w = vec![0.0f32; 256 * d];
        let mut v_out_w = vec![0.0f32; 256 * d];
        let mut m_out_n = vec![0.0f32; d];
        let mut v_out_n = vec![0.0f32; d];

        let mut last_loss = 0.0f32;
        let stride = 37.max(seq_len / 2);

        for step in 1..=steps {
            let offset = (step * stride) % max_offset;
            let input_slice = &corpus_bytes[offset..offset + seq_len];
            let target_slice = &corpus_bytes[offset + 1..offset + 1 + seq_len];

            // 1. Forward Pass
            let mut h: Vec<Vec<f32>> = input_slice.iter().map(|&b| self.w_te[b as usize].clone()).collect();
            let mut layer_intermediates = Vec::with_capacity(self.num_layers);

            for layer_idx in 0..self.num_layers {
                let layer = &self.native_layers[layer_idx];
                let h_in = h.clone();

                let h_norm1: Vec<Vec<f32>> = (0..seq_len)
                    .map(|s| rms_norm_f32(&h[s], &layer.attn_norm, self.rms_eps))
                    .collect();

                let mut q = vec![vec![0.0f32; d]; seq_len];
                let mut k = vec![vec![0.0f32; d]; seq_len];
                let mut v = vec![vec![0.0f32; d]; seq_len];
                for s in 0..seq_len {
                    q[s] = mv(&layer.attn_q, &h_norm1[s], d, d);
                    k[s] = mv(&layer.attn_k, &h_norm1[s], d, d);
                    v[s] = mv(&layer.attn_v, &h_norm1[s], d, d);
                    for h_idx in 0..num_heads {
                        let start = h_idx * head_dim;
                        apply_rope_f32(&mut q[s][start..start + head_dim], s, self.rope_base);
                        apply_rope_f32(&mut k[s][start..start + head_dim], s, self.rope_base);
                    }
                }

                let mut scores = vec![vec![vec![0.0f32; seq_len]; num_heads]; seq_len];
                let mut attn_context = vec![vec![0.0f32; d]; seq_len];

                for h_idx in 0..num_heads {
                    let h_start = h_idx * head_dim;
                    for i in 0..seq_len {
                        let mut raw_scores = vec![f32::NEG_INFINITY; i + 1];
                        for j in 0..=i {
                            let mut dot = 0.0f32;
                            for hd in 0..head_dim {
                                dot += q[i][h_start + hd] * k[j][h_start + hd];
                            }
                            raw_scores[j] = dot / scale;
                        }
                        let max_s = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0f32;
                        for j in 0..=i {
                            raw_scores[j] = (raw_scores[j] - max_s).exp();
                            sum_exp += raw_scores[j];
                        }
                        for j in 0..=i {
                            scores[i][h_idx][j] = raw_scores[j] / sum_exp.max(1e-12);
                            for hd in 0..head_dim {
                                attn_context[i][h_start + hd] += scores[i][h_idx][j] * v[j][h_start + hd];
                            }
                        }
                    }
                }

                let mut attn_out = vec![vec![0.0f32; d]; seq_len];
                for s in 0..seq_len {
                    attn_out[s] = mv(&layer.attn_out, &attn_context[s], d, d);
                    for c in 0..d { h[s][c] += attn_out[s][c]; }
                }

                let h_mid = h.clone();

                let h_norm2: Vec<Vec<f32>> = (0..seq_len)
                    .map(|s| rms_norm_f32(&h[s], &layer.ffn_norm, self.rms_eps))
                    .collect();

                let mut gate_val = vec![vec![0.0f32; d_ff]; seq_len];
                let mut up_val = vec![vec![0.0f32; d_ff]; seq_len];
                let mut inter_val = vec![vec![0.0f32; d_ff]; seq_len];
                let mut ffn_out = vec![vec![0.0f32; d]; seq_len];

                for s in 0..seq_len {
                    gate_val[s] = mv(&layer.ffn_gate, &h_norm2[s], d_ff, d);
                    up_val[s] = mv(&layer.ffn_up, &h_norm2[s], d_ff, d);
                    for i in 0..d_ff {
                        inter_val[s][i] = silu_f32(gate_val[s][i]) * up_val[s][i];
                    }
                    ffn_out[s] = mv(&layer.ffn_down, &inter_val[s], d, d_ff);
                    for c in 0..d { h[s][c] += ffn_out[s][c]; }
                }

                layer_intermediates.push((h_in, h_norm1, q, k, v, scores, attn_context, h_mid, h_norm2, gate_val, up_val, inter_val));
            }

            let h_final: Vec<Vec<f32>> = (0..seq_len)
                .map(|s| rms_norm_f32(&h[s], &self.output_norm, self.rms_eps))
                .collect();

            let mut logits = vec![vec![0.0f32; 256]; seq_len];
            let mut loss = 0.0f32;
            let mut d_logits = vec![vec![0.0f32; 256]; seq_len];

            for s in 0..seq_len {
                logits[s] = mv(&self.output_weight, &h_final[s], 256, d);
                let max_l = logits[s].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                let mut probs = vec![0.0f32; 256];
                for i in 0..256 {
                    probs[i] = (logits[s][i] - max_l).exp();
                    exp_sum += probs[i];
                }
                for i in 0..256 {
                    probs[i] /= exp_sum.max(1e-12);
                }
                let target = target_slice[s] as usize;
                loss -= (probs[target] + 1e-12).ln();
                for i in 0..256 {
                    let indicator = if i == target { 1.0f32 } else { 0.0f32 };
                    d_logits[s][i] = (probs[i] - indicator) / seq_len as f32;
                }
            }
            loss /= seq_len as f32;
            last_loss = loss;

            if step == 1 || step % 50 == 0 || step == steps {
                eprintln!("[NeuronLM Train] Step {:4}/{} | Cross-Entropy Loss: {:.4}", step, steps, loss);
            }

            // 2. Backward Pass
            let mut d_out_w = vec![0.0f32; 256 * d];
            let mut d_out_norm = vec![0.0f32; d];
            let mut dh = vec![vec![0.0f32; d]; seq_len];

            for s in 0..seq_len {
                outer_acc(&mut d_out_w, &d_logits[s], &h_final[s], 256, d);
                let dh_fin = mtv(&self.output_weight, &d_logits[s], 256, d);
                for i in 0..d { d_out_norm[i] += dh_fin[i] * h_final[s][i]; }
                dh[s] = rms_norm_backward(&dh_fin, &h[s], &self.output_norm, self.rms_eps);
            }

            for layer_idx in (0..self.num_layers).rev() {
                let layer = &self.native_layers[layer_idx];
                let (h_in, h_norm1, q, k, v, scores, attn_context, h_mid, h_norm2, gate_val, up_val, inter_val) = &layer_intermediates[layer_idx];

                let mut d_ffn_down = vec![0.0f32; d * d_ff];
                let mut d_ffn_gate = vec![0.0f32; d_ff * d];
                let mut d_ffn_up = vec![0.0f32; d_ff * d];
                let mut d_ffn_norm = vec![0.0f32; d];

                let mut dh_mid = dh.clone();

                for s in 0..seq_len {
                    outer_acc(&mut d_ffn_down, &dh[s], &inter_val[s], d, d_ff);
                    let d_inter = mtv(&layer.ffn_down, &dh[s], d, d_ff);
                    let mut d_gate = vec![0.0f32; d_ff];
                    let mut d_up = vec![0.0f32; d_ff];
                    for i in 0..d_ff {
                        d_gate[i] = d_inter[i] * up_val[s][i] * silu_grad(gate_val[s][i]);
                        d_up[i] = d_inter[i] * silu_f32(gate_val[s][i]);
                    }
                    outer_acc(&mut d_ffn_gate, &d_gate, &h_norm2[s], d_ff, d);
                    outer_acc(&mut d_ffn_up, &d_up, &h_norm2[s], d_ff, d);
                    let dh_norm2 = {
                        let v1 = mtv(&layer.ffn_gate, &d_gate, d_ff, d);
                        let v2 = mtv(&layer.ffn_up, &d_up, d_ff, d);
                        (0..d).map(|i| v1[i] + v2[i]).collect::<Vec<f32>>()
                    };
                    for i in 0..d { d_ffn_norm[i] += dh_norm2[i] * h_norm2[s][i]; }
                    let dx_ffn = rms_norm_backward(&dh_norm2, &h_mid[s], &layer.ffn_norm, self.rms_eps);
                    for i in 0..d { dh_mid[s][i] += dx_ffn[i]; }
                }

                let mut d_attn_out = vec![0.0f32; d * d];
                let mut d_attn_q = vec![0.0f32; d * d];
                let mut d_attn_k = vec![0.0f32; d * d];
                let mut d_attn_v = vec![0.0f32; d * d];
                let mut d_attn_norm = vec![0.0f32; d];

                let mut dh_in = dh_mid.clone();

                for s in 0..seq_len {
                    outer_acc(&mut d_attn_out, &dh_mid[s], &attn_context[s], d, d);
                }

                let mut d_context = vec![vec![0.0f32; d]; seq_len];
                for s in 0..seq_len {
                    d_context[s] = mtv(&layer.attn_out, &dh_mid[s], d, d);
                }

                let mut dq = vec![vec![0.0f32; d]; seq_len];
                let mut dk = vec![vec![0.0f32; d]; seq_len];
                let mut dv = vec![vec![0.0f32; d]; seq_len];

                for h_idx in 0..num_heads {
                    let h_start = h_idx * head_dim;
                    for i in 0..seq_len {
                        for j in 0..=i {
                            for hd in 0..head_dim {
                                dv[j][h_start + hd] += scores[i][h_idx][j] * d_context[i][h_start + hd];
                            }
                        }
                    }
                    for i in 0..seq_len {
                        let mut d_score = vec![0.0f32; i + 1];
                        for j in 0..=i {
                            let mut dot = 0.0f32;
                            for hd in 0..head_dim {
                                dot += d_context[i][h_start + hd] * v[j][h_start + hd];
                            }
                            d_score[j] = dot;
                        }
                        let mut sum_s_ds = 0.0f32;
                        for j in 0..=i {
                            sum_s_ds += scores[i][h_idx][j] * d_score[j];
                        }
                        for j in 0..=i {
                            let d_pre = scores[i][h_idx][j] * (d_score[j] - sum_s_ds) / scale;
                            for hd in 0..head_dim {
                                dq[i][h_start + hd] += d_pre * k[j][h_start + hd];
                                dk[j][h_start + hd] += d_pre * q[i][h_start + hd];
                            }
                        }
                    }
                }

                for s in 0..seq_len {
                    for h_idx in 0..num_heads {
                        let start = h_idx * head_dim;
                        apply_inverse_rope_f32(&mut dq[s][start..start + head_dim], s, self.rope_base);
                        apply_inverse_rope_f32(&mut dk[s][start..start + head_dim], s, self.rope_base);
                    }
                    outer_acc(&mut d_attn_q, &dq[s], &h_norm1[s], d, d);
                    outer_acc(&mut d_attn_k, &dk[s], &h_norm1[s], d, d);
                    outer_acc(&mut d_attn_v, &dv[s], &h_norm1[s], d, d);

                    let dh_norm1 = {
                        let vq = mtv(&layer.attn_q, &dq[s], d, d);
                        let vk = mtv(&layer.attn_k, &dk[s], d, d);
                        let vv = mtv(&layer.attn_v, &dv[s], d, d);
                        (0..d).map(|i| vq[i] + vk[i] + vv[i]).collect::<Vec<f32>>()
                    };
                    for i in 0..d { d_attn_norm[i] += dh_norm1[i] * h_norm1[s][i]; }
                    let dx_attn = rms_norm_backward(&dh_norm1, &h_in[s], &layer.attn_norm, self.rms_eps);
                    for i in 0..d { dh_in[s][i] += dx_attn[i]; }
                }

                let adam = &mut layer_adams[layer_idx];
                let l = &mut self.native_layers[layer_idx];
                adam_step(&mut l.attn_q, &d_attn_q, &mut adam.m_q, &mut adam.v_q, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.attn_k, &d_attn_k, &mut adam.m_k, &mut adam.v_k, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.attn_v, &d_attn_v, &mut adam.m_v, &mut adam.v_v, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.attn_out, &d_attn_out, &mut adam.m_out, &mut adam.v_out, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.attn_norm, &d_attn_norm, &mut adam.m_an, &mut adam.v_an, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.ffn_gate, &d_ffn_gate, &mut adam.m_gate, &mut adam.v_gate, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.ffn_up, &d_ffn_up, &mut adam.m_up, &mut adam.v_up, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.ffn_down, &d_ffn_down, &mut adam.m_down, &mut adam.v_down, lr, 0.9, 0.999, 1e-8, step);
                adam_step(&mut l.ffn_norm, &d_ffn_norm, &mut adam.m_fn, &mut adam.v_fn, lr, 0.9, 0.999, 1e-8, step);

                dh = dh_in;
            }

            for s in 0..seq_len {
                let tok = input_slice[s] as usize;
                adam_step(&mut self.w_te[tok], &dh[s], &mut m_te[tok], &mut v_te[tok], lr, 0.9, 0.999, 1e-8, step);
            }

            adam_step(&mut self.output_weight, &d_out_w, &mut m_out_w, &mut v_out_w, lr, 0.9, 0.999, 1e-8, step);
            adam_step(&mut self.output_norm, &d_out_norm, &mut m_out_n, &mut v_out_n, lr, 0.9, 0.999, 1e-8, step);
        }

        self.is_trained = true;
        last_loss
    }

    /// Autoregressively generate next bytes from raw UTF-8 prompt.
    pub fn generate_byte(&self, prompt: &str) -> String {
        let byte_ids: Vec<usize> = if prompt.is_empty() {
            vec![32]
        } else {
            prompt.as_bytes().iter().map(|&b| b as usize).collect()
        };

        let d = self.embed_dim;
        let seq_len = byte_ids.len();

        let mut h: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &bid in &byte_ids {
            let idx = bid.min(self.w_te.len().saturating_sub(1));
            h.push(self.w_te[idx].clone());
        }

        let mut kv_cache: Vec<Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> =
            vec![Vec::new(); self.num_layers];

        for layer in 0..self.num_layers {
            self.llama_block_forward_cached(&mut h, layer, &mut kv_cache[layer]);
        }

        let h_final = rms_norm_f32(&h[seq_len - 1], &self.output_norm, self.rms_eps);
        let mut generated_bytes = Vec::new();
        let mut last_h = h_final;

        for step in 0..64 {
            let mut logits = vec![0.0f32; self.vocab_size];
            mat_vec_mul(&mut logits, &last_h, &self.output_weight, self.vocab_size, d);

            // Apply repetition penalty on recent bytes to break repetitive loops
            for &past_b in generated_bytes.iter().rev().take(6) {
                logits[past_b as usize] -= 3.0;
            }

            let mut best_id = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (v, &val) in logits.iter().enumerate() {
                if v != 0 && val > best_score {
                    best_score = val;
                    best_id = v;
                }
            }

            if best_id == 0 || best_id == 3 || best_id == 4 {
                break;
            }
            if best_id == 10 && generated_bytes.len() >= 8 {
                generated_bytes.push(best_id as u8);
                break;
            }

            generated_bytes.push(best_id as u8);

            let pos = seq_len + step;
            let idx = best_id.min(self.w_te.len().saturating_sub(1));
            let mut h_tok = self.w_te[idx].clone();
            for layer in 0..self.num_layers {
                self.llama_single_token_cached(&mut h_tok, layer, pos, &mut kv_cache[layer]);
            }
            last_h = rms_norm_f32(&h_tok, &self.output_norm, self.rms_eps);
        }

        let result = String::from_utf8_lossy(&generated_bytes).to_string();
        format!("[NeuronLM/Byte]: {}", result.trim())
    }

    #[allow(dead_code)]
    #[cfg(not(target_arch = "wasm32"))]
    fn new_llama(weight_dir: &str) -> Self {
        let d = 4096;
        let d_ff = 14336;
        let num_heads = 32;
        let num_kv_heads = 8;
        let num_layers = 32;

        eprintln!("[NeuronLM] Loading Llama 8B from {}", weight_dir);

        // Load vocabulary
        let vocab_path = format!("{}/vocab.json", weight_dir);
        let vocab = match std::fs::read_to_string(&vocab_path) {
            Ok(json_str) => {
                let trimmed = json_str.trim();
                let inner = &trimmed[1..trimmed.len()-1];
                let mut tokens = Vec::new();
                let mut in_string = false;
                let mut escaped = false;
                let mut current = String::new();
                for ch in inner.chars() {
                    if escaped { current.push(ch); escaped = false; }
                    else if ch == '\\' && in_string { escaped = true; current.push(ch); }
                    else if ch == '"' {
                        if in_string {
                            let unescaped = current
                                .replace("\\n", "\n").replace("\\t", "\t")
                                .replace("\\\"", "\"").replace("\\\\", "\\");
                            tokens.push(unescaped);
                            current = String::new();
                        }
                        in_string = !in_string;
                    } else if in_string { current.push(ch); }
                }
                eprintln!("[NeuronLM] Loaded {} BPE tokens", tokens.len());
                tokens
            }
            Err(_) => Vec::new(),
        };
        let vocab_size = vocab.len();

        // Load token embeddings
        let mut w_te = vec![vec![0.0f32; d]; vocab_size];
        if let Some(embd_data) = load_f32_file(&format!("{}/token_embd.f32", weight_dir)) {
            let available = embd_data.len() / d;
            let load_count = vocab_size.min(available);
            for i in 0..load_count {
                for j in 0..d {
                    w_te[i][j] = embd_data[i * d + j];
                }
            }
            eprintln!("[NeuronLM] Loaded {} token embeddings", load_count);
        }

        // Load output norm
        let output_norm = load_f32_file(&format!("{}/output_norm.f32", weight_dir))
            .unwrap_or_else(|| vec![1.0; d]);

        // Load output (LM head) weight - flat [vocab_size * d]
        let output_weight = load_f32_file(&format!("{}/output.f32", weight_dir))
            .unwrap_or_default();

        eprintln!("[NeuronLM] Llama 8B ready — {} layers, {} vocab, d={}, GQA {}/{}",
            num_layers, vocab_size, d, num_heads, num_kv_heads);

        Self {
            embed_dim: d, num_heads, num_kv_heads, vocab_size, num_layers,
            d_ff, rope_base: 500000.0, rms_eps: 1e-5,
            w_te, w_pe: vec![vec![0.0; 1]; 1],
            w_q: vec![vec![0.0; 1]; 1], w_k: vec![vec![0.0; 1]; 1],
            w_v: vec![vec![0.0; 1]; 1], w_out: vec![vec![0.0; 1]; 1],
            w_ff1: vec![vec![0.0; 1]; 1], w_ff2: vec![vec![0.0; 1]; 1],
            w_ln_g: vec![1.0; d], w_ln_b: vec![0.0; d],
            output_norm, output_weight,
            is_trained: true, is_qwen: true,
            weight_dir: weight_dir.to_string(),
            vocab,
            attn_blocks: vec![true; num_layers], // All layers have attention
            is_byte_native: false,
            native_layers: Vec::new(),
            gguf_data: None,
            gguf_tensors: HashMap::new(),
        }
    }

    /// Load a model directly from GGUF bytes (works in WASM and native).
    /// This is the primary path for WASM — weights stay quantized, dequantized on-the-fly.
    pub fn new_from_gguf_bytes(data: Vec<u8>) -> Result<Self, String> {
        let model = gguf::parse_gguf(&data)?;

        // Read architecture from metadata
        let arch = model.metadata.get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        let prefix = if arch == "llama" { "llama" } else { arch };

        let d = model.metadata.get(&format!("{}.embedding_length", prefix))
            .and_then(|v| v.as_u32()).unwrap_or(2048) as usize;
        let d_ff = model.metadata.get(&format!("{}.feed_forward_length", prefix))
            .and_then(|v| v.as_u32()).unwrap_or(5632) as usize;
        let num_heads = model.metadata.get(&format!("{}.attention.head_count", prefix))
            .and_then(|v| v.as_u32()).unwrap_or(32) as usize;
        let num_kv_heads = model.metadata.get(&format!("{}.attention.head_count_kv", prefix))
            .and_then(|v| v.as_u32()).unwrap_or(4) as usize;
        let num_layers = model.metadata.get(&format!("{}.block_count", prefix))
            .and_then(|v| v.as_u32()).unwrap_or(22) as usize;
        let rope_base = model.metadata.get(&format!("{}.rope.freq_base", prefix))
            .and_then(|v| v.as_f32()).unwrap_or(10000.0);
        let rms_eps = model.metadata.get(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .and_then(|v| v.as_f32()).unwrap_or(1e-5);

        eprintln!("[NeuronLM/GGUF] Architecture: {}, d={}, d_ff={}, heads={}/{}, layers={}, rope={}",
            prefix, d, d_ff, num_heads, num_kv_heads, num_layers, rope_base);

        // Dequantize token embeddings (needed for fast lookup)
        let embd_tensor = model.tensors.get("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;
        let vocab_size = embd_tensor.shape[1] as usize; // shape is [d, vocab]
        let embd_flat = gguf::dequantize_tensor(
            &data[embd_tensor.offset..embd_tensor.offset + embd_tensor.byte_size()],
            embd_tensor.n_elements(),
            embd_tensor.tensor_type,
        );
        // GGUF shape [d, vocab] but storage is row-major [vocab][d] — read sequentially
        let mut w_te = vec![vec![0.0f32; d]; vocab_size];
        for v in 0..vocab_size {
            for j in 0..d {
                w_te[v][j] = embd_flat[v * d + j];
            }
        }
        eprintln!("[NeuronLM/GGUF] Loaded {} token embeddings (d={})", vocab_size, d);

        // Dequantize output_norm (always f32, small)
        let output_norm = if let Some(t) = model.tensors.get("output_norm.weight") {
            gguf::dequantize_tensor(
                &data[t.offset..t.offset + t.byte_size()],
                t.n_elements(), t.tensor_type,
            )
        } else {
            vec![1.0; d]
        };

        // Load BPE vocabulary from GGUF metadata
        let vocab = Self::extract_vocab_from_gguf(&model);
        eprintln!("[NeuronLM/GGUF] Loaded {} BPE tokens from GGUF metadata", vocab.len());

        let gguf_data = Arc::new(data);

        eprintln!("[NeuronLM/GGUF] Ready — {} layers, {} vocab, d={}, GQA {}/{}",
            num_layers, vocab_size, d, num_heads, num_kv_heads);

        Ok(Self {
            embed_dim: d, num_heads, num_kv_heads, vocab_size, num_layers,
            d_ff, rope_base, rms_eps,
            w_te, w_pe: vec![vec![0.0; 1]; 1],
            w_q: vec![vec![0.0; 1]; 1], w_k: vec![vec![0.0; 1]; 1],
            w_v: vec![vec![0.0; 1]; 1], w_out: vec![vec![0.0; 1]; 1],
            w_ff1: vec![vec![0.0; 1]; 1], w_ff2: vec![vec![0.0; 1]; 1],
            w_ln_g: vec![1.0; d], w_ln_b: vec![0.0; d],
            output_norm, output_weight: Vec::new(),
            is_trained: true, is_qwen: true,
            weight_dir: String::new(),
            vocab,
            attn_blocks: vec![true; num_layers],
            is_byte_native: false,
            native_layers: Vec::new(),
            gguf_data: Some(gguf_data),
            gguf_tensors: model.tensors,
        })
    }

    /// Load GGUF from a file path (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_from_gguf_file(path: &str) -> Result<Self, String> {
        eprintln!("[NeuronLM] Reading GGUF file: {}", path);
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read GGUF file: {}", e))?;
        eprintln!("[NeuronLM] Read {} MB", data.len() / 1024 / 1024);
        Self::new_from_gguf_bytes(data)
    }

    /// Extract vocabulary from GGUF metadata tokens
    fn extract_vocab_from_gguf(model: &gguf::GgufModel) -> Vec<String> {
        if let Some(gguf::GgufMetaValue::Array(tokens)) = model.metadata.get("tokenizer.ggml.tokens") {
            tokens.iter().map(|t| {
                if let gguf::GgufMetaValue::Str(s) = t {
                    s.clone()
                } else {
                    String::new()
                }
            }).collect()
        } else {
            Vec::new()
        }
    }

    /// Get raw bytes for a GGUF tensor by name
    fn gguf_weight(&self, name: &str) -> Option<(&[u8], &GgufTensor)> {
        if let Some(ref data) = self.gguf_data {
            if let Some(tensor) = self.gguf_tensors.get(name) {
                let end = tensor.offset + tensor.byte_size();
                if end <= data.len() {
                    return Some((&data[tensor.offset..end], tensor));
                }
            }
        }
        None
    }

    pub fn generate_reply(&self, prompt: &str) -> String {
        if self.is_byte_native || (!self.native_layers.is_empty() && self.gguf_data.is_none()) {
            return self.generate_byte(prompt);
        }
        if self.gguf_data.is_some() || self.is_qwen {
            return self.generate_llama(prompt);
        }
        // Fallback
        format!("[NeuronLM]: {}", prompt)
    }

    /// Full Llama forward pass with KV cache for fast autoregressive generation
    fn generate_llama(&self, prompt: &str) -> String {
        // Apply ChatML/TinyLlama chat template if not already present
        let formatted = if prompt.contains("<|user|>") || prompt.contains("<|system|>") {
            prompt.to_string()
        } else {
            format!("<|user|>\n{}</s>\n<|assistant|>\n", prompt)
        };

        // Tokenize with BOS
        let bos_token = if self.vocab_size < 50000 { 1usize } else { 128000usize }; // TinyLlama vs Llama
        let mut token_ids = vec![bos_token];
        token_ids.extend(self.tokenize_bpe(&formatted));
        eprintln!("[NeuronLM] Tokenized to {} tokens: {:?}", token_ids.len(), &token_ids);

        let d = self.embed_dim;
        let num_heads = self.num_heads;
        let _num_kv = self.num_kv_heads;
        let _head_dim = d / num_heads;
        let seq_len = token_ids.len();

        // Build embedding matrix [seq_len, d]
        let mut h: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &tid in &token_ids {
            if tid < self.w_te.len() {
                h.push(self.w_te[tid].clone());
            } else {
                h.push(vec![0.0f32; d]);
            }
        }

        // KV cache: [layer][position] -> (K[num_kv][head_dim], V[num_kv][head_dim])
        let mut kv_cache: Vec<Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> =
            vec![Vec::new(); self.num_layers];

        eprintln!("[NeuronLM] Running {} tokens × {} layers (building KV cache)...", seq_len, self.num_layers);

        // Forward pass through all layers, building KV cache
        for layer in 0..self.num_layers {
            self.llama_block_forward_cached(&mut h, layer, &mut kv_cache[layer]);
            if layer % 8 == 7 || layer == self.num_layers - 1 {
                eprintln!("[NeuronLM] Block {}/{} done (cached {} positions)", layer, self.num_layers - 1, kv_cache[layer].len());
            }
        }

        // Final RMSNorm on last position
        let h_final = rms_norm_f32(&h[seq_len - 1], &self.output_norm, self.rms_eps);

        // Generate tokens autoregressively with KV cache
        let mut generated_tokens = Vec::new();
        let mut last_h = h_final;

        for step in 0..128 {
            // Project to vocabulary — use GGUF quantized output weight if available
            let mut logits = vec![0.0f32; self.vocab_size];
            if let Some((w_data, tensor)) = self.gguf_weight("output.weight") {
                gguf::qmv_mul(&mut logits, &last_h, w_data, self.vocab_size, d, tensor.tensor_type);
            } else {
                let n_vocab = self.output_weight.len() / d;
                logits.resize(n_vocab, 0.0);
                mat_vec_mul(&mut logits, &last_h, &self.output_weight, n_vocab, d);
            }

            let mut best_id = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (v, &val) in logits.iter().enumerate() {
                if v != 0 && val > best_score { // allow EOS (id 2) but skip unk (id 0)
                    best_score = val;
                    best_id = v;
                }
            }

            let tok_str = if best_id < self.vocab.len() {
                self.vocab[best_id].clone()
            } else {
                "?".to_string()
            };
            eprintln!("[NeuronLM] Token {}: id={} '{}' score={:.2}", step, best_id, tok_str, best_score);

            // Check EOS (handle both TinyLlama and Llama EOS tokens)
            if best_id == 2 || best_id == 128001 || best_id == 128009 { break; }

            generated_tokens.push(best_id);
            if generated_tokens.len() >= 128 { break; }

            // Single-token forward pass with KV cache
            let pos = seq_len + step; // position in the full sequence
            if best_id < self.w_te.len() {
                let mut h_tok = self.w_te[best_id].clone();
                for layer in 0..self.num_layers {
                    self.llama_single_token_cached(&mut h_tok, layer, pos, &mut kv_cache[layer]);
                }
                last_h = rms_norm_f32(&h_tok, &self.output_norm, self.rms_eps);
            }
        }

        // Decode tokens
        let mut result = String::new();
        for &tid in &generated_tokens {
            if tid < self.vocab.len() {
                let tok = &self.vocab[tid];
                let clean = tok.replace("Ġ", " ").replace("Ċ", "\n");
                result.push_str(&clean);
            }
        }

        format!("[NeuronLM]: {}", result.trim())
    }

    /// Compute out = x @ W^T using native layers, GGUF quantized tensor, or f32 fallback
    fn mat_vec_gguf(&self, out: &mut [f32], x: &[f32], gguf_name: &str, f32_name: &str, n_out: usize, n_in: usize) -> bool {
        if !self.native_layers.is_empty() && f32_name.starts_with("blk_") && f32_name.len() >= 6 {
            if let Ok(layer_idx) = f32_name[4..6].parse::<usize>() {
                if layer_idx < self.native_layers.len() {
                    let layer = &self.native_layers[layer_idx];
                    let suffix = &f32_name[6..];
                    let w_slice = match suffix {
                        "_attn_q" => Some(&layer.attn_q),
                        "_attn_k" => Some(&layer.attn_k),
                        "_attn_v" => Some(&layer.attn_v),
                        "_attn_output" => Some(&layer.attn_out),
                        "_ffn_gate" => Some(&layer.ffn_gate),
                        "_ffn_up" => Some(&layer.ffn_up),
                        "_ffn_down" => Some(&layer.ffn_down),
                        _ => None,
                    };
                    if let Some(w) = w_slice {
                        mat_vec_mul(out, x, w, n_out, n_in);
                        return true;
                    }
                }
            }
        }
        if let Some((w_data, tensor)) = self.gguf_weight(gguf_name) {
            gguf::qmv_mul(out, x, w_data, n_out, n_in, tensor.tensor_type);
            return true;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(w) = load_weight_f32(&self.weight_dir, f32_name) {
            mat_vec_mul(out, x, &w, n_out, n_in);
            return true;
        }
        false
    }

    /// Load a norm vector from native layers, GGUF, or f32 file
    fn norm_gguf(&self, gguf_name: &str, f32_name: &str, _d: usize) -> Option<Vec<f32>> {
        if !self.native_layers.is_empty() && f32_name.starts_with("blk_") && f32_name.len() >= 6 {
            if let Ok(layer_idx) = f32_name[4..6].parse::<usize>() {
                if layer_idx < self.native_layers.len() {
                    let layer = &self.native_layers[layer_idx];
                    let suffix = &f32_name[6..];
                    match suffix {
                        "_attn_norm" => return Some(layer.attn_norm.clone()),
                        "_ffn_norm" => return Some(layer.ffn_norm.clone()),
                        _ => {}
                    }
                }
            }
        }
        if let Some((data, tensor)) = self.gguf_weight(gguf_name) {
            return Some(gguf::dequantize_tensor(data, tensor.n_elements(), tensor.tensor_type));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(v) = load_weight_f32(&self.weight_dir, f32_name) {
            return Some(v);
        }
        None
    }

    /// Sequence-level forward pass with KV cache building
    /// cache: Vec of (K_heads, V_heads) per position, built during this call
    fn llama_block_forward_cached(&self, h: &mut Vec<Vec<f32>>, layer: usize,
                                   cache: &mut Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>) {
        let d = self.embed_dim;
        let d_ff = self.d_ff;
        let seq_len = h.len();
        let gguf_prefix = format!("blk.{}", layer);
        let f32_prefix = format!("blk_{:02}", layer);
        let num_heads = self.num_heads;
        let num_kv = self.num_kv_heads;
        let head_dim = d / num_heads;
        let heads_per_kv = num_heads / num_kv;
        let kv_dim = num_kv * head_dim;

        // === Attention ===
        let an = self.norm_gguf(
            &format!("{}.attn_norm.weight", gguf_prefix),
            &format!("{}_attn_norm", f32_prefix), d);

        if let Some(an) = an {
            let h_normed: Vec<Vec<f32>> = h.iter()
                .map(|x| rms_norm_f32(x, &an, self.rms_eps))
                .collect();

            let mut all_q = vec![vec![vec![0.0f32; head_dim]; num_heads]; seq_len];
            let mut all_k = vec![vec![vec![0.0f32; head_dim]; num_kv]; seq_len];
            let mut all_v = vec![vec![vec![0.0f32; head_dim]; num_kv]; seq_len];
            for s in 0..seq_len {
                let mut q_flat = vec![0.0f32; d];
                self.mat_vec_gguf(&mut q_flat, &h_normed[s],
                    &format!("{}.attn_q.weight", gguf_prefix),
                    &format!("{}_attn_q", f32_prefix), d, d);
                for out_idx in 0..d {
                    all_q[s][out_idx / head_dim][out_idx % head_dim] = q_flat[out_idx];
                }

                let mut k_flat = vec![0.0f32; kv_dim];
                let mut v_flat = vec![0.0f32; kv_dim];
                self.mat_vec_gguf(&mut k_flat, &h_normed[s],
                    &format!("{}.attn_k.weight", gguf_prefix),
                    &format!("{}_attn_k", f32_prefix), kv_dim, d);
                self.mat_vec_gguf(&mut v_flat, &h_normed[s],
                    &format!("{}.attn_v.weight", gguf_prefix),
                    &format!("{}_attn_v", f32_prefix), kv_dim, d);
                for out_idx in 0..kv_dim {
                    all_k[s][out_idx / head_dim][out_idx % head_dim] = k_flat[out_idx];
                    all_v[s][out_idx / head_dim][out_idx % head_dim] = v_flat[out_idx];
                }

                for head in 0..num_heads {
                    apply_rope_f32(&mut all_q[s][head], s, self.rope_base);
                }
                for head in 0..num_kv {
                    apply_rope_f32(&mut all_k[s][head], s, self.rope_base);
                }

                // Store K, V in cache
                cache.push((all_k[s].clone(), all_v[s].clone()));
            }

            // Causal attention with GQA
            let scale = (head_dim as f32).sqrt();
            let mut attn_out = vec![vec![0.0f32; d]; seq_len];

            for q_head in 0..num_heads {
                let kv_head = q_head / heads_per_kv;
                for i in 0..seq_len {
                    let mut scores = vec![f32::NEG_INFINITY; seq_len];
                    for j in 0..=i {
                        let mut dot = 0.0f32;
                        for hd in 0..head_dim {
                            dot += all_q[i][q_head][hd] * all_k[j][kv_head][hd];
                        }
                        scores[j] = dot / scale;
                    }
                    let max_val = scores[0..=i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut exp_sum = 0.0f32;
                    for j in 0..=i { scores[j] = (scores[j] - max_val).exp(); exp_sum += scores[j]; }
                    for j in 0..=i { scores[j] /= exp_sum.max(1e-12); }

                    let ho = q_head * head_dim;
                    for hd in 0..head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..=i { sum += scores[j] * all_v[j][kv_head][hd]; }
                        attn_out[i][ho + hd] = sum;
                    }
                }
            }

            for s in 0..seq_len {
                let mut wo_out = vec![0.0f32; d];
                self.mat_vec_gguf(&mut wo_out, &attn_out[s],
                    &format!("{}.attn_output.weight", gguf_prefix),
                    &format!("{}_attn_output", f32_prefix), d, d);
                for out_idx in 0..d {
                    h[s][out_idx] += wo_out[out_idx];
                }
            }
        }

        // === SwiGLU FFN ===
        let pn = self.norm_gguf(
            &format!("{}.ffn_norm.weight", gguf_prefix),
            &format!("{}_ffn_norm", f32_prefix), d);

        if let Some(pn) = pn {
            for s in 0..seq_len {
                let h_norm = rms_norm_f32(&h[s], &pn, self.rms_eps);
                let mut gate_val = vec![0.0f32; d_ff];
                let mut up_val = vec![0.0f32; d_ff];
                self.mat_vec_gguf(&mut gate_val, &h_norm,
                    &format!("{}.ffn_gate.weight", gguf_prefix),
                    &format!("{}_ffn_gate", f32_prefix), d_ff, d);
                self.mat_vec_gguf(&mut up_val, &h_norm,
                    &format!("{}.ffn_up.weight", gguf_prefix),
                    &format!("{}_ffn_up", f32_prefix), d_ff, d);

                let mut ffn_inter = vec![0.0f32; d_ff];
                for i in 0..d_ff {
                    ffn_inter[i] = silu_f32(gate_val[i]) * up_val[i];
                }

                let mut ffn_out = vec![0.0f32; d];
                self.mat_vec_gguf(&mut ffn_out, &ffn_inter,
                    &format!("{}.ffn_down.weight", gguf_prefix),
                    &format!("{}_ffn_down", f32_prefix), d, d_ff);
                for out_idx in 0..d {
                    h[s][out_idx] += ffn_out[out_idx];
                }
            }
        }
    }

    /// Single-token forward pass using KV cache
    /// pos: absolute position of this token in the sequence
    fn llama_single_token_cached(&self, h: &mut Vec<f32>, layer: usize, pos: usize,
                                  cache: &mut Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>) {
        let d = self.embed_dim;
        let d_ff = self.d_ff;
        let gguf_prefix = format!("blk.{}", layer);
        let f32_prefix = format!("blk_{:02}", layer);
        let num_heads = self.num_heads;
        let num_kv = self.num_kv_heads;
        let head_dim = d / num_heads;
        let heads_per_kv = num_heads / num_kv;
        let kv_dim = num_kv * head_dim;

        // === Attention with KV cache ===
        let an = self.norm_gguf(
            &format!("{}.attn_norm.weight", gguf_prefix),
            &format!("{}_attn_norm", f32_prefix), d);

        if let Some(an) = an {
            let h_normed = rms_norm_f32(h, &an, self.rms_eps);

            // Compute Q
            let mut q_flat = vec![0.0f32; d];
            self.mat_vec_gguf(&mut q_flat, &h_normed,
                &format!("{}.attn_q.weight", gguf_prefix),
                &format!("{}_attn_q", f32_prefix), d, d);
            let mut q_heads = vec![vec![0.0f32; head_dim]; num_heads];
            for i in 0..d {
                q_heads[i / head_dim][i % head_dim] = q_flat[i];
            }

            // Compute K, V
            let mut k_flat = vec![0.0f32; kv_dim];
            let mut v_flat = vec![0.0f32; kv_dim];
            self.mat_vec_gguf(&mut k_flat, &h_normed,
                &format!("{}.attn_k.weight", gguf_prefix),
                &format!("{}_attn_k", f32_prefix), kv_dim, d);
            self.mat_vec_gguf(&mut v_flat, &h_normed,
                &format!("{}.attn_v.weight", gguf_prefix),
                &format!("{}_attn_v", f32_prefix), kv_dim, d);
            let mut k_heads = vec![vec![0.0f32; head_dim]; num_kv];
            let mut v_heads = vec![vec![0.0f32; head_dim]; num_kv];
            for i in 0..kv_dim {
                k_heads[i / head_dim][i % head_dim] = k_flat[i];
                v_heads[i / head_dim][i % head_dim] = v_flat[i];
            }

            // Apply RoPE
            for head in 0..num_heads { apply_rope_f32(&mut q_heads[head], pos, self.rope_base); }
            for head in 0..num_kv { apply_rope_f32(&mut k_heads[head], pos, self.rope_base); }

            // Append to cache
            cache.push((k_heads.clone(), v_heads.clone()));

            // Attention over all cached positions
            let n_cached = cache.len();
            let scale = (head_dim as f32).sqrt();
            let mut attn_out = vec![0.0f32; d];

            for q_head in 0..num_heads {
                let kv_head = q_head / heads_per_kv;

                let mut scores = vec![0.0f32; n_cached];
                for j in 0..n_cached {
                    let mut dot = 0.0f32;
                    for hd in 0..head_dim { dot += q_heads[q_head][hd] * cache[j].0[kv_head][hd]; }
                    scores[j] = dot / scale;
                }

                // Softmax
                let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for j in 0..n_cached { scores[j] = (scores[j] - max_val).exp(); exp_sum += scores[j]; }
                for j in 0..n_cached { scores[j] /= exp_sum.max(1e-12); }

                let ho = q_head * head_dim;
                for hd in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..n_cached { sum += scores[j] * cache[j].1[kv_head][hd]; }
                    attn_out[ho + hd] = sum;
                }
            }

            // Output projection + residual
            let mut wo_out = vec![0.0f32; d];
            self.mat_vec_gguf(&mut wo_out, &attn_out,
                &format!("{}.attn_output.weight", gguf_prefix),
                &format!("{}_attn_output", f32_prefix), d, d);
            for i in 0..d { h[i] += wo_out[i]; }
        }

        // === SwiGLU FFN ===
        let pn = self.norm_gguf(
            &format!("{}.ffn_norm.weight", gguf_prefix),
            &format!("{}_ffn_norm", f32_prefix), d);

        if let Some(pn) = pn {
            let h_norm = rms_norm_f32(h, &pn, self.rms_eps);
            let mut gate_val = vec![0.0f32; d_ff];
            let mut up_val = vec![0.0f32; d_ff];
            self.mat_vec_gguf(&mut gate_val, &h_norm,
                &format!("{}.ffn_gate.weight", gguf_prefix),
                &format!("{}_ffn_gate", f32_prefix), d_ff, d);
            self.mat_vec_gguf(&mut up_val, &h_norm,
                &format!("{}.ffn_up.weight", gguf_prefix),
                &format!("{}_ffn_up", f32_prefix), d_ff, d);

            let mut ffn_inter = vec![0.0f32; d_ff];
            for i in 0..d_ff {
                ffn_inter[i] = silu_f32(gate_val[i]) * up_val[i];
            }

            let mut ffn_out = vec![0.0f32; d];
            self.mat_vec_gguf(&mut ffn_out, &ffn_inter,
                &format!("{}.ffn_down.weight", gguf_prefix),
                &format!("{}_ffn_down", f32_prefix), d, d_ff);
            for i in 0..d {
                h[i] += ffn_out[i];
            }
        }
    }

    #[allow(dead_code)]
    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| c as usize).collect()
    }

    /// BPE tokenization: greedy longest-match with proper space handling
    fn tokenize_bpe(&self, text: &str) -> Vec<usize> {
        if self.vocab.is_empty() {
            return text.chars().map(|c| c as usize).collect();
        }

        let mut vocab_map: HashMap<String, usize> = HashMap::new();
        for (id, tok) in self.vocab.iter().enumerate() {
            vocab_map.insert(tok.clone(), id);
        }

        let mut tokens = Vec::new();
        let mut pos = 0;
        let chars: Vec<char> = text.chars().collect();

        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;

            let max_check = 20.min(chars.len() - pos);
            for len in (1..=max_check).rev() {
                let substr: String = chars[pos..pos+len].iter().collect();

                if let Some(&id) = vocab_map.get(&substr) {
                    best_len = len;
                    best_id = Some(id);
                    break;
                }

                if substr.starts_with(' ') {
                    let rest = &substr[1..];
                    // Try GPT-style BPE prefix (Ġ = U+0120)
                    let bpe_form = format!("Ġ{}", rest);
                    if let Some(&id) = vocab_map.get(&bpe_form) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                    // Try SentencePiece prefix (▁ = U+2581)
                    let sp_form = format!("\u{2581}{}", rest);
                    if let Some(&id) = vocab_map.get(&sp_form) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                pos += best_len;
            } else {
                tokens.push(chars[pos] as usize);
                pos += 1;
            }
        }

        tokens
    }
}

/// Matrix-Vector Multiplication: out = X @ W^T
/// W is [n_out, n_in] flat array.
/// Uses rayon parallelism on native, single-threaded on WASM.
#[cfg(feature = "native")]
fn mat_vec_mul(out: &mut [f32], x: &[f32], w: &[f32], _n_out: usize, n_in: usize) {
    out.par_iter_mut().enumerate().for_each(|(out_idx, val)| {
        let offset = out_idx * n_in;
        let mut sum = 0.0f32;
        let w_row = &w[offset..offset + n_in];
        for in_idx in 0..n_in {
            sum += x[in_idx] * w_row[in_idx];
        }
        *val = sum;
    });
}

/// Single-threaded Matrix-Vector Multiplication for WASM
#[cfg(not(feature = "native"))]
fn mat_vec_mul(out: &mut [f32], x: &[f32], w: &[f32], n_out: usize, n_in: usize) {
    for out_idx in 0..n_out {
        let offset = out_idx * n_in;
        let mut sum = 0.0f32;
        let w_row = &w[offset..offset + n_in];
        for in_idx in 0..n_in {
            sum += x[in_idx] * w_row[in_idx];
        }
        out[out_idx] = sum;
    }
}

/// RoPE with configurable base
fn apply_rope_f32(x: &mut [f32], pos: usize, base: f32) {
    let d = x.len();
    for i in (0..d).step_by(2) {
        if i + 1 >= d { break; }
        let freq = 1.0 / base.powf(i as f32 / d as f32);
        let angle = pos as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let x0 = x[i];
        let x1 = x[i + 1];
        x[i]     = x0 * cos_a - x1 * sin_a;
        x[i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x.clamp(-88.0, 88.0)).exp())
}

fn rms_norm_f32(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        sum_sq += x[i] * x[i];
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    x.iter().enumerate().map(|(i, &v)| {
        let w = if i < weight.len() { weight[i] } else { 1.0 };
        v / rms * w
    }).collect()
}

// Legacy helpers for small model
#[allow(dead_code)]
fn matmul_2d(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    if rows_a == 0 { return vec![]; }
    let cols_a = a[0].len();
    let cols_b = b[0].len();
    let mut res = vec![vec![0.0; cols_b]; rows_a];
    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a { sum += a[i][k] * b[k][j]; }
            res[i][j] = sum;
        }
    }
    res
}

#[allow(dead_code)]
fn layernorm_2d(a: &[Vec<f64>], gamma: &[f64], beta: &[f64]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut res = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        let mean: f64 = a[i].iter().sum::<f64>() / cols as f64;
        let var: f64 = a[i].iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / cols as f64;
        let std = (var + 1e-5).sqrt();
        for j in 0..cols {
            res[i][j] = gamma[j] * ((a[i][j] - mean) / std) + beta[j];
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_byte_model_init_and_generation() {
        let model = NeuronLM::new_byte_model(32, 2, 1, 64);
        assert_eq!(model.vocab_size, 256);
        assert!(model.is_byte_native);
        assert_eq!(model.native_layers.len(), 1);

        let reply = model.generate_byte("Hello NEURON");
        assert!(reply.starts_with("[NeuronLM/Byte]:"));
    }

    #[test]
    fn test_native_byte_model_training_and_serialization() {
        let mut model = NeuronLM::new_byte_model(32, 2, 1, 64);
        let corpus = "Hello world! NEURON is token-free. 2 + 2 = 4.";
        let initial_loss = model.train_byte_model(corpus, 5, 0.01);
        let final_loss = model.train_byte_model(corpus, 30, 0.01);
        assert!(final_loss < initial_loss, "Training should reduce cross-entropy loss: initial={}, final={}", initial_loss, final_loss);

        // Test serialization
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_neuron_byte_model.bin");
        let path_str = path.to_str().unwrap();

        model.save_byte_model(path_str).expect("Failed to save byte model");
        let loaded = NeuronLM::load_byte_model(path_str).expect("Failed to load byte model");
        assert_eq!(loaded.embed_dim, model.embed_dim);
        assert_eq!(loaded.num_layers, model.num_layers);
        assert_eq!(loaded.vocab_size, 256);
        assert_eq!(loaded.w_te[65], model.w_te[65]);

        let _ = std::fs::remove_file(path);
    }
}
