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
    // GGUF direct loading — quantized weights dequantized on-the-fly
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

impl NeuronLM {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Check if correct model weights exist
            let model_dir = "data/model_weights";
            let vocab_path = format!("{}/vocab.json", model_dir);
            let has_model = std::path::Path::new(&format!("{}/blk_00_attn_q.f32", model_dir)).exists()
                && std::path::Path::new(&vocab_path).exists();

            if has_model {
                return Self::new_llama(model_dir);
            }
        }

        // Fallback: small demo model
        let embed_dim = 8;
        let num_heads = 2;
        let vocab_size = 128;
        let max_seq_len = 32;

        let mut w_te_f32 = vec![vec![0.0f32; embed_dim]; vocab_size];
        for i in 0..vocab_size {
            for j in 0..embed_dim {
                w_te_f32[i][j] = ((i + j) as f32).sin() * 0.1;
            }
        }

        let w_pe = (0..max_seq_len)
            .map(|i| (0..embed_dim).map(|j| ((i * j) as f64).cos() * 0.05).collect())
            .collect();

        Self {
            embed_dim, num_heads, num_kv_heads: num_heads, vocab_size, num_layers: 1,
            d_ff: embed_dim * 2, rope_base: 10000.0, rms_eps: 1e-5,
            w_te: w_te_f32, w_pe,
            w_q: vec![vec![0.1; embed_dim]; embed_dim],
            w_k: vec![vec![0.1; embed_dim]; embed_dim],
            w_v: vec![vec![0.2; embed_dim]; embed_dim],
            w_out: vec![vec![0.15; embed_dim]; embed_dim],
            w_ff1: vec![vec![0.25; embed_dim * 2]; embed_dim],
            w_ff2: vec![vec![0.1; embed_dim]; embed_dim * 2],
            w_ln_g: vec![1.0; embed_dim],
            w_ln_b: vec![0.0; embed_dim],
            output_norm: Vec::new(),
            output_weight: Vec::new(),
            is_trained: false, is_qwen: false,
            weight_dir: String::new(),
            vocab: Vec::new(),
            attn_blocks: vec![true],
            gguf_data: None,
            gguf_tensors: HashMap::new(),
        }
    }

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
        if self.gguf_data.is_some() || self.is_qwen {
            return self.generate_llama(prompt);
        }
        // Small model fallback
        format!("[AGI Response]: I heard: {}", prompt)
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

    /// Compute out = x @ W^T using GGUF quantized tensor or f32 fallback
    fn mat_vec_gguf(&self, out: &mut [f32], x: &[f32], gguf_name: &str, f32_name: &str, n_out: usize, n_in: usize) -> bool {
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

    /// Load a norm vector from GGUF or f32 file
    fn norm_gguf(&self, gguf_name: &str, f32_name: &str, _d: usize) -> Option<Vec<f32>> {
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
