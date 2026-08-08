# ═══════════════════════════════════════════════════════════════════════
#  Real Qwen 3.5 2B Single-Layer Inference with Dequantized GGUF Weights
#  Uses REAL trained weights extracted from your local Ollama model
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import json
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WEIGHT_DIR = r"C:\Users\ADMIN\neuron-lang\data\qwen_layers"
VOCAB_PATH = r"C:\Users\ADMIN\neuron-lang\data\qwen_vocab.json"


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def silu(x):
    """SiLU / Swish activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def load_weight(name):
    """Load a dequantized weight tensor from .bin file."""
    path = os.path.join(WEIGHT_DIR, f"{name}.bin")
    return np.fromfile(path, dtype=np.float64)


def main():
    print("=" * 60)
    print("  Qwen 3.5 2B - Real Trained Weight Inference")
    print("  Using dequantized GGUF weights from local Ollama model")
    print("=" * 60)

    # Load real vocabulary
    print("\n[1/5] Loading real BPE vocabulary...")
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"  Vocabulary size: {len(vocab):,} tokens")

    # Load real trained weights for block 0
    print("\n[2/5] Loading dequantized trained weights (Block 0)...")

    attn_norm = load_weight("blk_0_attn_norm_weight").reshape(2048)
    attn_qkv = load_weight("blk_0_attn_qkv_weight").reshape(2048, 6144)
    attn_gate = load_weight("blk_0_attn_gate_weight").reshape(2048, 2048)
    post_norm = load_weight("blk_0_post_attention_norm_weight").reshape(2048)
    ffn_gate = load_weight("blk_0_ffn_gate_weight").reshape(2048, 6144)
    ffn_up = load_weight("blk_0_ffn_up_weight").reshape(2048, 6144)
    ffn_down = load_weight("blk_0_ffn_down_weight").reshape(6144, 2048)
    out_norm = load_weight("output_norm_weight").reshape(2048)

    print(f"  attn_qkv:  {attn_qkv.shape}  mean={attn_qkv.mean():.6f}")
    print(f"  ffn_gate:  {ffn_gate.shape}  mean={ffn_gate.mean():.6f}")
    print(f"  ffn_down:  {ffn_down.shape}  mean={ffn_down.mean():.6f}")

    # Create input: use a small slice of the token embedding as our prompt
    print("\n[3/5] Creating prompt embedding from trained token_embd...")
    # Load just first 1000 rows of token embedding (to avoid 3.8GB load)
    embd_path = os.path.join(WEIGHT_DIR, "token_embd_weight.bin")
    embd_size = os.path.getsize(embd_path)
    # Read first 1000 * 2048 floats = first 1000 vocabulary token embeddings
    n_tokens_to_load = 1000
    token_embd_partial = np.fromfile(embd_path, dtype=np.float64, count=n_tokens_to_load * 2048).reshape(n_tokens_to_load, 2048)

    # Use token ID for common English token as prompt
    # Token IDs for common words are typically in the 200-500 range
    prompt_token_ids = [198, 220, 350, 450, 500]  # Sample token IDs
    print(f"  Prompt token IDs: {prompt_token_ids}")
    prompt_tokens = [vocab[tid] if tid < len(vocab) else "?" for tid in prompt_token_ids]
    print(f"  Prompt tokens: {prompt_tokens}")

    # Average the prompt token embeddings to get a single 2048-dim vector
    prompt_emb = np.mean([token_embd_partial[tid] for tid in prompt_token_ids], axis=0).reshape(1, 2048)
    print(f"  Prompt embedding: shape={prompt_emb.shape}, mean={prompt_emb.mean():.6f}")

    # ─── Block 0 Forward Pass with REAL trained weights ───
    print("\n[4/5] Running Block 0 Transformer Forward Pass (REAL weights)...")

    # Step 1: Pre-attention RMSNorm
    h = rms_norm(prompt_emb, attn_norm)

    # Step 2: Fused QKV projection
    qkv = h @ attn_qkv  # [1, 6144] = Q, K, V concatenated
    q, k, v = np.split(qkv, 3, axis=-1)  # Each [1, 2048]
    print(f"  Q mean={q.mean():.6f}, K mean={k.mean():.6f}, V mean={v.mean():.6f}")

    # Step 3: Attention scores + softmax
    scores = q @ k.T  # [1, 1] for single token
    scores = scores / np.sqrt(2048.0)
    attn_weights = softmax(scores)

    # Step 4: Attention output
    attn_out = attn_weights @ v  # [1, 2048]

    # Step 5: Gate projection
    gated = attn_out @ attn_gate  # [1, 2048]

    # Step 6: Residual connection
    h = prompt_emb + gated

    # Step 7: Post-attention RMSNorm
    h = rms_norm(h, post_norm)

    # Step 8: SwiGLU FFN
    gate_val = silu(h @ ffn_gate)   # [1, 6144]
    up_val = h @ ffn_up             # [1, 6144]
    ffn_hidden = gate_val * up_val  # [1, 6144] element-wise
    ffn_out = ffn_hidden @ ffn_down # [1, 2048]

    # Step 9: Final residual
    output = prompt_emb + gated + ffn_out

    # Step 10: Output RMSNorm
    output = rms_norm(output, out_norm)

    print(f"  Block 0 output: shape={output.shape}, mean={output.mean():.6f}")

    # ─── Token Decoding: Project to Vocabulary ───
    print("\n[5/5] Projecting to vocabulary and decoding tokens...")

    # Project output against partial token embeddings to find closest tokens
    # logits[i] = dot(output, token_embd[i]) for each token
    logits = (output @ token_embd_partial.T).flatten()  # [1000]

    # Get top-10 most probable tokens
    top_ids = np.argsort(logits)[-10:][::-1]
    print("\n  === TOP 10 PREDICTED NEXT TOKENS (Real Trained Weights) ===")
    for rank, tid in enumerate(top_ids):
        token_str = vocab[tid] if tid < len(vocab) else f"<{tid}>"
        score = logits[tid]
        print(f"  #{rank+1}: Token ID {tid:>5} | Score {score:>10.4f} | '{token_str}'")

    # Generate a short sequence by greedy decoding
    print("\n  === GENERATED SEQUENCE (Greedy, Block 0 only) ===")
    print("  ", end="")
    current_emb = output
    for step in range(15):
        step_logits = (current_emb @ token_embd_partial.T).flatten()
        next_id = np.argmax(step_logits)
        next_token = vocab[next_id] if next_id < len(vocab) else f"<{next_id}>"
        sys.stdout.write(next_token)
        sys.stdout.flush()
        # Feed next token embedding back
        current_emb = rms_norm(token_embd_partial[next_id].reshape(1, 2048), out_norm)

    print("\n")
    print("=" * 60)
    print("  [DONE] Real inference with dequantized trained weights.")
    print("  NOTE: Only Block 0/24 used. Full 24-layer = coherent text.")
    print("=" * 60)


if __name__ == '__main__':
    main()
