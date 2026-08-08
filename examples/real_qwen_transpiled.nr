// Auto-transpiled from examples/real_qwen_inference.py by NEURON Python Transpiler
// Source: examples/real_qwen_inference.py

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")
let WEIGHT_DIR = "C:\Users\ADMIN\neuron-lang\data\qwen_layers"
let VOCAB_PATH = "C:\Users\ADMIN\neuron-lang\data\qwen_vocab.json"
fn rms_norm(x: Tensor, weight: Tensor, eps: Tensor) -> Tensor:
  "RMSNorm: x * weight / sqrt(mean(x^2) + eps)"
  let rms = sqrt(np.mean(x * x, axis=-1, keepdims=true) + eps)
  return x / rms * weight

fn softmax(x: Tensor) -> Tensor:
  "Numerically stable softmax."
  let x_max = np.max(x, axis=-1, keepdims=true)
  let e_x = exp(x - x_max)
  return e_x / np.sum(e_x, axis=-1, keepdims=true)

fn silu(x: Tensor) -> Tensor:
  "SiLU / Swish activation: x * sigmoid(x)"
  return x * 1.0 / 1.0 + exp(-x)

fn load_weight(name: Tensor) -> Tensor:
  "Load a dequantized weight tensor from .bin file."
  let path = os.path.join(WEIGHT_DIR, /* UNSUPPORTED: JoinedStr(values=[FormattedValue(value=Name(id='name', ctx=Load()), conversion=-1), Constant(value='.bin')]) */)
  return np.fromfile(path, dtype=np.float64)

fn main():
  print("=" * 60)
  print("  Qwen 3.5 2B - Real Trained Weight Inference")
  print("  Using dequantized GGUF weights from local Ollama model")
  print("=" * 60)
  print("
[1/5] Loading real BPE vocabulary...")
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Vocabulary size: '), FormattedValue(value=Call(func=Name(id='len', ctx=Load()), args=[Name(id='vocab', ctx=Load())], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value=',')])), Constant(value=' tokens')]) */)
  print("
[2/5] Loading dequantized trained weights (Block 0)...")
  let attn_norm = load_weight("blk_0_attn_norm_weight").reshape(2048)
  let attn_qkv = load_weight("blk_0_attn_qkv_weight").reshape(2048, 6144)
  let attn_gate = load_weight("blk_0_attn_gate_weight").reshape(2048, 2048)
  let post_norm = load_weight("blk_0_post_attention_norm_weight").reshape(2048)
  let ffn_gate = load_weight("blk_0_ffn_gate_weight").reshape(2048, 6144)
  let ffn_up = load_weight("blk_0_ffn_up_weight").reshape(2048, 6144)
  let ffn_down = load_weight("blk_0_ffn_down_weight").reshape(6144, 2048)
  let out_norm = load_weight("output_norm_weight").reshape(2048)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  attn_qkv:  '), FormattedValue(value=Attribute(value=Name(id='attn_qkv', ctx=Load()), attr='shape', ctx=Load()), conversion=-1), Constant(value='  mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='attn_qkv', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  ffn_gate:  '), FormattedValue(value=Attribute(value=Name(id='ffn_gate', ctx=Load()), attr='shape', ctx=Load()), conversion=-1), Constant(value='  mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='ffn_gate', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  ffn_down:  '), FormattedValue(value=Attribute(value=Name(id='ffn_down', ctx=Load()), attr='shape', ctx=Load()), conversion=-1), Constant(value='  mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='ffn_down', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  print("
[3/5] Creating prompt embedding from trained token_embd...")
  let embd_path = os.path.join(WEIGHT_DIR, "token_embd_weight.bin")
  let embd_size = os.path.getsize(embd_path)
  let n_tokens_to_load = 1000
  let token_embd_partial = np.fromfile(embd_path, dtype=np.float64, count=n_tokens_to_load * 2048).reshape(n_tokens_to_load, 2048)
  let prompt_token_ids = [198, 220, 350, 450, 500]
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Prompt token IDs: '), FormattedValue(value=Name(id='prompt_token_ids', ctx=Load()), conversion=-1)]) */)
  let prompt_tokens = /* UNSUPPORTED: ListComp(elt=IfExp(test=Compare(left=Name(id='tid', ctx=Load()), ops=[Lt()], comparators=[Call(func=Name(id='len', ctx=Load()), args=[Name(id='vocab', ctx=Load())], keywords=[])]), body=Subscript(value=Name(id='vocab', ctx=Load()), slice=Name(id='tid', ctx=Load()), ctx=Load()), orelse=Constant(value='?')), generators=[comprehension(target=Name(id='tid', ctx=Store()), iter=Name(id='prompt_token_ids', ctx=Load()), ifs=[], is_async=0)]) */
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Prompt tokens: '), FormattedValue(value=Name(id='prompt_tokens', ctx=Load()), conversion=-1)]) */)
  let prompt_emb = np.mean(/* UNSUPPORTED: ListComp(elt=Subscript(value=Name(id='token_embd_partial', ctx=Load()), slice=Name(id='tid', ctx=Load()), ctx=Load()), generators=[comprehension(target=Name(id='tid', ctx=Store()), iter=Name(id='prompt_token_ids', ctx=Load()), ifs=[], is_async=0)]) */, axis=0).reshape(1, 2048)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Prompt embedding: shape='), FormattedValue(value=Attribute(value=Name(id='prompt_emb', ctx=Load()), attr='shape', ctx=Load()), conversion=-1), Constant(value=', mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='prompt_emb', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  print("
[4/5] Running Block 0 Transformer Forward Pass (REAL weights)...")
  let h = rms_norm(prompt_emb, attn_norm)
  let qkv = h @ attn_qkv
  let (q, k, v) = np.split(qkv, 3, axis=-1)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Q mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='q', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')])), Constant(value=', K mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='k', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')])), Constant(value=', V mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='v', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  let scores = q @ k.T
  let scores = scores / sqrt(2048.0)
  let attn_weights = softmax(scores)
  let attn_out = attn_weights @ v
  let gated = attn_out @ attn_gate
  let h = prompt_emb + gated
  let h = rms_norm(h, post_norm)
  let gate_val = silu(h @ ffn_gate)
  let up_val = h @ ffn_up
  let ffn_hidden = gate_val * up_val
  let ffn_out = ffn_hidden @ ffn_down
  let output = prompt_emb + gated + ffn_out
  let output = rms_norm(output, out_norm)
  print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  Block 0 output: shape='), FormattedValue(value=Attribute(value=Name(id='output', ctx=Load()), attr='shape', ctx=Load()), conversion=-1), Constant(value=', mean='), FormattedValue(value=Call(func=Attribute(value=Name(id='output', ctx=Load()), attr='mean', ctx=Load()), args=[], keywords=[]), conversion=-1, format_spec=JoinedStr(values=[Constant(value='.6f')]))]) */)
  print("
[5/5] Projecting to vocabulary and decoding tokens...")
  let logits = output @ token_embd_partial.T.flatten()
  let top_ids = np.argsort(logits)[-10:][:]
  print("
  === TOP 10 PREDICTED NEXT TOKENS (Real Trained Weights) ===")
  for (rank, tid) in enumerate(top_ids):
    let token_str = if tid < len(vocab) then vocab[tid] else /* UNSUPPORTED: JoinedStr(values=[Constant(value='<'), FormattedValue(value=Name(id='tid', ctx=Load()), conversion=-1), Constant(value='>')]) */
    let score = logits[tid]
    print(/* UNSUPPORTED: JoinedStr(values=[Constant(value='  #'), FormattedValue(value=BinOp(left=Name(id='rank', ctx=Load()), op=Add(), right=Constant(value=1)), conversion=-1), Constant(value=': Token ID '), FormattedValue(value=Name(id='tid', ctx=Load()), conversion=-1, format_spec=JoinedStr(values=[Constant(value='>5')])), Constant(value=' | Score '), FormattedValue(value=Name(id='score', ctx=Load()), conversion=-1, format_spec=JoinedStr(values=[Constant(value='>10.4f')])), Constant(value=" | '"), FormattedValue(value=Name(id='token_str', ctx=Load()), conversion=-1), Constant(value="'")]) */)
  print("
  === GENERATED SEQUENCE (Greedy, Block 0 only) ===")
  print("  ", end="")
  let current_emb = output
  for step in range(15):
    let step_logits = current_emb @ token_embd_partial.T.flatten()
    let next_id = np.argmax(step_logits)
    let next_token = if next_id < len(vocab) then vocab[next_id] else /* UNSUPPORTED: JoinedStr(values=[Constant(value='<'), FormattedValue(value=Name(id='next_id', ctx=Load()), conversion=-1), Constant(value='>')]) */
    sys.stdout.write(next_token)
    sys.stdout.flush()
    let current_emb = rms_norm(token_embd_partial[next_id].reshape(1, 2048), out_norm)
  print("
")
  print("=" * 60)
  print("  [DONE] Real inference with dequantized trained weights.")
  print("  NOTE: Only Block 0/24 used. Full 24-layer = coherent text.")
  print("=" * 60)

fn main():
  main()
