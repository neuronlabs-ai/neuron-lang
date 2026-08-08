// Auto-transpiled from examples/qwen_mini_inference.py by NEURON Python Transpiler
// Source: examples/qwen_mini_inference.py

model RMSNorm(dim: Int):
  weight: Tensor[dim] = zeros(dim)
  eps: Float = 1e-06

  fn forward(self, x: Tensor) -> Tensor:
    let norm = sqrt(mean(x * x) + self.eps)
    return x / norm * self.weight


model AttentionHead(d_model: Int, head_dim: Int):
  wq: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wk: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wv: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wo: Tensor[head_dim, d_model] = glorot(head_dim, d_model)

  fn forward(self, x: Tensor) -> Tensor:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = q @ transpose(k, 0, 1)
    let scale = 1.0 / sqrt(16.0)
    let scaled = scores * scale
    let attn = softmax(scaled)
    let context = attn @ v
    return context @ self.wo


model SwiGLU_FFN(d_model: Int, d_ff: Int):
  w_gate: Tensor[d_model, d_ff] = glorot(d_model, d_ff)
  w_up: Tensor[d_model, d_ff] = glorot(d_model, d_ff)
  w_down: Tensor[d_ff, d_model] = glorot(d_ff, d_model)

  fn forward(self, x: Tensor) -> Tensor:
    let gate = gelu(x @ self.w_gate)
    let up = x @ self.w_up
    let hidden = gate * up
    return hidden @ self.w_down


model TransformerBlock(d_model: Int, head_dim: Int, d_ff: Int):
  attn: Tensor = AttentionHead(d_model, head_dim)
  ffn: Tensor = SwiGLU_FFN(d_model, d_ff)

  fn forward(self, x: Tensor) -> Tensor:
    let attn_out = self.attn.forward(x)
    let h = x + attn_out
    let ffn_out = self.ffn.forward(h)
    return h + ffn_out


model QwenMiniLM(d_model: Int, d_ff: Int, vocab_size: Int):
  embed: Tensor[vocab_size, d_model] = glorot(vocab_size, d_model)
  block1: Tensor = TransformerBlock(d_model, 16, d_ff)
  block2: Tensor = TransformerBlock(d_model, 16, d_ff)
  lm_head: Tensor[d_model, vocab_size] = glorot(d_model, vocab_size)

  fn forward(self, input_ids: Tensor) -> Tensor:
    let x = input_ids @ self.embed
    let h1 = self.block1.forward(x)
    let h2 = self.block2.forward(h1)
    let logits = h2 @ self.lm_head
    return softmax(logits)

  fn generate_token(self, input_ids: Tensor) -> Tensor:
    let logits = self.forward(input_ids)
    return logits


fn main():
  print("Initializing QwenMiniLM Transformer...")
  let model_ = QwenMiniLM(64, 256, 128)
  let input_data = zeros(1, 128)
  let output = model_.forward(input_data)
  print("Generation complete!")
