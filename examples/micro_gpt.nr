// ═══════════════════════════════════════════════════════════════════
//  micro_gpt.nr — A Trainable MicroGPT Language Model in NEURON
//  Character-level transformer trained on TinyStories data.
//  Architecture: 4-layer, 4-head, 64-dim transformer decoder.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Self-Attention Layer
// ─────────────────────────────────────────────
model AttentionHead(d_model: Int, head_dim: Int):
  wq: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wk: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wv: Tensor[d_model, head_dim] = glorot(d_model, head_dim)

  fn forward(self, x: Tensor[63, d_model]) -> Tensor[63, head_dim]:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let kt = transpose(k, 0, 1)
    let scores = q @ kt
    let scale = 1.0 / 4.0
    let scaled = scores * scale
    let attn = softmax(scaled)
    return attn @ v

// ─────────────────────────────────────────────
//  Multi-Head Attention (4 heads)
// ─────────────────────────────────────────────
model MultiHeadAttention(d_model: Int, n_heads: Int, head_dim: Int):
  h1: AttentionHead = AttentionHead(d_model, head_dim)
  h2: AttentionHead = AttentionHead(d_model, head_dim)
  h3: AttentionHead = AttentionHead(d_model, head_dim)
  h4: AttentionHead = AttentionHead(d_model, head_dim)
  wo: Tensor[d_model, d_model] = glorot(d_model, d_model)

  fn forward(self, x: Tensor[63, d_model]) -> Tensor[63, d_model]:
    let o1 = self.h1.forward(x)
    let o2 = self.h2.forward(x)
    let o3 = self.h3.forward(x)
    let o4 = self.h4.forward(x)
    let combined = concat([o1, o2, o3, o4])
    return combined @ self.wo

// ─────────────────────────────────────────────
//  Feed-Forward Network
// ─────────────────────────────────────────────
model FeedForward(d_model: Int, d_ff: Int):
  w1: Tensor[d_model, d_ff] = glorot(d_model, d_ff)
  w2: Tensor[d_ff, d_model] = glorot(d_ff, d_model)

  fn forward(self, x: Tensor[63, d_model]) -> Tensor[63, d_model]:
    let h = gelu(x @ self.w1)
    return h @ self.w2

// ─────────────────────────────────────────────
//  Transformer Block (Attention + FFN + Residual)
// ─────────────────────────────────────────────
model TransformerBlock(d_model: Int, n_heads: Int, head_dim: Int, d_ff: Int):
  attn: MultiHeadAttention = MultiHeadAttention(d_model, n_heads, head_dim)
  ffn: FeedForward = FeedForward(d_model, d_ff)

  fn forward(self, x: Tensor[63, d_model]) -> Tensor[63, d_model]:
    // Self-attention with residual connection
    let attn_out = self.attn.forward(x)
    let h = x + attn_out
    // Feed-forward with residual connection
    let ff_out = self.ffn.forward(h)
    return h + ff_out

// ─────────────────────────────────────────────
//  MicroGPT Model
//  4 transformer layers, 64-dim embeddings, vocab=128
// ─────────────────────────────────────────────
model MicroGPT:
  // Token embedding: vocab_size x d_model
  token_embed: Tensor[128, 64] = glorot(128, 64)
  // Position embedding: max_seq_len x d_model
  pos_embed: Tensor[63, 64] = glorot(63, 64)

  // 4 Transformer layers
  block1: TransformerBlock = TransformerBlock(64, 4, 16, 256)
  block2: TransformerBlock = TransformerBlock(64, 4, 16, 256)
  block3: TransformerBlock = TransformerBlock(64, 4, 16, 256)
  block4: TransformerBlock = TransformerBlock(64, 4, 16, 256)

  // Output projection: d_model -> vocab_size
  output_proj: Tensor[64, 128] = glorot(64, 128)

  fn forward_sequence(self, input_ids: Tensor[63, 128]) -> Tensor[63, 128]:
    // input_ids[63, 128] @ token_embed[128, 64] -> [63, 64]
    let embedded = input_ids @ self.token_embed
    let x = embedded + self.pos_embed

    // Run through transformer blocks
    let h1 = self.block1.forward(x)
    let h2 = self.block2.forward(h1)
    let h3 = self.block3.forward(h2)
    let h4 = self.block4.forward(h3)

    // Project to vocabulary logits
    let logits = h4 @ self.output_proj
    return logits

  fn train_step(self, input_ids: Tensor[63, 128], target_ids: Tensor[63, 128]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let logits = self.forward_sequence(input_ids)
    let loss = cross_entropy(logits, target_ids)
    print(loss)
    update self by adam(grad(loss), lr=3e-4)
    return loss

// ─────────────────────────────────────────────
//  Dataset Slicing Helper
// ─────────────────────────────────────────────
fn extract_sequence(dataset: Tensor[D, 128], sample_idx: Int) -> Tensor[63, 128]:
  let seq = zeros(63, 128)
  let k = 0
  while k < 63:
    let idx = sample_idx * 63 + k
    let row = dataset[idx]
    let seq = update_row(seq, k, row)
    let k = k + 1
  return seq

// ─────────────────────────────────────────────
//  Text Generation (Autoregressive)
// ─────────────────────────────────────────────
fn generate_text(gpt: MicroGPT, prompt_seq: Tensor[63, 128], gen_len: Int) -> Int:
  // Generate tokens autoregressively
  // Start with the prompt sequence, run forward, get next token
  let current_seq = prompt_seq
  let t = 0
  while t < gen_len:
    // Forward pass through the model
    let logits = gpt.forward_sequence(current_seq)
    // Get the predicted next token (argmax of last position's logits)
    let next_token = argmax(logits)
    // Print the generated character
    let ch = char_from_int(next_token)
    print(ch)
    // Build proper one-hot vector for the predicted token
    let new_row = onehot(next_token, 128)
    // Shift the sequence left by 1 and append the new token
    let new_seq = zeros(63, 128)
    let k = 0
    while k < 62:
      let old_row = current_seq[k + 1]
      let new_seq = update_row(new_seq, k, old_row)
      let k = k + 1
    // Put new token's one-hot at position 62 (last)
    let new_seq = update_row(new_seq, 62, new_row)
    let current_seq = new_seq
    let t = t + 1
  return gen_len

// ─────────────────────────────────────────────
//  Main Training + Generation Loop
// ─────────────────────────────────────────────
fn main() -> Tensor[1, 1]:
  print("═══════════════════════════════════════════════")
  print("  MicroGPT — NEURON Language Model Training")
  print("═══════════════════════════════════════════════")

  // Initialize model
  print("Initializing MicroGPT (4 layers, 4 heads, d=64)...")
  let gpt = MicroGPT()

  // Load preprocessed training data (1000 samples of length 63 = 63000 rows)
  print("Loading training data...")
  let num_samples = 1000
  let seq_len = 63
  let total_rows = 63000

  let train_inputs = load_tensor("data/train_inputs.bin", total_rows, 128)
  let train_targets = load_tensor("data/train_targets.bin", total_rows, 128)
  print("Training data loaded.")

  // Training loop
  print("Starting training...")
  let step = 0
  let total_steps = 50

  while step < total_steps:
    // Get the current sample (cycling through dataset)
    let sample_idx = step
    if sample_idx >= num_samples:
      let sample_idx = step - num_samples

    // Extract input and target rows as one-hot encoded sequences of shape [63, 128]
    let input_seq = extract_sequence(train_inputs, sample_idx)
    let target_seq = extract_sequence(train_targets, sample_idx)

    // Train step
    let loss = gpt.train_step(input_seq, target_seq)

    let step = step + 1

  print("Training complete!")

  // Save trained weights
  print("Saving trained token embeddings...")
  save_tensor(gpt.token_embed, "data/trained_token_embed.bin")
  print("Weights saved.")

  // ─── Text Generation Phase ───
  print("")
  print("═══════════════════════════════════════════════")
  print("  Text Generation")
  print("═══════════════════════════════════════════════")

  // Use the first training sequence as a prompt
  let prompt = extract_sequence(train_inputs, 0)
  print("Generating 20 characters...")
  let gen_count = generate_text(gpt, prompt, 20)
  print("")
  print("Generation complete!")

  // Final evaluation
  let final_input = extract_sequence(train_inputs, 0)
  let final_target = extract_sequence(train_targets, 0)
  let final_logits = gpt.forward_sequence(final_input)
  let final_loss = cross_entropy(final_logits, final_target)
  print("Final loss:")
  print(final_loss)

  return final_loss
