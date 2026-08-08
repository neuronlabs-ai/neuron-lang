// transformer.nr — A simple Transformer model in NEURON
// Demonstrates: model/layer declarations, Tensor shapes, Learnable, @compile

@compile(target="auto")
layer MultiHeadAttention(d_model: Int, n_heads: Int):
  head_dim: Int = 64
  wq: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wk: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wv: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wo: Tensor[d_model, d_model] = glorot(d_model, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = q @ k
    let attn = softmax(scores)
    let out = attn @ v
    return out @ self.wo

layer FeedForward(d_model: Int):
  w1: Tensor[d_model, 2048] = glorot(d_model, 2048)
  w2: Tensor[2048, d_model] = glorot(2048, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    let h = gelu(x @ self.w1)
    return h @ self.w2

@compile(target="auto")
model Transformer(d_model: Int, n_heads: Int, n_layers: Int):
  proj: Tensor[d_model, d_model] = glorot(d_model, d_model)

  fn forward(self, x: Tensor[1, d_model]) -> Tensor[1, d_model]:
    let h = relu(x @ self.proj)
    return softmax(h)

fn main():
  let tf_model = Transformer(512, 8, 6)
  let tokens = ones(1, 512)
  let logits = tf_model.forward(tokens)
  print(logits)
