// Auto-transpiled from test_transpile.py by NEURON Python Transpiler
// Source: test_transpile.py

model TransformerBlock(d_model: Int, n_heads: Int):
  wq: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wk: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wv: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wo: Tensor[d_model, d_model] = glorot(d_model, d_model)
  scale: Tensor = 1.0 / sqrt(d_model)

  fn forward(self, x: Tensor) -> Tensor:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = q @ k.T * self.scale
    let attn = softmax(scores)
    let out = attn @ v
    return out @ self.wo


model MLP(input_dim: Int, hidden_dim: Int, output_dim: Int):
  w1: Tensor[input_dim, hidden_dim] = glorot(input_dim, hidden_dim)
  w2: Tensor[hidden_dim, output_dim] = glorot(hidden_dim, output_dim)
  b1: Tensor[1, hidden_dim] = zeros(1, hidden_dim)
  b2: Tensor[1, output_dim] = zeros(1, output_dim)

  fn forward(self, x: Tensor) -> Tensor:
    let h = relu(x @ self.w1 + self.b1)
    return h @ self.w2 + self.b2


fn train_step(model_: MLP, x: Tensor, y: Tensor, lr: Float) -> Tensor:
  let pred = model_.forward(x)
  let loss = mse(pred, y)
  return loss

fn main():
  let model_ = MLP(784, 256, 10)
  let x = glorot(32, 784)
  let y = zeros(32, 10)
  let loss = train_step(model_, x, y, 0.001)
  print(loss)
