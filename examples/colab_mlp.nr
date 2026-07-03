// colab_mlp.nr — A Multi-Layer Perceptron trained on Google Colab
// Fits a 2-layer MLP to map input [x1, x2] to target y on the GPU automatically.

model MLP:
  w1: Tensor[2, 8] = glorot(2, 8)
  b1: Tensor[1, 8] = zeros(1, 8)
  w2: Tensor[8, 1] = glorot(8, 1)
  b2: Tensor[1, 1] = zeros(1, 1)

  fn forward(self, x: Tensor[1, 2]) -> Tensor[1, 1]:
    let h1 = relu(x @ self.w1 + self.b1)
    let pred = h1 @ self.w2 + self.b2
    return pred

  fn train_step(self, x: Tensor[1, 2], y: Tensor[1, 1], lr: Float) -> Tensor[1, 1] [Effect[Mut[self], IO]]:
    let pred = self.forward(x)
    let loss = mse(pred, y)
    print(loss)
    
    // Compute gradients and update weights using SGD
    update self.w1 by sgd(grad(loss), lr=lr)
    update self.b1 by sgd(grad(loss), lr=lr)
    update self.w2 by sgd(grad(loss), lr=lr)
    update self.b2 by sgd(grad(loss), lr=lr)
    
    return loss

fn main() -> Tensor[1, 1]:
  let mlp = MLP()
  
  // Generate a random training pair
  let x = glorot(1, 2)
  let target_w = zeros(2, 1) + 3.0 // Target weights: 3.0
  let y = x @ target_w
  
  print("Training MLP to approximate target function on T4 GPU:")
  let l1 = mlp.train_step(x, y, 0.1)
  let l2 = mlp.train_step(x, y, 0.1)
  let l3 = mlp.train_step(x, y, 0.1)
  let l4 = mlp.train_step(x, y, 0.1)
  let l5 = mlp.train_step(x, y, 0.1)
  let l6 = mlp.train_step(x, y, 0.1)
  let l7 = mlp.train_step(x, y, 0.1)
  let l8 = mlp.train_step(x, y, 0.1)
  let l9 = mlp.train_step(x, y, 0.1)
  let l10 = mlp.train_step(x, y, 0.1)
  
  print("Final predicted output:")
  let pred = mlp.forward(x)
  print(pred)
  print("Expected target output:")
  print(y)
  
  return pred
