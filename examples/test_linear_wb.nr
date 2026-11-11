// Test LinearModel with w and b and x @ self.w + self.b
model LinearModel:
  w: Tensor[1, 1] = zeros(1, 1) + 2.0
  b: Tensor[1, 1] = zeros(1, 1) + 0.0

  fn predict(self, x: Tensor[1, 1]) -> Tensor[1, 1]:
    return x @ self.w + self.b

  fn train_step(self, x: Tensor[1, 1], target: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let loss = mse(self.predict(x), target)
    update self.w by sgd(grad(loss), lr=0.1)
    update self.b by sgd(grad(loss), lr=0.1)
    return loss

fn main():
  let net = LinearModel()
  let x: Tensor[1, 1] = zeros(1, 1) + 2.0
  let target: Tensor[1, 1] = zeros(1, 1) + 7.0 // y = 3x + 1 -> for x=2, y=7

  print("Initial w:")
  print(net.w)
  print("Initial b:")
  print(net.b)

  let l1 = net.train_step(x, target)
  print("Step 1 loss:")
  print(l1)
  let l2 = net.train_step(x, target)
  print("Step 2 loss:")
  print(l2)
  let l3 = net.train_step(x, target)
  print("Step 3 loss:")
  print(l3)

  print("Trained w:")
  print(net.w)
  print("Trained b:")
  print(net.b)
