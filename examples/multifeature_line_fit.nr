// Multi-feature line fit
// Learns y = 2*x1 + 3*x2 + 1 from a four-row batch.

model PlaneLearner:
  w: Tensor[2, 1] = zeros(2, 1) + 0.5
  b: Tensor[1, 1] = zeros(1, 1)

  fn predict(self, x: Tensor[B, 2]) -> Tensor[B, 1]:
    return x @ self.w + self.b

  fn train_step(self, x: Tensor[4, 2], y: Tensor[4, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let loss = mse(self.predict(x), y)
    update self.w by adam(grad(loss), lr=0.05)
    update self.b by adam(grad(loss), lr=0.05)
    return loss

fn main():
  // Training rows: (0,0)->1, (1,0)->3, (0,1)->4, (2,1)->8.
  let x = zeros(4, 2)
  let x = update_row(x, 0, concat([zeros(1, 1) + 0.0, zeros(1, 1) + 0.0]))
  let x = update_row(x, 1, concat([zeros(1, 1) + 1.0, zeros(1, 1) + 0.0]))
  let x = update_row(x, 2, concat([zeros(1, 1) + 0.0, zeros(1, 1) + 1.0]))
  let x = update_row(x, 3, concat([zeros(1, 1) + 2.0, zeros(1, 1) + 1.0]))

  let y = zeros(4, 1)
  let y = update_row(y, 0, zeros(1, 1) + 1.0)
  let y = update_row(y, 1, zeros(1, 1) + 3.0)
  let y = update_row(y, 2, zeros(1, 1) + 4.0)
  let y = update_row(y, 3, zeros(1, 1) + 8.0)

  let learner = PlaneLearner()
  print("Initial MSE:")
  print(mse(learner.predict(x), y))

  let epoch = 0
  while epoch < 1000:
    let loss = learner.train_step(x, y)
    let epoch = epoch + 1

  print("Final MSE:")
  print(mse(learner.predict(x), y))
  print("Learned weights (expected [2, 3]) and bias (expected 1):")
  print(learner.w)
  print(learner.b)

  let probe_x1 = concat([zeros(1, 1) + 1.0, zeros(1, 1) + 0.0])
  let probe_x2 = concat([zeros(1, 1) + 0.0, zeros(1, 1) + 1.0])
  print("Probe x1=1, x2=0 (expected 3):")
  print(learner.predict(probe_x1))
  print("Probe x1=0, x2=1 (expected 4):")
  print(learner.predict(probe_x2))

  let x_test = concat([zeros(1, 1) + 3.0, zeros(1, 1) + 2.0])
  print("Prediction for (x1=3, x2=2), expected 13:")
  print(learner.predict(x_test))
