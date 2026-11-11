// Tiny Line Learner
// Learns the gain in y = 3x with gradient descent, then predicts at x = 4.

model GainLearner:
  gain: Tensor[1, 1] = zeros(1, 1) + 5.0

  fn predict(self, x: Tensor[1, 1]) -> Tensor[1, 1]:
    return x @ self.gain

  fn train_step(self, x: Tensor[1, 1], target: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let loss = mse(self.predict(x), target)
    update self.gain by sgd(grad(loss), lr=0.1)
    return loss

fn main():
  let learner = GainLearner()
  let x_train: Tensor[1, 1] = zeros(1, 1) + 2.0
  let target: Tensor[1, 1] = zeros(1, 1) + 6.0

  print("Training: x = 2 should map to y = 6")
  print("Loss after each SGD step:")
  let loss1 = learner.train_step(x_train, target)
  print(loss1)
  let loss2 = learner.train_step(x_train, target)
  print(loss2)
  let loss3 = learner.train_step(x_train, target)
  print(loss3)
  let loss4 = learner.train_step(x_train, target)
  print(loss4)
  let loss5 = learner.train_step(x_train, target)
  print(loss5)

  print("Learned gain:")
  print(learner.gain)

  let x_test: Tensor[1, 1] = zeros(1, 1) + 4.0
  print("Prediction for x = 4 (target 12):")
  print(learner.predict(x_test))
