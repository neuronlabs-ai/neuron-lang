// One-shot adaptation with a single support example, followed by a Reptile
// outer update that moves the shared model toward the adapted task model.
import agi

fn main():
  let learner = MetaLearner(1, 1, 4)
  let support_x: Tensor[1, 1] = zeros(1, 1) + 1.0
  let support_y: Tensor[1, 1] = zeros(1, 1) + 3.0
  let query_x: Tensor[1, 1] = zeros(1, 1) + 2.0
  let query_y: Tensor[1, 1] = zeros(1, 1) + 6.0

  let before = learner.forward(query_x)
  let before_loss = mse(before, query_y)
  print("Query prediction before seeing the support example:")
  print(before)
  print("Query loss before adaptation:")
  print(before_loss)

  learner.inner_adapt(support_x, support_y)
  let after = learner.forward_adapted(query_x)
  let after_loss = mse(after, query_y)
  print("Query prediction after one-example adaptation:")
  print(after)
  print("Query loss after adaptation:")
  print(after_loss)

  print("Shared-model query prediction before the Reptile outer update:")
  print(learner.forward(query_x))
  learner.meta_update(support_x, support_y)
  print("Shared-model query prediction after the Reptile outer update:")
  print(learner.forward(query_x))
