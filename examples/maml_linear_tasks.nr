// One-shot MAML regression over a family of y = a*x tasks.
// Each meta-training episode has one support pair and one query pair.
import agi

fn main():
  let learner = MetaLearner(1, 1, 4)
  learner.set_hyperparameters(0.1, 0.01, 1)
  let support_x: Tensor[1, 1] = zeros(1, 1) + 1.0
  let query_x: Tensor[1, 1] = zeros(1, 1) + 2.0

  let task_a_support: Tensor[1, 1] = zeros(1, 1) + 0.5
  let task_a_query: Tensor[1, 1] = zeros(1, 1) + 1.0
  let task_b_support: Tensor[1, 1] = zeros(1, 1) + 1.0
  let task_b_query: Tensor[1, 1] = zeros(1, 1) + 2.0
  let task_c_support: Tensor[1, 1] = zeros(1, 1) + 2.0
  let task_c_query: Tensor[1, 1] = zeros(1, 1) + 4.0
  let task_d_support: Tensor[1, 1] = zeros(1, 1) + 3.0
  let task_d_query: Tensor[1, 1] = zeros(1, 1) + 6.0

  let heldout_support: Tensor[1, 1] = zeros(1, 1) + 1.5
  let heldout_query: Tensor[1, 1] = zeros(1, 1) + 3.0
  learner.inner_adapt(support_x, heldout_support)
  let loss_before = mse(learner.forward_adapted(query_x), heldout_query)

  print("Held-out slope: 1.5; one support pair: (1, 1.5)")
  print("Held-out query loss before meta-training:")
  print(loss_before)

  let epoch = 0
  while epoch < 3:
    learner.maml_update(support_x, task_a_support, query_x, task_a_query)
    learner.maml_update(support_x, task_b_support, query_x, task_b_query)
    learner.maml_update(support_x, task_c_support, query_x, task_c_query)
    learner.maml_update(support_x, task_d_support, query_x, task_d_query)
    let epoch = epoch + 1

  learner.inner_adapt(support_x, heldout_support)
  let heldout_prediction = learner.forward_adapted(query_x)
  let loss_after = mse(heldout_prediction, heldout_query)
  print("Held-out prediction after one-shot adaptation:")
  print(heldout_prediction)
  print("Held-out query loss after meta-training:")
  print(loss_after)
