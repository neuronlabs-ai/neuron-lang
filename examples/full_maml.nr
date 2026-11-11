// Full MAML in NEURON: differentiate query loss through support-set gradient steps.
import agi

fn main():
  print("Second-order scalar check")
  let initial_w: Tensor[1] = zeros(1) + 1.0
  let scalar_x: Tensor[1] = zeros(1) + 1.0
  let support_y: Tensor[1] = zeros(1) + 2.0
  let query_y: Tensor[1] = zeros(1)
  let inner_lr: Tensor[1] = zeros(1) + 0.1
  let support_loss = mse(scalar_x * initial_w, support_y)
  let support_grad = grad(support_loss, wrt=initial_w)
  let adapted_w = initial_w - (inner_lr * support_grad)
  let query_loss = mse(scalar_x * adapted_w, query_y)
  let meta_grad = grad(query_loss, wrt=initial_w)
  print("Inner gradient (expected -2):")
  print(support_grad)
  print("Adapted weight (expected 1.2):")
  print(adapted_w)
  print("Full MAML meta-gradient (expected 1.92):")
  print(meta_grad)

  print("Full neural MetaLearner update")
  let learner = MetaLearner(1, 1, 4)
  let task_support_x: Tensor[1, 1] = zeros(1, 1) + 1.0
  let task_support_y: Tensor[1, 1] = zeros(1, 1) + 3.0
  let task_query_x: Tensor[1, 1] = zeros(1, 1) + 2.0
  let task_query_y: Tensor[1, 1] = zeros(1, 1) + 6.0

  learner.inner_adapt(task_support_x, task_support_y)
  let loss_before = mse(learner.forward_adapted(task_query_x), task_query_y)
  print("Adapted query loss before outer update:")
  print(loss_before)

  let maml_loss = learner.maml_update(task_support_x, task_support_y, task_query_x, task_query_y)
  print("Query loss used for the MAML outer gradient:")
  print(maml_loss)

  learner.inner_adapt(task_support_x, task_support_y)
  let loss_after = mse(learner.forward_adapted(task_query_x), task_query_y)
  print("Adapted query loss after the MAML outer update:")
  print(loss_after)
