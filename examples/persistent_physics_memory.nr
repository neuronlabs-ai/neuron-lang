// persistent_physics_memory.nr
// Fit a physical parameter, persist it as a tensor-valued semantic fact,
// reload it in a fresh memory instance, and use the recalled value again.

fn hamiltonian(k: Tensor[1], q: Tensor[1], p: Tensor[1]) -> Tensor[1]:
  let half: Tensor[1] = zeros(1) + 0.5
  return (half * k * q * q) + (half * p * p)

fn main():
  print("NEURON: persistent memory meets differentiable physics")

  // Fit k from F = dH/dq = k*q at q=2, where observed F=6.
  let k0: Tensor[1] = zeros(1) + 0.5
  let q_obs: Tensor[1] = zeros(1) + 2.0
  let p_obs: Tensor[1] = zeros(1) + 0.0
  let target_force: Tensor[1] = zeros(1) + 6.0
  let h_obs = hamiltonian(k0, q_obs, p_obs)
  let predicted_force = grad(h_obs, wrt=q_obs)
  let residual = predicted_force - target_force
  let loss = 0.5 * residual * residual
  let g = grad(loss, wrt=k0)
  let h = hessian(loss, wrt=k0)
  let fitted_k = k0 - (g / h)

  print("Fitted spring constant:")
  print(fitted_k)

  // Persist the fitted tensor and reopen the same store as a fresh instance.
  let path = "C:/Users/ADMIN/neuron-lang/target/codex_physics_memory.nmem"
  let memory = SemanticMemory(64, 1000, path)
  memory.clear()
  memory.store_fact("oscillator_1", "spring_constant", fitted_k)
  memory.sync()

  let reloaded = SemanticMemory(64, 1000, path)
  let remembered_k = reloaded.query_value("oscillator_1", "spring_constant")
  print("Spring constant recalled from disk:")
  print(remembered_k)

  // Use the value restored from disk in a new force calculation.
  let h_recalled = hamiltonian(remembered_k, q_obs, p_obs)
  let recalled_force = grad(h_recalled, wrt=q_obs)
  print("Force from the reloaded model (expected 6):")
  print(recalled_force)
  print("Persisted semantic facts:")
  print(reloaded.len())
