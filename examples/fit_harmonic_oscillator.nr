// fit_harmonic_oscillator.nr
// Infer a spring constant from a force observation with autodiff + Newton,
// then use the inferred Hamiltonian in a symplectic Euler trajectory.

fn hamiltonian(k: Tensor[1], q: Tensor[1], p: Tensor[1]) -> Tensor[1]:
  let half: Tensor[1] = zeros(1) + 0.5
  let potential = half * k * q * q
  let kinetic = half * p * p
  return potential + kinetic

fn main():
  print("NEURON: differentiable physics parameter fit")

  // At q=2, the measured force magnitude is 6, so the expected k is 3.
  let k0: Tensor[1] = zeros(1) + 0.5
  let q_obs: Tensor[1] = zeros(1) + 2.0
  let p_obs: Tensor[1] = zeros(1) + 0.0
  let target_force: Tensor[1] = zeros(1) + 6.0
  let h_obs = hamiltonian(k0, q_obs, p_obs)
  let force_obs = grad(h_obs, wrt=q_obs)
  let force_error = force_obs - target_force
  let fit_loss = 0.5 * force_error * force_error
  let fit_grad = grad(fit_loss, wrt=k0)
  let fit_hessian = hessian(fit_loss, wrt=k0)
  let k_fit = k0 - (fit_grad / fit_hessian)

  print("Force predicted by initial model (expected 1):")
  print(force_obs)
  print("Newton fit for k (expected 3):")
  print(k_fit)

  // Integrate the fitted oscillator with kick-then-drift symplectic Euler.
  let dt: Tensor[1] = zeros(1) + 0.05
  let q0: Tensor[1] = zeros(1) + 1.0
  let p0: Tensor[1] = zeros(1) + 0.0
  let energy0 = hamiltonian(k_fit, q0, p0)
  let q = q0
  let p = p0
  let step = 0
  while step < 20:
    let h_for_force = hamiltonian(k_fit, q, p)
    let dH_dq = grad(h_for_force, wrt=q)
    let p = p - (dH_dq * dt)
    let h_for_velocity = hamiltonian(k_fit, q, p)
    let dH_dp = grad(h_for_velocity, wrt=p)
    let q = q + (dH_dp * dt)
    let step = step + 1

  let energy_final = hamiltonian(k_fit, q, p)
  print("Final position after 20 symplectic steps:")
  print(q)
  print("Initial / final energy (should stay close):")
  print(energy0)
  print(energy_final)

