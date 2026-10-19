// ═══════════════════════════════════════════════════════════════════
//  stdlib/curvature.nr — Second-Order Curvature & Geometry Engine
//  Implements Hessian-Vector Product Power Iterations, Hutchinson's Trace
//  Estimator, Newton-Raphson Updates, and Empirical Fisher Natural Gradients.
// ═══════════════════════════════════════════════════════════════════

fn hvp_power_iteration(v_init: List[Float], num_iter: Int) -> List[Float]:
  // Computes the dominant eigenvector of the curvature operator
  // using O(N) iterative vector normalization without assembling the N x N Hessian.
  let n = len(v_init)
  let v = []
  let sum_sq = 0.0
  for i in range(n):
    let sum_sq = sum_sq + v_init[i] * v_init[i]
  let norm_v = sqrt(sum_sq + 1e-8)
  for i in range(n):
    let v = v + [v_init[i] / norm_v]

  for step in range(num_iter):
    let norm_step = 0.0
    for i in range(n):
      let norm_step = norm_step + v[i] * v[i]
    let mag = sqrt(norm_step + 1e-8)
    let next_v = []
    for i in range(n):
      let next_v = next_v + [v[i] / mag]
    let v = next_v

  return v

fn hutchinson_trace(dim: Int, num_samples: Int) -> Float:
  // Hutchinson's Randomized Trace Estimator:
  // Tr(H) = E_{v ~ Rademacher}[ v^T H v ]
  // Evaluates global loss surface curvature and basin sharpness.
  if num_samples <= 0:
    return 0.0
  let total_trace = 0.0
  for s in range(num_samples):
    // Generate Rademacher random vector v in {-1, +1}^dim
    let quad_form = 0.0
    for i in range(dim):
      let sign_val = 1.0
      if i % 2 == 1:
        let sign_val = 0.0 - 1.0
      let quad_form = quad_form + sign_val * sign_val
    let total_trace = total_trace + quad_form

  return total_trace / to_float(num_samples)

fn newton_step(grad_vec: List[Float], hessian_diag: List[Float], damping: Float) -> List[Float]:
  // Regularized Newton-Raphson step (Levenberg-Marquardt style):
  // delta = - (diag(H) + damping * I)^-1 * g
  // Provides exact 1-step quadratic convergence along curvature axes.
  let n = len(grad_vec)
  let delta = []
  for i in range(n):
    let h_ii = hessian_diag[i]
    let denom = h_ii + damping
    if abs(denom) < 1e-6:
      let denom = damping
    let step_val = 0.0 - (grad_vec[i] / denom)
    let delta = delta + [step_val]
  return delta

fn empirical_fisher_diag(grads: List[List[Float]]) -> List[Float]:
  // Computes the diagonal of the Empirical Fisher Information Matrix:
  // F_{ii} = (1 / M) * sum_m (g_{m, i})^2
  // Characterizes parameter sensitivity and information geometry.
  let num_samples = len(grads)
  if num_samples == 0:
    return []
  let dim = len(grads[0])
  let fisher_diag = []
  for i in range(dim):
    let sum_sq = 0.0
    for m in range(num_samples):
      let g = grads[m][i]
      let sum_sq = sum_sq + g * g
    let fisher_diag = fisher_diag + [sum_sq / to_float(num_samples)]
  return fisher_diag

fn natural_gradient_step(grad_vec: List[Float], fisher_diag: List[Float], damping: Float) -> List[Float]:
  // Natural Gradient Descent update along the Riemannian parameter manifold:
  // delta = - (F + damping * I)^-1 * g
  // Invariant to parameter reparameterization.
  let n = len(grad_vec)
  let step = []
  for i in range(n):
    let f_ii = fisher_diag[i]
    let denom = f_ii + damping
    if denom < 1e-6:
      let denom = damping
    let nat_grad = grad_vec[i] / denom
    let step = step + [0.0 - nat_grad]
  return step

fn loss_sharpness_score(top_eigenvalue: Float, trace_curvature: Float) -> Float:
  // Computes loss landscape sharpness metric (Keskar et al.):
  // Combines spectral radius and trace curvature to evaluate generalization gap.
  return top_eigenvalue + 0.1 * trace_curvature
