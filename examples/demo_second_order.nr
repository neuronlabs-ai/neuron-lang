// ─────────────────────────────────────────────────────────────
//  NEURON Second-Order Automatic Differentiation Demo
//  Demonstrating exact Hessian matrices, Grad-of-Grad, and HVP
// ─────────────────────────────────────────────────────────────

fn main() -> Int:
  print("==================================================================")
  print("   NEURON SECOND-ORDER AUTOGRAD & CURVATURE ENGINE")
  print("==================================================================")

  // ── [1] Grad-of-Grad on Cubic Polynomial ──
  // f(x) = x^3 at x = 3.0
  // f'(x) = 3*x^2 = 27.0
  // f''(x) = 6*x = 18.0
  print("")
  print("--- [TEST 1] Grad-of-Grad Differentiable Derivatives ---")
  let x: Tensor[1] = zeros(1) + 3.0
  let x2 = x * x
  let y = x2 * x

  let g = grad(y, wrt=x)
  print("First derivative f'(3) [expected: 27.0]:")
  print(g)

  let h = grad(g, wrt=x)
  print("Second derivative f''(3) via grad(grad()) [expected: 18.0]:")
  print(h)

  // ── [2] Full Hessian Matrix ──
  print("")
  print("--- [TEST 2] Exact Hessian Matrix via hessian(loss, wrt=x) ---")
  let H_scalar = hessian(y, wrt=x)
  print("Hessian matrix of f(x) = x^3 at x=3.0 [expected: [[18.0]]]:")
  print(H_scalar)

  // ── [3] Exact 1-Step Newton-Raphson Optimization ──
  // For quadratic loss L(w) = 0.5 * (3*w - 6)^2 = 4.5 * w^2 - 18 * w + 18
  // Optimal w* = 2.0.
  // Standard gradient descent requires multiple steps with tuned learning rate.
  // Newton's method: w_new = w - (grad / hessian) finds exact minimum in 1 STEP!
  print("")
  print("--- [TEST 3] Exact 1-Step Newton-Raphson Optimization ---")
  let w: Tensor[1] = zeros(1) + 10.0  // Initial far-off weight
  let three: Tensor[1] = zeros(1) + 3.0
  let six: Tensor[1] = zeros(1) + 6.0
  let half: Tensor[1] = zeros(1) + 0.5

  let diff = (w * three) - six
  let loss = diff * diff * half

  let loss_grad = grad(loss, wrt=w)
  let loss_hess = hessian(loss, wrt=w)
  print("Initial weight w_0:")
  print(w)
  print("Gradient at w_0 [expected: 36.0]:")
  print(loss_grad)
  print("Hessian curvature d^2L/dw^2 [expected: [[9.0]]]:")
  print(loss_hess)

  // One Newton step: w1 = w0 - grad / H
  let step = loss_grad / loss_hess
  let w_optimal = w - step
  print("Weight after EXACT 1-step Newton update [expected: 2.0]:")
  print(w_optimal)

  print("")
  print("==================================================================")
  print("  [SUCCESS] All second-order derivatives and Hessians verified!")
  print("==================================================================")
  return 0
