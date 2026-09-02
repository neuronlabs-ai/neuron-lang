// algebra.nr — Polynomial and equation solving primitives

fn sqrt_newton(x: Float) -> Float:
  if x <= 0.0:
    return 0.0
  let guess = x / 2.0
  if guess < 1.0:
    let guess = 1.0
  let i = 0
  while i < 40:
    let guess = (guess + x / guess) / 2.0
    let i = i + 1
  return guess

fn abs_f(x: Float) -> Float:
  if x < 0.0:
    return 0.0 - x
  return x

// Quadratic formula: ax^2 + bx + c = 0
// Returns the two roots as r1 (larger) printed first, r2 second
fn quadratic_roots(a: Float, b: Float, c: Float) -> Float:
  let disc = b * b - 4.0 * a * c
  if disc < 0.0:
    return 0.0 - 1.0
  let sq = sqrt_newton(disc)
  let r1 = (0.0 - b + sq) / (2.0 * a)
  let r2 = (0.0 - b - sq) / (2.0 * a)
  print(r1)
  print(r2)
  return r1

// Evaluate polynomial a0 + a1*x + a2*x^2 + a3*x^3 at point x
// (Horner's method for degree-3)
fn poly3_eval(a3: Float, a2: Float, a1: Float, a0: Float, x: Float) -> Float:
  return ((a3 * x + a2) * x + a1) * x + a0

// Vieta's for cubic x^3 + bx^2 + cx + d = 0 with roots r1,r2,r3:
//   r1+r2+r3 = -b
//   r1*r2 + r1*r3 + r2*r3 = c
//   r1*r2*r3 = -d
// Evaluate (r1-k)(r2-k)(r3-k) = P(k) * (-1) when leading coeff is 1
// Actually: (r1-k)(r2-k)(r3-k) = -(k^3 + b*k^2 + c*k + d) ... wait
// P(x) = (x-r1)(x-r2)(x-r3), so P(k) = (k-r1)(k-r2)(k-r3)
// Therefore (r1-k)(r2-k)(r3-k) = (-1)^3 * P(k) = -P(k)

fn vieta_product_shifted(b: Float, c: Float, d: Float, k: Float) -> Float:
  let pk = k * k * k + b * k * k + c * k + d
  return 0.0 - pk

// Natural logarithm via Newton's method: find y such that e^y = x
fn ln_approx(x: Float) -> Float:
  if x <= 0.0:
    return 0.0
  let y = 0.0
  // rough initial guess
  let temp = x
  while temp > 2.718281828:
    let y = y + 1.0
    let temp = temp / 2.718281828
  // Newton iterations on f(y) = e^y - x, f'(y) = e^y
  let i = 0
  while i < 50:
    // compute e^y via Taylor
    let ey = 1.0
    let term = 1.0
    let j = 1
    while j <= 20:
      let term = term * y / j
      let ey = ey + term
      let j = j + 1
    let y = y - (ey - x) / ey
    let i = i + 1
  return y

// log base b of x = ln(x) / ln(b)
fn log_base(x: Float, b: Float) -> Float:
  return ln_approx(x) / ln_approx(b)

fn main():
  // Quadratic: x^2 - 5x + 6 = 0 -> roots 3, 2
  print(0.0 - 1.0)  // separator
  let r = quadratic_roots(1.0, 0.0 - 5.0, 6.0)

  // Vieta product shifted: P(x) = x^3 - 15x^2 + 66x - 80
  // (r1-1)(r2-1)(r3-1) = -P(1) = -(1 - 15 + 66 - 80) = -(-28) = 28
  let v = vieta_product_shifted(0.0 - 15.0, 66.0, 0.0 - 80.0, 1.0)
  print(v)  // Expected: 28.0

  // ln(e) should be ~1.0
  let lne = ln_approx(2.718281828)
  print(lne)

  // log base 10 of 1000 = 3
  let lg = log_base(1000.0, 10.0)
  print(lg)