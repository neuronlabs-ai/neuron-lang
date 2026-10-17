// ═══════════════════════════════════════════════════════════════
//  stdlib/math.nr — Standard Mathematical Utilities for NEURON
// ═══════════════════════════════════════════════════════════════

// ─── Mathematical Constants ──────────────────────────────────
fn pi() -> Float:
  return 3.14159265358979323846

fn e() -> Float:
  return 2.71828182845904523536

fn tau() -> Float:
  return 6.28318530717958647692

// ─── Powers, Exponents and Roots ─────────────────────────────
fn square(x: Float) -> Float:
  return x * x

fn cube(x: Float) -> Float:
  return x * x * x

fn pow_f(x: Float, p: Float) -> Float:
  return x ** p

fn sqrt_f(x: Float) -> Float:
  if x < 0.0:
    print("Warning: sqrt of negative float is undefined in real numbers")
    return 0.0
  return sqrt(x)

fn cbrt(x: Float) -> Float:
  if x < 0.0:
    let pos_x = -x
    return -(pos_x ** (1.0 / 3.0))
  return x ** (1.0 / 3.0)

fn hypot(a: Float, b: Float) -> Float:
  return sqrt(a * a + b * b)

// ─── Basic Operations & Clamping ─────────────────────────────
fn abs_f(x: Float) -> Float:
  if x < 0.0:
    return 0.0 - x
  return x

fn sign(x: Float) -> Float:
  if x > 0.0:
    return 1.0
  if x < 0.0:
    return -1.0
  return 0.0

fn clamp(x: Float, min_val: Float, max_val: Float) -> Float:
  if x < min_val:
    return min_val
  if x > max_val:
    return max_val
  return x

fn lerp(a: Float, b: Float, t: Float) -> Float:
  return a + t * (b - a)

// ─── Logarithms and Exponentials ─────────────────────────────
fn ln(x: Float) -> Float:
  return log(x)

fn log10(x: Float) -> Float:
  return log(x) / log(10.0)

fn log2(x: Float) -> Float:
  return log(x) / log(2.0)

fn exp_f(x: Float) -> Float:
  return exp(x)

// ─── Angular & Trigonometric Functions ──────────────────────
fn deg2rad(deg: Float) -> Float:
  return deg * (3.141592653589793 / 180.0)

fn rad2deg(rad: Float) -> Float:
  return rad * (180.0 / 3.141592653589793)

fn sin_f(x: Float) -> Float:
  return sin(x)

fn cos_f(x: Float) -> Float:
  return cos(x)

fn tan_f(x: Float) -> Float:
  return sin(x) / cos(x)


// ─── Integer & Number-Theoretic Utilities ───────────────────
fn gcd(a: Int, b: Int) -> Int:
  let x = a
  let y = b
  while y != 0:
    let temp = y
    let y = x % y
    let x = temp
  if x < 0:
    return -x
  return x

fn lcm(a: Int, b: Int) -> Int:
  if a == 0:
    return 0
  if b == 0:
    return 0
  let g = gcd(a, b)
  return (a / g) * b

fn factorial(n: Int) -> Int:
  if n <= 1:
    return 1
  let res = 1
  let i = 2
  while i <= n:
    let res = res * i
    let i = i + 1
  return res

fn is_even(n: Int) -> Bool:
  return (n % 2) == 0

fn is_odd(n: Int) -> Bool:
  return (n % 2) != 0
