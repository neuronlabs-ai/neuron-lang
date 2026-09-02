// trig.nr — Trigonometric functions via Taylor series for NEURON
// All angles in radians. Precision: ~12 significant digits.

fn abs_f(x: Float) -> Float:
  if x < 0.0:
    return 0.0 - x
  return x

fn fmod(x: Float, y: Float) -> Float:
  let n = x / y
  // truncate toward zero
  let ni = 0.0
  if n >= 0.0:
    let whole = 1.0
    let remain = n
    while remain >= 1.0:
      let remain = remain - 1.0
      let ni = ni + 1.0
    return x - ni * y
  else:
    let pos = 0.0 - n
    let whole = 1.0
    let remain = pos
    while remain >= 1.0:
      let remain = remain - 1.0
      let ni = ni - 1.0
    return x - ni * y

fn normalize_angle(x: Float) -> Float:
  let pi = 3.14159265358979323846
  let two_pi = 6.28318530717958647692
  let r = fmod(x, two_pi)
  if r < 0.0:
    let r = r + two_pi
  return r

fn sin_taylor(x: Float) -> Float:
  let pi = 3.14159265358979323846
  let two_pi = 6.28318530717958647692
  let a = fmod(x, two_pi)
  let term = a
  let sum = a
  let i = 1
  while i <= 12:
    let n = 2 * i
    let term = (0.0 - term) * a * a / (n * (n + 1))
    let sum = sum + term
    let i = i + 1
  return sum

fn cos_taylor(x: Float) -> Float:
  let pi = 3.14159265358979323846
  let two_pi = 6.28318530717958647692
  let a = fmod(x, two_pi)
  let term = 1.0
  let sum = 1.0
  let i = 1
  while i <= 12:
    let n = 2 * i
    let term = (0.0 - term) * a * a / ((n - 1) * n)
    let sum = sum + term
    let i = i + 1
  return sum

fn tan_taylor(x: Float) -> Float:
  let s = sin_taylor(x)
  let c = cos_taylor(x)
  return s / c

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

fn main():
  let pi = 3.14159265358979323846
  // sin(pi/6) = 0.5
  let s1 = sin_taylor(pi / 6.0)
  print(s1)
  // cos(pi/3) = 0.5
  let c1 = cos_taylor(pi / 3.0)
  print(c1)
  // sin(pi/2) = 1.0
  let s2 = sin_taylor(pi / 2.0)
  print(s2)
  // cos(0) = 1.0
  let c2 = cos_taylor(0.0)
  print(c2)
  // sqrt(2)
  let sq = sqrt_newton(2.0)
  print(sq)
  // sqrt(3)
  let sq3 = sqrt_newton(3.0)
  print(sq3)