// geometry.nr — Coordinate geometry primitives for Olympiad competition
// Provides distance, area, circumradius, inradius, and line intersection tools.

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

// Distance between two points (x1,y1) and (x2,y2)
fn dist(x1: Float, y1: Float, x2: Float, y2: Float) -> Float:
  let dx = x2 - x1
  let dy = y2 - y1
  return sqrt_newton(dx * dx + dy * dy)

// Area of triangle given three side lengths (Heron's formula)
fn heron_area(a: Float, b: Float, c: Float) -> Float:
  let s = (a + b + c) / 2.0
  let val = s * (s - a) * (s - b) * (s - c)
  return sqrt_newton(val)

// Area of triangle given three vertices (Shoelace formula)
fn shoelace_area(x1: Float, y1: Float, x2: Float, y2: Float, x3: Float, y3: Float) -> Float:
  let area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
  return abs_f(area) / 2.0

// Circumradius of triangle with sides a, b, c
fn circumradius(a: Float, b: Float, c: Float) -> Float:
  let area = heron_area(a, b, c)
  if area <= 0.0:
    return 0.0
  return (a * b * c) / (4.0 * area)

// Inradius of triangle with sides a, b, c
fn inradius(a: Float, b: Float, c: Float) -> Float:
  let s = (a + b + c) / 2.0
  let area = heron_area(a, b, c)
  if s <= 0.0:
    return 0.0
  return area / s

// Law of cosines: find angle C opposite side c, given sides a, b, c
// Returns cos(C)
fn law_of_cosines_cos(a: Float, b: Float, c: Float) -> Float:
  return (a * a + b * b - c * c) / (2.0 * a * b)

fn main():
  // Test: 3-4-5 right triangle
  let a = 3.0
  let b = 4.0
  let c = 5.0
  let area = heron_area(a, b, c)
  print(area)  // Expected: 6.0

  // Circumradius of 3-4-5 triangle = 2.5
  let R = circumradius(a, b, c)
  print(R)  // Expected: 2.5

  // Inradius of 3-4-5 triangle = 1.0
  let r = inradius(a, b, c)
  print(r)  // Expected: 1.0

  // Shoelace: triangle (0,0), (3,0), (0,4) -> area = 6
  let sa = shoelace_area(0.0, 0.0, 3.0, 0.0, 0.0, 4.0)
  print(sa)  // Expected: 6.0

  // Distance (0,0) to (3,4) = 5
  let d = dist(0.0, 0.0, 3.0, 4.0)
  print(d)  // Expected: 5.0

  // cos(C) for 3-4-5 where C is opposite 5: should be 0
  let cosC = law_of_cosines_cos(3.0, 4.0, 5.0)
  print(cosC)  // Expected: 0.0 (right angle)