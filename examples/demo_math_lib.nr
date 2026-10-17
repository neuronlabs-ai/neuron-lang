// demo_math_lib.nr
// Demonstrates using the new NEURON standard math library

from math import square, cube, sqrt_f, cbrt, gcd, factorial, pi

fn main():
  let n = 64.0
  print("Number:")
  print(n)
  print("Square:")
  print(square(n))
  print("Cube:")
  print(cube(n))
  print("Square Root:")
  print(sqrt_f(n))
  print("Cube Root:")
  print(cbrt(n))

  print("-------------------")
  print("Math utilities:")
  print("Pi:")
  print(pi())
  print("GCD(48, 18):")
  print(gcd(48, 18))
  print("Factorial(5):")
  print(factorial(5))
