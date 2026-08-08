// stress_test_python.nr — Thorough test of the Python bridge
// Tests multiple libraries, data types, and edge cases

@python
import numpy as np

@python
import math

fn main():
  // Test 1: numpy zeros
  let a = np.zeros(2, 3)
  print(a)

  // Test 2: numpy ones
  let b = np.ones(3, 3)
  print(b)

  // Test 3: numpy eye (identity)
  let c = np.eye(4)
  print(c)

  // Test 4: numpy random
  let d = np.random.randn(2, 2)
  print(d)

  // Test 5: numpy linspace
  let e = np.linspace(0, 10, 5)
  print(e)

  // Test 6: math.sqrt (scalar)
  let f = math.sqrt(144)
  print(f)

  // Test 7: math.pi (constant access - this will test edge cases)
  let g = math.factorial(10)
  print(g)

  // Test 8: numpy arange
  let h = np.arange(0, 10, 2)
  print(h)
