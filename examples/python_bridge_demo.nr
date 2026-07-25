// ═══════════════════════════════════════════════════════════════
//  python_bridge_demo.nr — Calling Python from NEURON
//  
//  Demonstrates NEURON's Python interop bridge.
//  NEURON type-checks your code, then calls Python libraries
//  for heavy computation. Best of both worlds.
// ═══════════════════════════════════════════════════════════════

@python
import numpy as np

fn main():
  // Call numpy directly from NEURON
  let data = np.zeros(3, 3)
  print(data)

  let identity = np.eye(4)
  print(identity)

  let random = np.random.randn(2, 3)
  print(random)
