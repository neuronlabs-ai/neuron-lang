// ─────────────────────────────────────────────────────────────
//  NEURON Native In-Memory Cranelift JIT Speed Test
// ─────────────────────────────────────────────────────────────

fn main() -> Int:
  let a: Int = 1000
  let b: Int = 42
  let product: Int = a * b
  let offset: Int = 58
  let result: Int = product + offset
  return result
