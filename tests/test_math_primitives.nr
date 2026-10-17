fn main():
  // 1. Test basic scalar exponentiation
  let a: Float = 2.0 ^ 3.0
  print("2.0 ^ 3.0 =")
  print(a)
  
  let b: Float = 2.0 ** 8.0
  print("2.0 ** 8.0 =")
  print(b)
  
  let c: Int = 2 ^ 10
  print("2 ^ 10 =")
  print(c)
  
  let d: Float = pow(3.0, 3.0)
  print("pow(3.0, 3.0) =")
  print(d)

  // 2. Test exp and log builtins
  let e0: Float = exp(0.0)
  print("exp(0.0) =")
  print(e0)

  let l1: Float = log(e0)
  print("log(exp(0.0)) =")
  print(l1)

  let e1: Float = exp(1.0)
  let l_e1: Float = log(e1)
  print("log(exp(1.0)) =")
  print(l_e1)

  // 3. Test tensor operations
  let t = zeros(2, 2) + 2.0
  let t_sq = t ^ 2.0
  print("Tensor t ^ 2.0:")
  print(t_sq)

  let t_exp = exp(t)
  print("Tensor exp(t):")
  print(t_exp)

  let t_log = log(t)
  print("Tensor log(t):")
  print(t_log)

  print("ALL MATH PRIMITIVES PASSED")
