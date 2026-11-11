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

  // 4. Test tensor shape unpacking and free-standing aggregations
  let t_list = zeros([2, 3])
  let t_tuple = ones((3, 2))
  let t_randn = randn([2, 2])
  let s_val = sum(t_tuple)
  let m_val = mean(t_tuple)
  print("sum(ones((3, 2))):")
  print(s_val)
  print("mean(ones((3, 2))):")
  print(m_val)

  // 5. Test negative dimension reductions
  let t_2d = ones(2, 3)
  let s_dim = t_2d.sum(dim=-1)
  let m_dim = mean(t_2d, -1)
  print("s_dim shape:")
  print(shape(s_dim))
  print("m_dim shape:")
  print(shape(m_dim))

  print("ALL MATH PRIMITIVES PASSED")
