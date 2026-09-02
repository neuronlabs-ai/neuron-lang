// math_preamble.nr — Standard Olympiad Preamble for NEURON

fn gcd(a: Int, b: Int) -> Int:
  let x = a
  let y = b
  while y != 0:
    let temp = y
    let y = x % y
    let x = temp
  return x

fn lcm(a: Int, b: Int) -> Int:
  if a == 0:
    return 0
  if b == 0:
    return 0
  return (a * b) / gcd(a, b)

fn mod_pow(base: Int, exp: Int, m: Int) -> Int:
  let res = 1
  let b = base % m
  let e = exp
  while e > 0:
    if e % 2 == 1:
      let res = (res * b) % m
    let b = (b * b) % m
    let e = e / 2
  return res

fn is_prime(n: Int) -> Bool:
  if n <= 1:
    return false
  if n <= 3:
    return true
  if n % 2 == 0:
    return false
  if n % 3 == 0:
    return false
  let i = 5
  while i * i <= n:
    if n % i == 0:
      return false
    if n % (i + 2) == 0:
      return false
    let i = i + 6
  return true

fn count_divisors(n: Int) -> Int:
  let count = 0
  let i = 1
  while i * i <= n:
    if n % i == 0:
      if i * i == n:
        let count = count + 1
      else:
        let count = count + 2
    let i = i + 1
  return count

fn sum_divisors(n: Int) -> Int:
  let total = 0
  let i = 1
  while i * i <= n:
    if n % i == 0:
      if i * i == n:
        let total = total + i
      else:
        let total = total + i + (n / i)
    let i = i + 1
  return total

fn nPr(n: Int, r: Int) -> Int:
  if r < 0:
    return 0
  if r > n:
    return 0
  let res = 1
  let i = 0
  while i < r:
    let res = res * (n - i)
    let i = i + 1
  return res

fn nCr(n: Int, r: Int) -> Int:
  if r < 0:
    return 0
  if r > n:
    return 0
  if r == 0:
    return 1
  if r == n:
    return 1
  let k = r
  if k > n - k:
    let k = n - k
  let num = 1
  let den = 1
  let i = 1
  while i <= k:
    let num = num * (n - i + 1)
    let den = den * i
    let i = i + 1
  return num / den

fn stars_and_bars(n_items: Int, k_bins: Int) -> Int:
  if k_bins <= 0:
    return 0
  return nCr(n_items + k_bins - 1, k_bins - 1)

fn derangements(n: Int) -> Int:
  if n == 0:
    return 1
  if n == 1:
    return 0
  if n == 2:
    return 1
  let d_prev2 = 1
  let d_prev1 = 0
  let i = 2
  let cur = 1
  while i <= n:
    let cur = (i - 1) * (d_prev1 + d_prev2)
    let d_prev2 = d_prev1
    let d_prev1 = cur
    let i = i + 1
  return cur

fn catalan(n: Int) -> Int:
  if n < 0:
    return 0
  return nCr(2 * n, n) / (n + 1)

fn count_factor_triples(n: Int) -> Int:
  if n <= 0:
    return 0
  let temp = n
  let total_ways = 1
  let d = 2
  while d * d <= temp:
    if temp % d == 0:
      let exp = 0
      while temp % d == 0:
        let exp = exp + 1
        let temp = temp / d
      let ways = (exp + 2) * (exp + 1) / 2
      let total_ways = total_ways * ways
    let d = d + 1
  if temp > 1:
    let total_ways = total_ways * 3
  return total_ways

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

fn heron_area(a: Float, b: Float, c: Float) -> Float:
  let s = (a + b + c) / 2.0
  let val = s * (s - a) * (s - b) * (s - c)
  return sqrt_newton(val)

fn main():
  let t = count_factor_triples(360)
  print(t)