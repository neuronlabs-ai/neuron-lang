// probability.nr — Combinatorics and Counting Primitives for NEURON

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

// Stars and Bars (weak partitions): non-negative solutions to x_1 + ... + x_k = n
// Formula: C(n + k - 1, k - 1)
fn stars_and_bars(n_items: Int, k_bins: Int) -> Int:
  if k_bins <= 0:
    return 0
  return nCr(n_items + k_bins - 1, k_bins - 1)

// Stars and Bars (strict partitions): positive solutions x_i >= 1
// Formula: C(n - 1, k - 1)
fn stars_and_bars_positive(n_items: Int, k_bins: Int) -> Int:
  if n_items < k_bins:
    return 0
  if k_bins <= 0:
    return 0
  return nCr(n_items - 1, k_bins - 1)

// Subfactorial / Derangements: !n = (n-1)*(!(n-1) + !(n-2))
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

// Catalan Number: C_n = (1 / (n+1)) * C(2n, n)
fn catalan(n: Int) -> Int:
  if n < 0:
    return 0
  let c = nCr(2 * n, n)
  return c / (n + 1)

// Count ordered triples (a, b, c) of positive integers such that a * b * c = n
// Uses prime factorization and stars-and-bars on exponents:
// For n = p_1^e_1 * ... * p_m^e_m:
// Each e_i can be split across 3 bins in C(e_i + 3 - 1, 3 - 1) ways.
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
    let ways = 3
    let total_ways = total_ways * ways
  return total_ways

// General count of ordered k-tuples (x_1, ..., x_k) such that x_1 * ... * x_k = n
fn count_factor_ktubles(n: Int, k: Int) -> Int:
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
      let ways = stars_and_bars(exp, k)
      let total_ways = total_ways * ways
    let d = d + 1
  if temp > 1:
    let ways = stars_and_bars(1, k)
    let total_ways = total_ways * ways
  return total_ways

fn main():
  print("[1] Testing Combinatorics Primitives:")
  let p = nPr(5, 2)
  print(p) // Expected: 20

  let c = nCr(10, 4)
  print(c) // Expected: 210

  print("[2] Testing Stars and Bars:")
  let sb = stars_and_bars(7, 3)
  print(sb) // C(7+3-1, 2) = C(9, 2) = 36

  print("[3] Testing Derangements (!5):")
  let d5 = derangements(5)
  print(d5) // Expected: 44

  print("[4] Testing Catalan Number C_4:")
  let cat4 = catalan(4)
  print(cat4) // Expected: 14

  print("[5] Testing Count Factor Triples for 360:")
  let triples = count_factor_triples(360)
  print(triples) // Expected: 180!

  print("[6] Testing General k-tuples for 360 into 3 factors:")
  let kt = count_factor_ktubles(360, 3)
  print(kt) // Expected: 180!