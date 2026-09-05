// NEURON Testing M_67 (composite) and M_521 (157 digits!)

fn is_small_prime(n: Int) -> Bool:
    if n < 2:
        return false
    if n == 2:
        return true
    if n % 2 == 0:
        return false
    let d = 3
    while d * d <= n:
        if n % d == 0:
            return false
        let d = d + 2
    return true

fn mersenne_number(p: Int) -> Int:
    let m = 1
    let i = 0
    while i < p:
        let m = m * 2
        let i = i + 1
    return m - 1

fn lucas_lehmer_test(p: Int) -> Bool:
    if p == 2:
        return true
    if is_small_prime(p) == false:
        return false
    let mp = mersenne_number(p)
    let s = 4
    let i = 0
    let steps = p - 2
    while i < steps:
        let s2 = s * s - 2
        let s = s2 % mp
        let i = i + 1
    return s == 0

fn main():
    print("==================================================")
    print("  ◈ NEURON Testing M_67 (Famous Mersenne Error)   ")
    print("==================================================")
    print("M_67 = 2^67 - 1:")
    let m67 = mersenne_number(67)
    print(m67)
    let is_m67_prime = lucas_lehmer_test(67)
    print("Is M_67 prime? (Expected: false)")
    print(is_m67_prime)

    print("==================================================")
    print("  ◈ NEURON Testing M_521 (First SWAC Prime, 1952)")
    print("  ◈ 157 decimal digits                           ")
    print("==================================================")
    let m521 = mersenne_number(521)
    print("M_521:")
    print(m521)
    print("Running 519 Lucas-Lehmer squarings on 157-digit numbers...")
    let is_m521_prime = lucas_lehmer_test(521)
    print("Is M_521 prime? (Expected: true)")
    print(is_m521_prime)
