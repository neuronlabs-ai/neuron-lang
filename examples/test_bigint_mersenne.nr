// NEURON Native Arbitrary-Precision Mersenne Prime Hunter
// Testing primes beyond the 64-bit limit: M89, M107, M127!

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

// Computes 2^p - 1 using native arbitrary-precision integers
fn mersenne_number(p: Int) -> Int:
    let m = 1
    let i = 0
    while i < p:
        let m = m * 2
        let i = i + 1
    return m - 1

// Lucas-Lehmer test with native arbitrary-precision arithmetic
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
    print("  ◈ NEURON BigInt Mersenne Hunter (Native)       ")
    print("==================================================")
    
    print("Testing M_89 (2^89 - 1, 27 digits)...")
    let m89 = mersenne_number(89)
    print(m89)
    let is_m89_prime = lucas_lehmer_test(89)
    print("Is M_89 prime?")
    print(is_m89_prime)
    
    print("--------------------------------------------------")
    print("Testing M_107 (2^107 - 1, 33 digits)...")
    let m107 = mersenne_number(107)
    print(m107)
    let is_m107_prime = lucas_lehmer_test(107)
    print("Is M_107 prime?")
    print(is_m107_prime)

    print("--------------------------------------------------")
    print("Testing M_127 (2^127 - 1, 39 digits - Lucas 1876)...")
    let m127 = mersenne_number(127)
    print(m127)
    let is_m127_prime = lucas_lehmer_test(127)
    print("Is M_127 prime?")
    print(is_m127_prime)
