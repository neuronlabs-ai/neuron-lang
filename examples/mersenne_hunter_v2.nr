// NEURON Prime Hunter v2: Lucas-Lehmer with Modular Multiplication
// Can test Mersenne primes up to p=61 (i64 limit for 2^p-1)
// Uses Russian-peasant modular multiplication to avoid S*S overflow

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

// Compute 2^p - 1 for p <= 62
fn mersenne_number(p: Int) -> Int:
    let m = 1
    let i = 0
    while i < p:
        let m = m * 2
        let i = i + 1
    return m - 1

// Modular multiplication: (a * b) % m
// Uses Russian-peasant / binary method to avoid overflow
// Requires a, b < m and m < 2^62 (so a + a doesn't overflow i64)
fn mod_mul(a: Int, b: Int, m: Int) -> Int:
    let result = 0
    let a_cur = a % m
    let b_cur = b
    while b_cur > 0:
        if b_cur % 2 == 1:
            let result = (result + a_cur) % m
        let a_cur = (a_cur + a_cur) % m
        let b_cur = b_cur / 2
    return result

// Lucas-Lehmer test using modular multiplication
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
        // S = (S * S - 2) % M_p, using mod_mul to avoid overflow
        let s2 = mod_mul(s, s, mp)
        // s2 - 2 could go negative if s2 < 2, so handle it
        let s_next = s2 - 2
        if s_next < 0:
            let s = s_next + mp
        else:
            let s = s_next % mp
        let i = i + 1
    
    return s == 0

fn hunt_mersenne(max_p: Int):
    print("==================================================")
    print("  ◈ NEURON Mersenne Prime Hunter v2              ")
    print("  ◈ Modular multiplication (overflow-safe)       ")
    print("==================================================")
    let p = 2
    let found = 0
    while p <= max_p:
        if is_small_prime(p):
            let is_m_prime = lucas_lehmer_test(p)
            if is_m_prime:
                let mp = mersenne_number(p)
                print("FOUND MERSENNE PRIME:")
                print("  Exponent p:")
                print(p)
                print("  M_p = 2^p - 1:")
                print(mp)
                let found = found + 1
        let p = p + 1
    print("--------------------------------------------------")
    print("Total Mersenne primes found:")
    print(found)

fn main():
    hunt_mersenne(61)
