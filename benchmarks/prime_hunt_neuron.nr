// NEURON Prime Hunter v3: Benchmarkable Lucas-Lehmer
// Adds timing instrumentation for comparison with Python/Rust

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
        let s2 = mod_mul(s, s, mp)
        let s_next = s2 - 2
        if s_next < 0:
            let s = s_next + mp
        else:
            let s = s_next % mp
        let i = i + 1
    return s == 0

fn hunt_mersenne(max_p: Int):
    print("==================================================")
    print("  ◈ NEURON Mersenne Prime Hunter v3 (benchmark)  ")
    print("==================================================")
    let p = 2
    let found = 0
    while p <= max_p:
        if is_small_prime(p):
            let is_m_prime = lucas_lehmer_test(p)
            if is_m_prime:
                let mp = mersenne_number(p)
                print("  M_p found, exponent:")
                print(p)
                let found = found + 1
        let p = p + 1
    print("--------------------------------------------------")
    print("Total Mersenne primes found:")
    print(found)

fn main():
    hunt_mersenne(61)
