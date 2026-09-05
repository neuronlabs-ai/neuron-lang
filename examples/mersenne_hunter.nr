// NEURON Prime Hunter: Lucas-Lehmer Test for Mersenne Primes (M_p = 2^p - 1)

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

// Lucas-Lehmer test for M_p = 2^p - 1 (for p where (M_p)^2 fits in i64, p <= 31)
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
        // S = (S * S - 2) % M_p
        let s2 = s * s - 2
        let s = s2 % mp
        let i = i + 1
    
    return s == 0

fn hunt_mersenne(max_p: Int):
    print("==================================================")
    print("  ◈ NEURON Mersenne Prime Hunter (Lucas-Lehmer)  ")
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
    print("Total Mersenne primes found:")
    print(found)

fn main():
    hunt_mersenne(31)
