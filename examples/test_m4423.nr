// NEURON Testing M_2281 (687 digits) and M_4423 (1,332 digits - 1957 ILLIAC I)

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
    print("  ◈ NEURON Testing M_2281 (687 decimal digits)    ")
    print("==================================================")
    let is_m2281 = lucas_lehmer_test(2281)
    print("Is M_2281 prime? (Expected: true)")
    print(is_m2281)

    print("==================================================")
    print("  ◈ NEURON Testing M_4423 (1,332 decimal digits!) ")
    print("  ◈ Crossing the 1,000-digit frontier!            ")
    print("==================================================")
    let is_m4423 = lucas_lehmer_test(4423)
    print("Is M_4423 prime? (Expected: true)")
    print(is_m4423)
