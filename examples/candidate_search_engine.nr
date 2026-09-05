// NEURON Automated Mersenne Prime Search Engine
// Scans an exponent range [min_p, max_p] for prime candidates and executes Lucas-Lehmer

fn is_prime_candidate(n: Int) -> Bool:
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

fn hunt_range(min_p: Int, max_p: Int):
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON PRIME HUNTER: CANDIDATE SWEEP          ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Searching exponents in range:")
    print(min_p)
    print("to:")
    print(max_p)
    print("--------------------------------------------------")
    
    let p = min_p
    let tested = 0
    let found = 0
    while p <= max_p:
        if is_prime_candidate(p):
            let tested = tested + 1
            print("Testing candidate exponent p:")
            print(p)
            let is_m_prime = mersenne_lucas_lehmer(p)
            if is_m_prime:
                print(">>> [!!!] FOUND MERSENNE PRIME [!!!] <<<")
                print("    Exponent p:")
                print(p)
                let found = found + 1
            else:
                print("    Composite.")
        let p = p + 1
    
    print("==================================================")
    print("Search sweep complete!")
    print("Total prime exponents tested:")
    print(tested)
    print("Mersenne primes found:")
    print(found)

fn main():
    // Example: Sweep between 2,000 and 4,500
    // Expected to find M_2203, M_2281, M_3217, M_4423
    hunt_range(2000, 4500)
