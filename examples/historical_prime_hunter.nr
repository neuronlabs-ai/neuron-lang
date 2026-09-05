// NEURON Historical Mersenne Prime Hunter (Arbitrary-Precision)
// Verifies Mersenne primes across history: from antiquity to the computing age!

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

fn test_and_report(p: Int, label: String):
    print("--------------------------------------------------")
    print(label)
    print("Testing exponent p:")
    print(p)
    let is_prime = lucas_lehmer_test(p)
    if is_prime:
        print("  VERIFIED MERSENNE PRIME!")
    else:
        print("  Composite (not prime).")

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON ARBITRARY-PRECISION PRIME HUNTER       ║")
    print("║  ◈ Testing Historical Milestones                 ║")
    print("╚══════════════════════════════════════════════════╝")
    
    // Ancient & Renaissance
    test_and_report(2, "M_2 (Antiquity - Euclid)")
    test_and_report(3, "M_3 (Antiquity - Euclid)")
    test_and_report(7, "M_7 (Antiquity - Euclid)")
    test_and_report(13, "M_13 (1456 - Anonymous)")
    test_and_report(19, "M_19 (1588 - Pietro Cataldi)")
    test_and_report(31, "M_31 (1772 - Leonhard Euler)")
    test_and_report(61, "M_61 (1883 - Ivan Pervushin)")
    
    // 20th Century Hand Calculations
    test_and_report(89, "M_89 (1911 - Powers)")
    test_and_report(107, "M_107 (1914 - Powers)")
    test_and_report(127, "M_127 (1876 - Edouard Lucas, by hand!)")
    
    // Counterexamples (Marin Mersenne conjectures that failed)
    test_and_report(67, "M_67 (Marin Mersenne 1644 conjecture - actually composite)")
    
    // The Early Computer Era
    test_and_report(521, "M_521 (1952 - SWAC Computer, 157 digits)")
    test_and_report(607, "M_607 (1952 - SWAC Computer, 183 digits)")
    test_and_report(1279, "M_1279 (1952 - SWAC Computer, 386 digits)")
    test_and_report(2203, "M_2203 (1952 - SWAC Computer, 664 digits)")
    test_and_report(2281, "M_2281 (1952 - SWAC Computer, 687 digits)")
    test_and_report(3217, "M_3217 (1957 - ILLIAC I, 969 digits)")
    test_and_report(4423, "M_4423 (1957 - ILLIAC I, 1,332 digits!)")
    
    print("==================================================")
    print("All historical milestones processed successfully!")
