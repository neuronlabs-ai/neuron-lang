fn test_factor(p: Int):
    let factor = mersenne_find_factor(p, 5000)
    if factor > 0:
        print("Candidate exponent p:")
        print(p)
        print("  -> FACTORED via Euler-Fermat sieve! Factor q =")
        print(factor)
    else:
        print("Candidate exponent p:")
        print(p)
        print("  -> No factor found with k <= 5000. Probing Lucas-Lehmer...")
        let is_p = mersenne_lucas_lehmer(p)
        if is_p:
            print("  >>> [HIT] CONFIRMED MERSENNE PRIME! <<<")
        else:
            print("  -> Composite (via Lucas-Lehmer)")

fn main():
    print("=== Testing Euler-Fermat Trial Factoring Sieve ===")
    test_factor(3)
    test_factor(5)
    test_factor(7)
    test_factor(11)
    test_factor(13)
    test_factor(23)
    test_factor(29)
    test_factor(31)
    test_factor(43)
    test_factor(67)
    test_factor(83)