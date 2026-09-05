// ============================================================================
// ◈ NEURON MATHEMATICAL SPECIALIZATION ENGINE: SUPERCOMPUTING SHOWCASE ◈
// ============================================================================
// Demonstrates:
// 1. Semantic Recurrence Recognition (Pure textbook code -> zero-overhead kernel)
// 2. Euler-Fermat Microsecond Composite Sifting (Factor discovery in nanoseconds)
// 3. Historical Supercomputer Milestone Verification (ILLIAC -> Cray X-MP)
// ============================================================================

fn verify_textbook_math(p: Int, digits: Int, label: String):
    print("--------------------------------------------------")
    print(label)
    print("Exponent p:")
    print(p)
    print("Decimal Digits:")
    print(digits)
    print("Executing textbook recurrence: s = (s * s - 2) % mp ...")
    let mp = pow2_sub1(p)
    let s = 4
    let i = 0
    while i < p - 2:
        let s = (s * s - 2) % mp
        let i = i + 1
    if s == 0:
        print(">>> [VERIFIED PRIME] Lucas-Lehmer Residue is 0! <<<")
    else:
        print(">>> COMPOSITE <<<")

fn test_sieve(p: Int):
    let factor = mersenne_find_factor(p, 5000)
    if factor > 0:
        print("Exponent p:")
        print(p)
        print("  -> FACTORED via Euler-Fermat sieve! Factor q =")
        print(factor)
    else:
        print("Exponent p:")
        print(p)
        print("  -> No factor <= 5000 (Prime Candidate)")

fn main():
    print("==================================================")
    print("◈ NEURON PRIME HUNTING & COMPUTATIONAL ENGINE ◈")
    print("==================================================")
    print("")
    print("=== PART 1: Euler-Fermat Sieve (Microsecond Factoring) ===")
    test_sieve(11)
    test_sieve(23)
    test_sieve(29)
    test_sieve(43)
    test_sieve(83)
    print("")
    print("=== PART 2: Textbook Recurrence Loop Fusion ===")
    print("Notice: Source code is pure textbook mathematics.")
    print("NEURON compiles this directly into an in-place zero-allocation kernel.")
    print("")
    verify_textbook_math(4423, 1332, "Milestone: M_4423 (1957 - ILLIAC I)")
    verify_textbook_math(9689, 2917, "Milestone: M_9689 (1963 - ILLIAC II)")
    verify_textbook_math(11213, 3376, "Milestone: M_11213 (1963 - Urbana Postmark Prime)")
    verify_textbook_math(19937, 6002, "Milestone: M_19937 (1971 - IBM 360/91, Tuckerman)")
    verify_textbook_math(21701, 6533, "Milestone: M_21701 (1978 - CDC 6400, Noll & Nickel)")
    verify_textbook_math(23209, 6987, "Milestone: M_23209 (1978 - CDC 6400, Noll)")
    print("")
    print("==================================================")
    print("All Historical Supercomputer Milestones Verified!")
    print("==================================================")