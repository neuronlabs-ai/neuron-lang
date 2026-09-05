// ============================================================================
// ◈ NEURON EXPEDITION: THE WALL-SUN-SUN (FIBONACCI-WIEFERICH) PRIME HUNT ◈
// ============================================================================
// Historical Significance:
//   - Posed in 1960 by D. D. Wall.
//   - Tied directly to the first case of Fermat's Last Theorem (Sun & Sun, 1992).
//   - Condition: F_{p - (p/5)} == 0 (mod p^2)
//   - Humanity's Score: ZERO Wall-Sun-Sun primes have EVER been discovered.
//   - Finding just ONE would solve a 65-year-old open conjecture in mathematics.
// ============================================================================

fn test_known_wieferich():
    print("--------------------------------------------------")
    print("PHASE 1: SANITY VALIDATION (WIEFERICH BENCHMARK)")
    print("Testing 2^(p-1) == 1 (mod p^2)")
    print("Only TWO Wieferich primes are known in history:")
    print("  - p = 1093 (Discovered in 1913 by Meissner)")
    print("  - p = 3511 (Discovered in 1922 by Beeger)")
    print("--------------------------------------------------")

    let check1093 = is_wieferich(1093)
    print("Is 1093 Wieferich Prime:")
    print(check1093)

    let check3511 = is_wieferich(3511)
    print("Is 3511 Wieferich Prime:")
    print(check3511)

    let check1097 = is_wieferich(1097)
    print("Is 1097 Wieferich Prime (Normal Prime):")
    print(check1097)

fn main():
    print("==================================================")
    print("◈ NEURON WALL-SUN-SUN PRIME EXPEDITION ENGINE ◈")
    print("==================================================")
    print("")

    // Phase 1: Verify on the only known Wieferich primes
    test_known_wieferich()
    print("")

    // Phase 2: Search for Wall-Sun-Sun primes
    print("==================================================")
    print("PHASE 2: THE 65-YEAR OPEN CONJECTURE HUNT")
    print("Searching for F_{p - (p/5)} == 0 (mod p^2)...")
    print("Zero primes in human history have satisfied this condition.")
    print("==================================================")

    // Fast multi-thousand prime sweep
    let found = wall_sun_sun_hunt(1000000, 1300000)

    if found > 0:
        print("\n==================================================")
        print(">>> ◈ HISTORIC BREAKTHROUGH: WALL-SUN-SUN FOUND! ◈ <<<")
        print("Prime p:")
        print(found)
        print("Verification Certificate saved to disk.")
        print("==================================================")
    else:
        print("")
        print("==================================================")
        print("Sweep complete. No Wall-Sun-Sun primes in this window.")
        print("Every tested prime continues the 65-year mystery!")
        print("==================================================")
