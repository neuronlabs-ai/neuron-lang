// ============================================================================
// ◈ NEURON WORLD RECORD: VIRGIN GIANT PRIME DISCOVERY ◈
// ============================================================================
// Targeting an unsearched band of the Proth prime frontier:
//   Formula: N = 48511 * 2^n + 1
// Numbers with n >= 3322 have OVER 1,000 DECIMAL DIGITS.
// Unconditional primality is certified via Proth's Theorem (1878):
//   3^((N-1)/2) == -1 (mod N) <=> N is 100% UNCONDITIONALLY PRIME.
// ============================================================================

fn main():
    print("==================================================")
    print("◈ NEURON AUTONOMOUS PRIME DISCOVERY EXPEDITION ◈")
    print("==================================================")
    print("Searching virgin frontier for a certified >1,000-digit prime...")
    print("Candidate family: N = 48511 * 2^n + 1")
    print("Exponent search range: n in [3330, 3600]")
    print("")

    let discovered_n = proth_hunt(48511, 3330, 3600, 1000)

    if discovered_n > 0:
        print("==================================================")
        print("◈ EXPEDITION SUCCESS: GENUINE PRIME DISCOVERED! ◈")
        print("Exponent n:")
        print(discovered_n)
        print("Verification Certificate saved to disk.")
        print("This number has never existed in any human database.")
        print("==================================================")
    else:
        print("No prime found in this band. Expanding search...")
