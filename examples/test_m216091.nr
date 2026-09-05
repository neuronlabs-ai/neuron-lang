// NEURON Cray X-MP/24 Era: Testing M_216091 (65,050 decimal digits!)
// Discovered September 1985 by David Slowinski on the Cray X-MP/24 supercomputer
// The 30th known Mersenne Prime!

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON CRAY X-MP/24 PRIME HUNTER              ║")
    print("║  ◈ Testing M_216091 (65,050 Decimal Digits!)     ║")
    print("║  ◈ The 30th Known Mersenne Prime                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Exponent p: 216,091")
    print("Decimal digits: 65,050")
    print("Executing 216,089 squarings with O(N) Mersenne bitshift reduction...")
    let is_prime = mersenne_lucas_lehmer(216091)
    if is_prime:
        print(">>> SUCCESS: M_216091 IS VERIFIED PRIME! <<<")
        print("    Digits: 65,050")
        print("    Rank: 30th Known Mersenne Prime")
        print("    Discovered: September 1985 (David Slowinski)")
    else:
        print("Failed.")
