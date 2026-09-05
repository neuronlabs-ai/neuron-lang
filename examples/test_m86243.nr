// NEURON Cray X-MP Supercomputer Era: Testing M_86243 (25,962 decimal digits!)
// Discovered September 1982 by David Slowinski on the Cray X-MP

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON CRAY X-MP PRIME HUNTER                 ║")
    print("║  ◈ Testing M_86243 (25,962 Decimal Digits!)      ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Running 86,241 squarings of a 25,962-digit integer...")
    let is_prime = mersenne_lucas_lehmer(86243)
    if is_prime:
        print(">>> SUCCESS: M_86243 IS VERIFIED PRIME! <<<")
        print("    Digits: 25,962")
        print("    Discovered: September 1982 (Slowinski)")
    else:
        print("Failed.")
