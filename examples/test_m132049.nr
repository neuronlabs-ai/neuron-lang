// NEURON Cray X-MP Era: Testing M_132049 (39,751 decimal digits!)
// Discovered September 1983 by David Slowinski on the Cray X-MP supercomputer

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON CRAY X-MP PRIME HUNTER                 ║")
    print("║  ◈ Testing M_132049 (39,751 Decimal Digits!)     ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Running 132,047 squarings of a 39,751-digit integer...")
    let is_prime = mersenne_lucas_lehmer(132049)
    if is_prime:
        print(">>> SUCCESS: M_132049 IS VERIFIED PRIME! <<<")
        print("    Digits: 39,751")
        print("    Discovered: September 1983 (Slowinski / Cray X-MP)")
    else:
        print("Failed.")
