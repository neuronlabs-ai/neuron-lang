// NEURON Testing M_110503 (33,265 decimal digits!)
// Discovered January 1988 by Walter Colquitt and Luke Welsh on an NEC SX-2 supercomputer
// The 29th known Mersenne Prime!

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON NEC SX-2 PRIME HUNTER                  ║")
    print("║  ◈ Testing M_110503 (33,265 Decimal Digits!)     ║")
    print("║  ◈ The 29th Known Mersenne Prime                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Exponent p: 110,503")
    print("Decimal digits: 33,265")
    print("Executing 110,501 squarings with single-pass O(N) bitshift reduction...")
    let is_prime = mersenne_lucas_lehmer(110503)
    if is_prime:
        print(">>> SUCCESS: M_110503 IS VERIFIED PRIME! <<<")
        print("    Digits: 33,265")
        print("    Rank: 29th Known Mersenne Prime")
        print("    Discovered: January 1988 (Colquitt & Welsh)")
    else:
        print("Failed.")
