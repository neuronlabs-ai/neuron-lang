// NEURON Cray Supercomputer Era Mersenne Hunter
// Crossing 10,000 decimal digits!
// Testing M_23209 (1979) and M_44497 (1979 Cray-1 prime, 13,395 digits!)

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON CRAY-1 SUPERCOMPUTER PRIME HUNTER      ║")
    print("║  ◈ Crossing 10,000 Decimal Digits!               ║")
    print("╚══════════════════════════════════════════════════╝")
    
    print("--------------------------------------------------")
    print("Testing M_23209 (6,987 digits - 1979)...")
    let is_23209 = mersenne_lucas_lehmer(23209)
    if is_23209:
        print(">>> M_23209 VERIFIED PRIME! <<<")
    else:
        print("M_23209 composite.")

    print("--------------------------------------------------")
    print("Testing M_44497 (13,395 digits - 1979 Cray-1 Supercomputer)...")
    print("Running 44,495 squarings of a 13,395-digit number...")
    let is_44497 = mersenne_lucas_lehmer(44497)
    if is_44497:
        print(">>> M_44497 VERIFIED PRIME! (13,395 DIGITS) <<<")
    else:
        print("M_44497 composite.")
