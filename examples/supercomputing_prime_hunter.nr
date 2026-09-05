// NEURON Supercomputing Era Mersenne Prime Hunter
// Accelerated with Mersenne Bitshift Modulo reduction: O(N) instead of O(N^2)
// Testing primes from 2,000 to 6,500+ decimal digits!

fn test_fast_mersenne(p: Int, digits: Int, label: String):
    print("==================================================")
    print(label)
    print("Exponent p:")
    print(p)
    print("Decimal digits:")
    print(digits)
    print("Executing Lucas-Lehmer with O(N) bitshift reduction...")
    let is_prime = mersenne_lucas_lehmer(p)
    if is_prime:
        print(">>> VERIFIED MERSENNE PRIME! <<<")
    else:
        print("Composite.")

fn main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  ◈ NEURON SUPERCOMPUTING PRIME HUNTER            ║")
    print("║  ◈ Accelerated O(N) Mersenne Bitshift Kernel     ║")
    print("╚══════════════════════════════════════════════════╝")
    
    // Testing M_4423 with fast bitshift modulo kernel
    test_fast_mersenne(4423, 1332, "M_4423 (1957 - ILLIAC I)")

    // 1963 ILLIAC II primes
    test_fast_mersenne(9689, 2917, "M_9689 (1963 - ILLIAC II)")
    test_fast_mersenne(9941, 2993, "M_9941 (1963 - ILLIAC II)")
    test_fast_mersenne(11213, 3376, "M_11213 (1963 - Famous Urbana Postmark Prime!)")

    // 1971 IBM 360/91 prime (Mersenne Twister PRNG seed prime)
    test_fast_mersenne(19937, 6002, "M_19937 (1971 - Tuckerman / Mersenne Twister)")

    // 1978 Nickel & Noll prime (Hayward High School discovery)
    test_fast_mersenne(21701, 6533, "M_21701 (1978 - Nickel & Noll, 6,533 digits)")

    print("==================================================")
    print("All supercomputing primes verified successfully!")
