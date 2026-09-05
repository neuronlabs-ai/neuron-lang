"""
NEURON Prime Benchmark: Python Reference Implementation
Lucas-Lehmer test for Mersenne primes with modular multiplication
"""
import time
import sys

def is_small_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

def mersenne_number(p: int) -> int:
    return (1 << p) - 1

def mod_mul(a: int, b: int, m: int) -> int:
    """Modular multiplication using binary method (for fairness with NEURON i64)"""
    result = 0
    a_cur = a % m
    b_cur = b
    while b_cur > 0:
        if b_cur & 1:
            result = (result + a_cur) % m
        a_cur = (a_cur + a_cur) % m
        b_cur >>= 1
    return result

def lucas_lehmer_test_modmul(p: int) -> bool:
    """Lucas-Lehmer with mod_mul (apples-to-apples with NEURON i64 version)"""
    if p == 2:
        return True
    if not is_small_prime(p):
        return False
    mp = mersenne_number(p)
    s = 4
    for _ in range(p - 2):
        s = (mod_mul(s, s, mp) - 2) % mp
    return s == 0

def lucas_lehmer_test_native(p: int) -> bool:
    """Lucas-Lehmer using Python native bigint (Python's advantage)"""
    if p == 2:
        return True
    if not is_small_prime(p):
        return False
    mp = mersenne_number(p)
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % mp
    return s == 0

def hunt_mersenne(max_p: int, use_modmul: bool = True):
    test_fn = lucas_lehmer_test_modmul if use_modmul else lucas_lehmer_test_native
    mode = "mod_mul (i64-fair)" if use_modmul else "native bigint"
    
    print(f"{'='*60}")
    print(f"  Python Mersenne Prime Hunter ({mode})")
    print(f"{'='*60}")
    
    found = 0
    start = time.perf_counter()
    
    for p in range(2, max_p + 1):
        if is_small_prime(p):
            if test_fn(p):
                mp = mersenne_number(p)
                digits = len(str(mp))
                print(f"  M_{p} = 2^{p}-1  ({digits} digits)")
                found += 1
    
    elapsed = time.perf_counter() - start
    print(f"{'-'*60}")
    print(f"  Found {found} Mersenne primes in {elapsed:.4f}s")
    print(f"{'='*60}")
    return elapsed

if __name__ == "__main__":
    max_p = int(sys.argv[1]) if len(sys.argv) > 1 else 61
    
    # Run mod_mul version (apples-to-apples with NEURON)
    t1 = hunt_mersenne(max_p, use_modmul=True)
    print()
    
    # Also run native bigint version if max_p > 61
    if max_p > 61:
        t2 = hunt_mersenne(max_p, use_modmul=False)
        print(f"\nSpeedup from native bigint: {t1/t2:.2f}x")
    else:
        t2 = hunt_mersenne(max_p, use_modmul=False)
        print(f"\nmod_mul vs native: {t1/t2:.2f}x ratio")
