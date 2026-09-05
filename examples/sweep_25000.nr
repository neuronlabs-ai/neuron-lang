// NEURON 12-Core Parallel Hunt: Sweeping Exponents [5000, 25000]
// Testing thousands of candidates concurrently across all 12 CPU cores!

fn main():
    print("Launching 12-Core Parallel Mersenne Hunt across [5000, 25000]...")
    let primes = parallel_mersenne_hunt(5000, 25000)
    print("==================================================")
    print("Parallel Sweep Complete!")
    print("Discovered Mersenne primes in range [5000, 25000]:")
    print(primes)
