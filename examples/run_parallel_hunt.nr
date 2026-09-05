// NEURON 12-Core Multi-Threaded Parallel Mersenne Prime Hunter
// Sweeps 669 candidate prime exponents across all CPU threads concurrently!

fn main():
    print("Launching 12-Core Parallel Mersenne Hunt across [2, 5000]...")
    let primes = parallel_mersenne_hunt(2, 5000)
    print("==================================================")
    print("Parallel Sweep Complete!")
    print("Total Mersenne primes discovered in range [2, 5000]:")
    print(primes)
