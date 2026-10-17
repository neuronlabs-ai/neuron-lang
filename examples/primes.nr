// primes.nr
// Prime number generator in NEURON using iterative trial division.

fn is_prime(n: Int) -> Bool:
    if n <= 1:
        return false
    if n <= 3:
        return true
    if n % 2 == 0:
        return false
    
    let divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return false
        let divisor = divisor + 2
    return true

fn main():
    print("---------------------------------------")
    print("NEURON: Prime Number Generator")
    print("---------------------------------------")
    
    let count = 0
    let limit = 30
    print("Searching for primes up to 30:")
    
    for n in range(limit + 1):
        if is_prime(n):
            print(n)
            let count = count + 1
            
    print("---------------------------------------")
    print("Completed trial division sweep.")
