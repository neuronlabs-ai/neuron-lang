fn is_prime(n: Int) -> Bool:
    if n < 2:
        return false
    if n == 2:
        return true
    if n % 2 == 0:
        return false
    let d = 3
    while d * d <= n:
        if n % d == 0:
            return false
        let d = d + 2
    return true

fn count_primes(limit: Int) -> Int:
    let count = 0
    let i = 2
    while i <= limit:
        if is_prime(i):
            let count = count + 1
        let i = i + 1
    return count

fn main():
    let p100 = count_primes(100)
    print("Primes up to 100:")
    print(p100)
