// NEURON Prime Benchmark: Rust Reference Implementation
// Lucas-Lehmer test for Mersenne primes with modular multiplication
// Compile: rustc -O prime_hunt_rust.rs -o prime_hunt_rust

use std::time::Instant;

fn is_small_prime(n: i64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let mut d: i64 = 3;
    while d * d <= n {
        if n % d == 0 { return false; }
        d += 2;
    }
    true
}

fn mersenne_number(p: i64) -> i64 {
    (1_i64 << p) - 1
}

/// Modular multiplication: (a * b) % m using binary method
/// Avoids overflow for a, b < m < 2^62
fn mod_mul(mut a: i64, mut b: i64, m: i64) -> i64 {
    let mut result: i64 = 0;
    a %= m;
    while b > 0 {
        if b & 1 == 1 {
            result = (result + a) % m;
        }
        a = (a + a) % m;
        b >>= 1;
    }
    result
}

fn lucas_lehmer_test(p: i64) -> bool {
    if p == 2 { return true; }
    if !is_small_prime(p) { return false; }
    let mp = mersenne_number(p);
    let mut s: i64 = 4;
    for _ in 0..(p - 2) {
        s = (mod_mul(s, s, mp) - 2).rem_euclid(mp);
    }
    s == 0
}

fn hunt_mersenne(max_p: i64) -> f64 {
    println!("{}", "=".repeat(60));
    println!("  Rust Mersenne Prime Hunter (i64 mod_mul)");
    println!("{}", "=".repeat(60));

    let mut found = 0;
    let start = Instant::now();

    for p in 2..=max_p {
        if is_small_prime(p) && lucas_lehmer_test(p) {
            let mp = mersenne_number(p);
            let digits = format!("{}", mp).len();
            println!("  M_{} = 2^{}-1  ({} digits)", p, p, digits);
            found += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("{}", "-".repeat(60));
    println!("  Found {} Mersenne primes in {:.4}s", found, elapsed);
    println!("{}", "=".repeat(60));
    elapsed
}

fn main() {
    let max_p: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(61);
    
    hunt_mersenne(max_p);
}
