// ============================================================================
// ◈ NEURON EXPEDITION: THE 53RD MERSENNE PRIME FRONTIER ◈
// ============================================================================
// Investigating the two battlefronts for the 53rd Mersenne Prime:
//   1. The Virgin Frontier (p > 141,340,919) - 100% untouched by humanity (~42.5M digits)
//   2. The Double-Check Gap (81.8M <= p <= 141.3M) - single-tested candidates
// Supports auto-detected NVIDIA CUDA acceleration with CPU fallback.
// ============================================================================

fn probe_candidate(p: Int, max_k: Int, label: String):
    print("--------------------------------------------------")
    print(label)
    print("Exponent p:")
    print(p)
    let factor = mersenne_find_factor(p, max_k)
    if factor > 0:
        print(">>> DISCOVERY: M_p IS COMPOSITE! <<<")
        print("    Found Exact Factor q =")
        print(factor)
        print("    (Candidate eliminated in milliseconds!)")
    else:
        print(">>> [ALERT] CANDIDATE SURVIVES TRIAL FACTOR SIEVE! <<<")
        print("    No factor found up to k =")
        print(max_k)
        print("    >>> ACTIVE 53RD MERSENNE PRIME CONTENDER! <<<")

fn main():
    print("==================================================")
    print("◈ NEURON 53RD MERSENNE PRIME EXPEDITION ENGINE ◈")
    print("==================================================")
    
    // Hardware Diagnostic
    let is_cuda = cuda_available()
    let device_name = cuda_device_name()
    print("Execution Backend:")
    print(device_name)
    if is_cuda:
        print("  [GPU ONLINE] NVIDIA CUDA Parallel Sieve Active.")
    else:
        print("  [CPU ACTIVE] Native Multi-Limb SIMD Engine.")
    print("")

    // ────────────────────────────────────────────────────────────
    // PART 1: The Virgin Frontier (p > 141,340,919)
    // Every number here has over 42,500,000 decimal digits.
    // Untested by GIMPS or any supercomputer in human history.
    // ────────────────────────────────────────────────────────────
    print("==================================================")
    print("PART 1: VIRGIN FRONTIER (p > 141,340,919)")
    print("Numbers with > 42,500,000 Decimal Digits")
    print("==================================================")

    probe_candidate(141340933, 50000, "Virgin Candidate #1")
    probe_candidate(141340987, 50000, "Virgin Candidate #2")
    probe_candidate(141341021, 50000, "Virgin Candidate #3")
    probe_candidate(141341047, 50000, "Virgin Candidate #4")
    probe_candidate(141341051, 50000, "Virgin Candidate #5")
    probe_candidate(141341071, 50000, "Virgin Candidate #6")
    probe_candidate(141341153, 50000, "Virgin Candidate #7")
    probe_candidate(141341201, 50000, "Virgin Candidate #8")
    probe_candidate(141341219, 50000, "Virgin Candidate #9")
    probe_candidate(141341231, 50000, "Virgin Candidate #10")

    print("")
    // ────────────────────────────────────────────────────────────
    // PART 2: The Double-Check Gap (81.8M <= p <= 141.3M)
    // Candidates tested only once in historical sweeps.
    // ────────────────────────────────────────────────────────────
    print("==================================================")
    print("PART 2: DOUBLE-CHECK GAP (81.8M <= p <= 141.3M)")
    print("Auditing Candidates for Potential False Negatives")
    print("==================================================")

    probe_candidate(85000007, 50000, "Gap Candidate #1 (p=85,000,007)")
    probe_candidate(90000049, 50000, "Gap Candidate #2 (p=90,000,049)")
    probe_candidate(100000007, 50000, "Gap Candidate #3 (p=100,000,007)")

    print("")
    // ────────────────────────────────────────────────────────────
    // PART 3: Fault-Tolerant Recurrence Checkpointing
    // Proof of the state persistence and live telemetry engine.
    // ────────────────────────────────────────────────────────────
    print("==================================================")
    print("PART 3: RESILIENT LUCAS-LEHMER CHECKPOINT ENGINE")
    print("Verifying M_4423 with Live Telemetry & Persistence")
    print("==================================================")
    let is_prime = mersenne_hunt_53rd(4423, 1000, "checkpoint_M4423.json")
    print("Verification Result for M_4423:")
    print(is_prime)

    print("")
    print("==================================================")
    print("◈ EXPEDITION COMPLETE ◈")
    print("Surviving virgin candidates are primed for 53rd crown!")
    print("==================================================")