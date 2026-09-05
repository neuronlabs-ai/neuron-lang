// ============================================================================
// ◈ NEURON 53RD MERSENNE PRIME FRONTIER EXPEDITION ◈
// ============================================================================
// Features:
// 1. Euler-Fermat Trial Factor Sieve pre-filtering
// 2. Fault-Tolerant Gerbicz-style Checkpointing (auto-resume on disconnect)
// 3. Real-Time Telemetry (iteration speed, progress %, accurate ETA)
// 4. In-Place Squaring and Branchless Mersenne Folding
// ============================================================================

fn main():
    print("==================================================")
    print("◈ NEURON 53RD MERSENNE PRIME FRONTIER ENGINE ◈")
    print("==================================================")
    print("")
    print("Launching checkpointed frontier hunt on M_44497 (13,395 digits)...")
    print("Checkpoints saved every 5,000 iterations to 'checkpoint_M44497.json'")
    print("If Colab disconnects, re-running automatically resumes!")
    print("")
    
    let is_prime = mersenne_hunt_53rd(44497, 5000, "checkpoint_M44497.json")
    if is_prime:
        print(">>> [SUCCESS] M_44497 VERIFIED PRIME BY NEURON! <<<")
    else:
        print(">>> Candidate is Composite. <<<")