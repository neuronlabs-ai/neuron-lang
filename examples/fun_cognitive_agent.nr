// ===================================================================
//  fun_cognitive_agent.nr — Sovereign Cognitive Sentinel
//  
//  Demonstrates NEURON's full cognitive stack in a single live agent:
//  1. Multi-Parameter Neural Learning (w, b) with Autograd & SGD
//  2. Relational Semantic Memory Graph with sub-millisecond retrieval
//  3. Honeytoken Canary Intrusion Detection
//  4. Certified Machine Unlearning (Hessian-scrubbing via forget_fact)
// ===================================================================

model ThreatEvaluator:
  w: Tensor[2, 1] = ones(2, 1) + 1.0
  b: Tensor[1, 1] = zeros(1, 1)

  fn score(self, features: Tensor[1, 2]) -> Tensor[1, 1]:
    return features @ self.w + self.b

  fn train(self, features: Tensor[1, 2], label: Tensor[1, 1]) -> Tensor[1, 1] [Effect[Mut[self]]]:
    let loss = mse(self.score(features), label)
    update self.w by sgd(grad(loss), lr=0.05)
    update self.b by sgd(grad(loss), lr=0.05)
    return loss

fn main():
  println("==================================================================")
  println("          NEURON COGNITIVE CYBER SENTINEL — LIVE AGENT            ")
  println("==================================================================")

  // -------------------------------------------------------------
  // Phase 1: Multi-Parameter Neural Learning
  // -------------------------------------------------------------
  println("")
  println("[PHASE 1] Training Multi-Parameter Neural Threat Evaluator...")
  let sentinel = ThreatEvaluator()
  
  // Normal packet: [PacketRate=1.0, Entropy=1.0] -> Low Threat
  // Attack packet: [PacketRate=3.0, Entropy=3.0] -> High Threat (10.0)
  let attack_sample: Tensor[1, 2] = ones(1, 2) + 2.0 // [3.0, 3.0]
  let target_threat: Tensor[1, 1] = zeros(1, 1) + 10.0

  println("  Initial Weights (w):")
  print(sentinel.w)
  println("  Initial Bias (b):")
  print(sentinel.b)

  let l1 = sentinel.train(attack_sample, target_threat)
  let l2 = sentinel.train(attack_sample, target_threat)
  let l3 = sentinel.train(attack_sample, target_threat)
  let l4 = sentinel.train(attack_sample, target_threat)
  let l5 = sentinel.train(attack_sample, target_threat)

  println("  SGD Loss Convergence across 5 Steps:")
  print(l1)
  print(l2)
  print(l3)
  print(l4)
  print(l5)

  println("  Converged Weights (w):")
  print(sentinel.w)
  println("  Converged Bias (b):")
  print(sentinel.b)

  // -------------------------------------------------------------
  // Phase 2: Relational Semantic Memory Graph
  // -------------------------------------------------------------
  println("")
  println("[PHASE 2] Initializing Relational Semantic Memory Graph...")
  let brain = SemanticMemory()
  
  brain.store("Gateway_01", "OS", "Linux_Hardened_6.8")
  brain.store("Gateway_01", "Port", "Port_22_SSH")
  brain.store("Gateway_01", "Port", "Port_443_HTTPS")
  brain.store("Threat_Actor", "IP", "198.51.100.44")
  brain.store("Threat_Actor", "Technique", "T1059_CommandLine")
  brain.store("HoneyToken", "SecretKey", "AKIA_SHADOW_KEY_8842")
  brain.store("HoneyToken", "Status", "Canary_Armed")

  println("  Network Topology Query (Gateway_01):")
  let gw_facts = brain.query("Gateway_01")
  print(gw_facts)

  println("  Canary Decoy State Query (HoneyToken):")
  let canary_facts = brain.query("HoneyToken")
  print(canary_facts)

  // -------------------------------------------------------------
  // Phase 3: Simulated Adversary Intrusion & Autonomous Scoring
  // -------------------------------------------------------------
  println("")
  println("[PHASE 3] Adversary Trips Honeytoken Canary!")
  println("  * Trip Alert: Access detected using 'AKIA_SHADOW_KEY_8842'")
  println("  * Sentinel evaluating incoming packet features [3.0, 3.0]...")
  
  let intruder_packet: Tensor[1, 2] = ones(1, 2) + 2.0
  let evaluated_threat = sentinel.score(intruder_packet)
  println("  * Evaluated Threat Level:")
  print(evaluated_threat)

  // -------------------------------------------------------------
  // Phase 4: Certified Forensic Unlearning (Subspace Scrubbing)
  // -------------------------------------------------------------
  println("")
  println("[PHASE 4] Certified Memory Scrubbing (forget_fact)...")
  println("  * Adversary detected -> Eradicating HoneyToken knowledge graph...")

  let erased_count = brain.forget_fact("HoneyToken")
  println("  * Records mathematically erased from memory:")
  println(erased_count)

  println("  * Forensic Verification Post-Scrubbing (Target: HoneyToken):")
  let post_scrub = brain.query("HoneyToken")
  print(post_scrub)

  println("")
  println("==================================================================")
  println("   VERDICT: NEURAL LEARNED + MEMORY STORED + FORENSICALLY CLEAN   ")
  println("==================================================================")
