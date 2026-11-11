// ================================================================
// NEURON Recursive Tian-Shpitser ID Algorithm — Stress Tests
// ================================================================
// Tests 3 classic causal identification scenarios:
//   1. Bow-graph (non-identifiable: X -> Y with X <-> Y)
//   2. Frontdoor criterion with latent confounding
//   3. Napkin graph (4-variable, identifiable via recursive ID)
// ================================================================

// --- Test 1: Bow Graph (NON-IDENTIFIABLE) ---
// X -> Y with X <-> Y confounding.
// This is the canonical non-identifiable structure.
causal model BowGraph:
  variables: [X, Y]
  X -> Y: 0.5
  X <-> Y

fn test_bow_graph():
  let scm = BowGraph()
  let result = scm.identify("Y", "X")
  print("=== BOW GRAPH (Expected: NOT identifiable) ===")
  print(result.identifiable)
  print(result.method)
  print(result.formula)
  print(result.hedge_outer)
  print(result.hedge_inner)

// --- Test 2: Frontdoor with Latent Confounding ---
// X -> M -> Y, with X <-> Y confounded.
// Identifiable via frontdoor criterion.
causal model FrontdoorLatent:
  variables: [X, M, Y]
  X -> M: 0.6
  M -> Y: 0.8
  X <-> Y

fn test_frontdoor():
  let scm = FrontdoorLatent()
  let result = scm.identify("Y", "X")
  print("=== FRONTDOOR (Expected: identifiable) ===")
  print(result.identifiable)
  print(result.method)
  print(result.formula)

// --- Test 3: Canonical Napkin Graph ---
// W -> Z -> X -> Y, with latent confounding W <-> X and W <-> Y.
// Identifiable despite latent confounding via the recursive ID path.
causal model NapkinGraph:
  variables: [W, X, Y, Z]
  W -> Z: 0.7
  Z -> X: 0.6
  X -> Y: 0.5
  W <-> X
  W <-> Y

fn test_napkin():
  let scm = NapkinGraph()
  let result = scm.identify("Y", "X")
  print("=== NAPKIN GRAPH (Expected: identifiable via recursive ID) ===")
  print(result.identifiable)
  print(result.method)
  print(result.formula)

// --- Test 4: Multi-outcome joint identification ---
// Two independent causal chains: X -> Y and A -> B.
// Joint P(Y,B | do(X,A)) should be identifiable.
causal model JointChains:
  variables: [X, Y, A, B]
  X -> Y: 0.4
  A -> B: 0.9

fn test_joint():
  let scm = JointChains()
  let result = scm.identify(["Y", "B"], ["X", "A"])
  print("=== JOINT MULTI-OUTCOME (Expected: identifiable) ===")
  print(result.identifiable)
  print(result.method)
  print(result.formula)

fn main():
  test_bow_graph()
  print("")
  test_frontdoor()
  print("")
  test_napkin()
  print("")
  test_joint()
  print("")
  print("=== ALL RECURSIVE ID STRESS TESTS COMPLETE ===")
