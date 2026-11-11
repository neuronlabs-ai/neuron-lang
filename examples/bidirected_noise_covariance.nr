// Demonstrate an explicit covariance for a shared exogenous noise edge.
// With unit noise variances and Cov(U_A, U_B) = 0.8, observing A updates B.

causal model SharedNoise:
  variables: [A, B]
  A <-> B: 0.8

fn main():
  let scm = SharedNoise()
  print("==================================================================")
  print("   NEURON EXPLICIT BIDIRECTED NOISE COVARIANCE TEST (Codex/Antigravity)")
  print("==================================================================")
  print("Causal Model initialized with Cov(U_A, U_B) = 0.8.")

  let evidence = [["A", 1.0]]
  let intervention = [["A", 0.0]]
  let query = ["B"]

  let cf = scm.counterfactual(evidence, intervention, query)
  print("Counterfactual Query B under evidence A=1.0 and do(A=0.0):")
  print("Expected B = 0.8 via off-diagonal noise covariance:")
  print(cf)
