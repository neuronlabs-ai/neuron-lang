fn main():
  print("=== TESTING CAUSAL IDENTIFICATION (TIAN & PEARL DO-CALCULUS) ===")

  print("--- Test 1: Backdoor Criterion ---")
  let m1 = ConfoundedTrial()
  print(m1)
  let id1 = m1.identify("y", do="x")
  print("Identifiable:")
  print(id1.identifiable)
  print("Method:")
  print(id1.method)
  print("Adjustment set:")
  print(id1.adjustment_set)
  print("Formula:")
  print(id1.formula)
  let eff1 = m1.causal_effect("x", "y", 2.0)
  print("True Causal Effect of x=2 on y:")
  print(eff1)

  print("--- Test 2: Frontdoor Criterion ---")
  let m2 = SmokingFrontdoor()
  print(m2)
  let id2 = m2.identify("cancer", do="smoking")
  print("Identifiable:")
  print(id2.identifiable)
  print("Method:")
  print(id2.method)
  print("Adjustment set (Mediators):")
  print(id2.adjustment_set)
  print("Formula:")
  print(id2.formula)
  let eff2 = m2.causal_effect("smoking", "cancer", 1.0)
  print("True Causal Effect of smoking=1 on cancer:")
  print(eff2)

  print("--- Test 3: Provably Unidentifiable Hedge ---")
  let m3 = UnidentifiableModel()
  print(m3)
  let id3 = m3.identify("y", do="x")
  print("Identifiable:")
  print(id3.identifiable)
  print("Method:")
  print(id3.method)
  print("Hedge:")
  print(id3.hedge)

  print("--- Test 4: Confounding Bias Quantification ---")
  let bias1 = m1.confounding_bias("x", "y", 2.0)
  print("Omitted Variable Bias for x=2:")
  print(bias1)

  print("ALL CAUSAL IDENTIFICATION TESTS PASSED")

causal model ConfoundedTrial:
  z -> x
  z -> y
  x -> y: 2.5

causal model SmokingFrontdoor:
  smoking -> tar: 0.8
  tar -> cancer: 0.5
  smoking <-> cancer

causal model UnidentifiableModel:
  x -> y: 1.5
  x <-> y
