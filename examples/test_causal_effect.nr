// test_causal_effect.nr

causal model ClinicalStudy:
  variables: [severity, treatment, recovery]
  severity -> treatment: 0.6
  severity -> recovery: -0.5
  treatment -> recovery: 0.25

fn main():
  let study = ClinicalStudy()
  print("==================================================================")
  print("   NEURON STRUCTURAL CAUSAL MODEL & PEARL DO-CALCULUS ENGINE")
  print("==================================================================")
  
  let effect_treated = study.causal_effect("treatment", "recovery", 1.0)
  print("True Causal Effect E[Recovery | do(Treatment = 1.0)]:")
  print(effect_treated)

  let effect_control = study.causal_effect("treatment", "recovery", 0.0)
  print("True Control Baseline E[Recovery | do(Treatment = 0.0)]:")
  print(effect_control)

  let true_ate = effect_treated - effect_control
  print("Average Treatment Effect (ATE):")
  print(true_ate)

  let bias = study.confounding_bias("treatment", "recovery", 1.0)
  print("Confounding Bias (Observational Distortion):")
  print(bias)

  let backdoor_set = study.backdoor("treatment", "recovery")
  print("Pearl Backdoor Adjustment Conditioning Set:")
  print(backdoor_set)

  // Counterfactual Reasoning (Pearl Layer 3)
  // Evidence: Patient had severity=1.5, untreated (treatment=0.0)
  let evidence = [["severity", 1.5], ["treatment", 0.0]]
  // Counterfactual intervention: What if they WERE treated?
  let intervention = [["treatment", 1.0]]
  let query = ["recovery"]
  
  let cf_result = study.counterfactual(evidence, intervention, query)
  print("")
  print("--- Counterfactual Query (Pearl Layer 3) ---")
  print("Factual: Severe patient without drug. Counterfactual: What if they received drug?")
  print(cf_result)
