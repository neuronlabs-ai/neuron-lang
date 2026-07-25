// ═══════════════════════════════════════════════════════════════
//  clinical_trial.nr — Causal Treatment Effect Estimation
//  
//  Demonstrates Causal[T] types preventing a clinical AI from
//  confusing "patients who take drug X tend to recover" (correlation)
//  with "drug X causes recovery" (causation).
//  
//  In PyTorch, both are just float tensors.
//  In NEURON, they are different types that cannot be mixed.
// ═══════════════════════════════════════════════════════════════

// ─────────────────────────────────────────
//  Observational: what we SEE in the data
// ─────────────────────────────────────────
fn observational_recovery(trial, patient_data):
  let obs: Causal[Float, observed] = trial.observe(treatment = 1.0)
  return obs

// ─────────────────────────────────────────
//  Interventional: what happens when we
//  FORCE the treatment (randomized trial)
// ─────────────────────────────────────────
fn causal_recovery(trial):
  let forced: Causal[Float, intervened] = trial.intervene(treatment = 1.0)
  return forced

// ─────────────────────────────────────────
//  Safe Prescription — ONLY accepts
//  interventional (causal) evidence
// ─────────────────────────────────────────
fn should_prescribe(effect: Causal[Float, intervened]) -> Float:
  return effect

// ─────────────────────────────────────────
//  Outcome Predictor Network
// ─────────────────────────────────────────
model OutcomePredictor:
  w1: Tensor[4, 16] = glorot(4, 16)
  w2: Tensor[16, 1] = glorot(16, 1)

  fn predict(self, features: Tensor[1, 4]) -> Tensor[1, 1]:
    let h = relu(features @ self.w1)
    return sigmoid(h @ self.w2)

fn main():
  let predictor = OutcomePredictor()

  // Patient: [age, weight, biomarker_1, biomarker_2]
  let patient = zeros(1, 4) + 0.5
  let target = zeros(1, 1) + 0.8

  let epoch = 0
  while epoch < 40:
    let pred = predictor.predict(patient)
    let loss = mse(pred, target)
    update predictor by adam(grad(loss), lr=1e-3)
    let epoch = epoch + 1

  let result = predictor.predict(patient)
  print(result)
