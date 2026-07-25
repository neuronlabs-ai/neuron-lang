// ═══════════════════════════════════════════════════════════════
//  medical_dosage.nr — Safe Drug Dosage Prediction with NEURON
//  
//  Demonstrates Uncertain[T] types preventing a medical AI from
//  administering a drug dose when prediction confidence is too low.
//  In PyTorch, the confidence would be silently ignored.
//  In NEURON, the compiler forces you to check it.
// ═══════════════════════════════════════════════════════════════

model PatientEncoder:
  w1: Tensor[6, 32] = glorot(6, 32)
  w2: Tensor[32, 16] = glorot(32, 16)

  fn forward(self, vitals: Tensor[1, 6]) -> Tensor[1, 16]:
    let h = relu(vitals @ self.w1)
    return h @ self.w2

model DosagePredictor:
  encoder: PatientEncoder = PatientEncoder()
  w_dose: Tensor[16, 1] = glorot(16, 1)
  w_conf: Tensor[16, 1] = glorot(16, 1)

  fn predict(self, vitals: Tensor[1, 6]) -> Uncertain[Tensor[1, 1]]:
    let features = self.encoder.forward(vitals)
    let dose = features @ self.w_dose
    let confidence = sigmoid(features @ self.w_conf)
    return Uncertain(dose, confidence)

fn administer_safe(prediction: Uncertain[Tensor[1, 1]], threshold: Float) -> Tensor[1, 1]:
  if prediction.confidence > threshold:
    return prediction.value
  else:
    return zeros(1, 1)

fn main():
  let predictor = DosagePredictor()

  // Patient vitals: [age, weight_kg, heart_rate, bp, creatinine, albumin]
  let patient_vitals = zeros(1, 6) + 0.5
  let target_dose = zeros(1, 1) + 0.25

  let epoch = 0
  while epoch < 50:
    let prediction = predictor.predict(patient_vitals)
    let dose = prediction.value
    let loss = mse(dose, target_dose)
    update predictor by adam(grad(loss), lr=1e-3)
    let epoch = epoch + 1

  let final_prediction = predictor.predict(patient_vitals)
  let safe_dose = administer_safe(final_prediction, 0.7)
  print(safe_dose)
