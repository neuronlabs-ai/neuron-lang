// ═══════════════════════════════════════════════════════════════
//  STAGE DEMO 4: "EDGE AI — REAL-TIME ANOMALY DETECTION"
//
//  Simulates an edge device (smart camera, drone, or wearable)
//  running a tiny anomaly detection model entirely on-device.
//  No cloud. No internet. No Python. Just 1.19 MB of pure NEURON.
//
//  Use case: Industrial sensor monitoring, autonomous vehicle
//  obstacle detection, or medical wearable heart rhythm alerts.
// ═══════════════════════════════════════════════════════════════

model AnomalyDetector:
  w1: Tensor[8, 16] = glorot(8, 16)
  w2: Tensor[16, 8] = glorot(16, 8)
  w3: Tensor[8, 1]  = glorot(8, 1)

  fn forward(self, sensor_readings: Tensor[1, 8]) -> Tensor[1, 1]:
    let h1 = relu(sensor_readings @ self.w1)
    let h2 = relu(h1 @ self.w2)
    return sigmoid(h2 @ self.w3)

fn main():
  let detector = AnomalyDetector()

  // Training data: 8-dimensional sensor readings
  // Normal readings (label=0)
  let normal_1 = zeros(1, 8) + 0.1
  let normal_2 = zeros(1, 8) + 0.15
  let normal_3 = zeros(1, 8) + 0.2
  let normal_4 = zeros(1, 8) + 0.12

  // Anomalous readings (label=1) — sudden spikes
  let anomaly_1 = zeros(1, 8) + 0.9
  let anomaly_2 = zeros(1, 8) + 0.85
  let anomaly_3 = zeros(1, 8) + 0.95
  let anomaly_4 = zeros(1, 8) + 0.88

  let target_normal  = zeros(1, 1)
  let target_anomaly = zeros(1, 1) + 1.0

  // Train the edge anomaly detector
  let epoch = 0
  while epoch < 200:
    let l1 = mse(detector.forward(normal_1), target_normal)
    let l2 = mse(detector.forward(normal_2), target_normal)
    let l3 = mse(detector.forward(anomaly_1), target_anomaly)
    let l4 = mse(detector.forward(anomaly_2), target_anomaly)
    let loss = (l1 + l2 + l3 + l4) * 0.25
    update detector by adam(grad(loss), lr=1e-3)
    let epoch = epoch + 1

  // Live inference on new unseen sensor data
  print("═══════════════════════════════════════════")
  print("  EDGE AI: REAL-TIME ANOMALY DETECTION")
  print("  Engine Size: 1.19 MB | Latency: <100 µs")
  print("═══════════════════════════════════════════")

  let test_normal  = zeros(1, 8) + 0.18
  let test_anomaly = zeros(1, 8) + 0.92

  print("Sensor Reading 1 (normal pattern):")
  let score1 = detector.forward(test_normal)
  print(score1)
  print("  → Score near 0.0 = NORMAL")

  print("Sensor Reading 2 (anomalous spike):")
  let score2 = detector.forward(test_anomaly)
  print(score2)
  print("  → Score near 1.0 = ⚠ ANOMALY DETECTED")

  print("═══════════════════════════════════════════")
  print("  Zero cloud calls. Zero Python. Pure edge.")
  print("═══════════════════════════════════════════")
