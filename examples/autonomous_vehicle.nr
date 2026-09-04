// ═══════════════════════════════════════════════════════════════
//  autonomous_vehicle.nr — Self-Driving Sensor Fusion with NEURON
//  
//  Demonstrates Temporal[T] types preventing a self-driving car
//  from using future sensor readings to make current decisions.
//  This exact bug class caused real autonomous vehicle crashes.
// ═══════════════════════════════════════════════════════════════

// ─────────────────────────────────────────
//  Sensor Fusion + Steering
// ─────────────────────────────────────────
model SensorFusion:
  w_cam: Tensor[8, 16] = glorot(8, 16)
  w_lid: Tensor[6, 16] = glorot(6, 16)
  w_steer: Tensor[32, 1] = glorot(32, 1)

  fn predict_steering(self, camera: Tensor[1, 8], lidar: Tensor[1, 6]) -> Tensor[1, 1]:
    let cam_feat = relu(camera @ self.w_cam)
    let lid_feat = relu(lidar @ self.w_lid)
    let combined = concat([cam_feat, lid_feat])
    return tanh(combined @ self.w_steer)

// ─────────────────────────────────────────
//  Safe Driving — Temporal[T] enforced
//  All sensor data must flow past → future
// ─────────────────────────────────────────
fn safe_drive(sensor_history: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let past_readings = sensor_history.before(20)
  let w = glorot(20, 1)
  let act = past_readings @ w
  return act.snapshot()

fn main() -> Tensor[1]:
  let sensors: Temporal[Tensor, past_to_future] = load("market_data.csv")
  return safe_drive(sensors)
