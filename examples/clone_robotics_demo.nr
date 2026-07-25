// ═══════════════════════════════════════════════════════════════
//  Clone Robotics Myofiber Controller — WORKING NEURON Program
//  A safe gait controller that compiles, runs, and outputs torque
// ═══════════════════════════════════════════════════════════════

// Safe muscle controller: reads past 20 fiber pressure readings
// and predicts the next joint torque command
fn predict_torque(fiber_history: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let prev_pressures = fiber_history.before(20)
  let w = glorot(20, 1)
  return prev_pressures @ w

// Safe gait planner: chains temporal-safe functions together
fn gait_planner(fibers: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  return predict_torque(fibers)

fn main() -> Tensor[1]:
  let fibers: Temporal[Tensor, past_to_future] = load("myofiber_data.csv")
  let torque = gait_planner(fibers)
  return torque
