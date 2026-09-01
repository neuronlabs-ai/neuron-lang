// ══════════════════════════════════════════════════════════════════════
//  NEURON Offset-Based Temporal Safety Demo (§3.1.1)
// ══════════════════════════════════════════════════════════════════════
//  Demonstrates compile-time temporal offset algebra (t + k),
//  lag/lead composition, and zero-lookahead boundary guarantees.
// ══════════════════════════════════════════════════════════════════════

fn execute_trading_strategy(features: Temporal[Tensor, 0]) -> Tensor:
  print("  Executing strategy on clean past-aligned features...")
  return features

fn forecast_5step(features: Temporal[Tensor, 0]) -> Temporal[Tensor, 5]:
  print("  Generating 5-step ahead multi-horizon forecast...")
  let pred: Temporal[Tensor, 5] = features.after(5)
  return pred

fn main():
  print("=================================================================")
  print("  NEURON Offset-Based Temporal Types (§3.1.1) Demo")
  print("=================================================================")

  // 1. Initialize raw price tensor at reference time t = 0
  let raw_prices: Temporal[Tensor, 0] = randn(20, 4)
  print("[1] Initialized price stream with Temporal[Tensor, 0]")

  // 2. Compute 5-period moving average lookback window -> offset becomes -5
  let sma5: Temporal[Tensor, -5] = raw_prices.shift(-5)
  print("[2] Computed 5-period lookback -> Temporal[Tensor, -5]")

  // 3. Apply 2-period signal forward shift -> (-5) + 2 = -3 (still in past!)
  let safe_signal = sma5.shift(2)
  print("[3] Applied .shift(2) -> Resulting offset: -3 (Safe past data)")

  // 4. Execute strategy expecting offset <= 0
  let order = execute_trading_strategy(safe_signal)
  print("[4] Verified: Strategy accepted -3 offset data with zero lookahead bias")

  // 5. Multi-horizon forecasting alignment
  let target_horizon5: Temporal[Tensor, 5] = randn(20, 1).after(5)
  let forecast = forecast_5step(raw_prices)
  print("[5] Aligned prediction Temporal[Tensor, +5] with target Temporal[Tensor, +5]")

  print("=================================================================")
  print("  [SUCCESS] All temporal offset algebra verified at compile-time!")
  print("=================================================================")