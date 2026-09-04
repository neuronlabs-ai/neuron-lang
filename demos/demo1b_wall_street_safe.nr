// ═══════════════════════════════════════════════════════════════
//  STAGE DEMO 1b: "THE SAFE VERSION"
//  
//  The same trading strategy, written correctly.
//  NEURON compiles and runs it because no future data leaks.
// ═══════════════════════════════════════════════════════════════

fn predict_signal(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let features = prices.before(20)
  let w = glorot(20, 1)
  return features @ w

// ✓ CORRECT — only passes historical data downstream
fn backtest_safe(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  return predict_signal(prices)

fn main() -> Tensor[1]:
  let prices: Temporal[Tensor, past_to_future] = load("market_data.csv")
  return backtest_safe(prices)
