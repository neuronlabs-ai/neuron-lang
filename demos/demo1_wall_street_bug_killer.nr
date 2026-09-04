// ═══════════════════════════════════════════════════════════════
//  STAGE DEMO 1: "THE MILLION-DOLLAR BUG KILLER"
//  
//  A hedge fund quant wrote this backtest. It trained perfectly.
//  It backtested beautifully. It passed every Python unit test.
//  Then it lost $4.2 million on the first live trading day.
//  
//  Why? One line of code leaked FUTURE price data into a function
//  that should only see the PAST. Python can't catch this.
//  NEURON catches it at compile time — before a single dollar moves.
// ═══════════════════════════════════════════════════════════════

// The prediction model: must only see PAST market data
fn predict_signal(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let features = prices.before(20)   // ✓ Past 20 price bars only
  let w = glorot(20, 1)
  return features @ w

// ✗ THE BUG — a quant accidentally peeked 1 bar into the future
fn backtest_with_leak(prices: Temporal[Tensor, past_to_future]) -> Tensor[1]:
  let future_prices = prices.after(1)   // Peek 1 step into the FUTURE
  return predict_signal(future_prices)  // ERROR: future data flows into past-only fn

fn main() -> Tensor[1]:
  let prices: Temporal[Tensor, past_to_future] = load("market_data.csv")
  return backtest_with_leak(prices)
