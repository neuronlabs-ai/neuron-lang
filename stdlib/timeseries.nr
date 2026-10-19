// ═══════════════════════════════════════════════════════════════════
//  stdlib/timeseries.nr — Strictly Causal & Leak-Free Time-Series Engine
//  Enforces non-anticipative rolling statistics, microstructure metrics,
//  volatility estimators, and causal walk-forward partitioners.
// ═══════════════════════════════════════════════════════════════════

fn rolling_mean_causal(data: List[Float], window: Int) -> List[Float]:
  // Computes rolling mean strictly using historical information up to index i.
  // For i < window, computes historical cumulative average without lookahead.
  let res = []
  let n = len(data)
  let running_sum = 0.0
  for i in range(n):
    let running_sum = running_sum + data[i]
    if i < window:
      let denom = to_float(i) + 1.0
      let mean_val = running_sum / denom
      let res = res + [mean_val]
    else:
      let window_sum = 0.0
      for j in range(window):
        let window_sum = window_sum + data[i - j]
      let res = res + [window_sum / to_float(window)]
  return res

fn rolling_std_causal(data: List[Float], window: Int) -> List[Float]:
  // Computes causal rolling sample standard deviation without lookahead bias.
  let res = []
  let n = len(data)
  let means = rolling_mean_causal(data, window)
  for i in range(n):
    if i < 2:
      let res = res + [0.0]
    else:
      if i < window:
        let cur_mean = means[i]
        let sum_sq = 0.0
        for j in range(i + 1):
          let diff = data[j] - cur_mean
          let sum_sq = sum_sq + diff * diff
        let variance = sum_sq / to_float(i)
        let res = res + [sqrt(variance + 1e-8)]
      else:
        let cur_mean = means[i]
        let sum_sq = 0.0
        for j in range(window):
          let diff = data[i - j] - cur_mean
          let sum_sq = sum_sq + diff * diff
        let variance = sum_sq / (to_float(window) - 1.0)
        let res = res + [sqrt(variance + 1e-8)]
  return res

fn rolling_zscore_causal(data: List[Float], window: Int) -> List[Float]:
  // Strictly causal Z-Score standardization:
  // Z_t = (x_t - mu_{t-1}) / sigma_{t-1}
  // Uses strictly prior-step distribution parameters to prevent contemporaneous leakage.
  let res = []
  let n = len(data)
  let means = rolling_mean_causal(data, window)
  let stds = rolling_std_causal(data, window)
  for i in range(n):
    if i == 0:
      let res = res + [0.0]
    else:
      let prior_mean = means[i - 1]
      let prior_std = stds[i - 1]
      if prior_std < 1e-6:
        let res = res + [0.0]
      else:
        let z = (data[i] - prior_mean) / prior_std
        let res = res + [z]
  return res

fn ema_decay(data: List[Float], alpha: Float) -> List[Float]:
  // Exponential Moving Average with strictly causal recursion:
  // S_0 = x_0
  // S_t = alpha * x_t + (1 - alpha) * S_{t-1}
  let res = []
  let n = len(data)
  if n == 0:
    return res
  let cur = data[0]
  let res = res + [cur]
  for i in range(n - 1):
    let idx = i + 1
    let cur = alpha * data[idx] + (1.0 - alpha) * cur
    let res = res + [cur]
  return res

fn ema_halflife(data: List[Float], halflife: Float) -> List[Float]:
  // Exponential Moving Average parameterized by half-life decay.
  // alpha = 1 - exp(-ln(2) / halflife)
  let alpha = 1.0 - exp(0.0 - 0.69314718056 / halflife)
  return ema_decay(data, alpha)

fn order_flow_imbalance(bids: List[Float], asks: List[Float], bid_sizes: List[Float], ask_sizes: List[Float]) -> List[Float]:
  // High-frequency Order Flow Imbalance (OFI) metric (Cont, Kukanov, Stoikov):
  // Measures directional supply/demand pressure between consecutive orderbook snapshots.
  let res = []
  let n = len(bids)
  for i in range(n):
    if i == 0:
      let res = res + [0.0]
    else:
      let delta_b = 0.0
      if bids[i] > bids[i - 1]:
        let delta_b = bid_sizes[i]
      else:
        if bids[i] == bids[i - 1]:
          let delta_b = bid_sizes[i] - bid_sizes[i - 1]
        else:
          let delta_b = 0.0 - bid_sizes[i - 1]

      let delta_a = 0.0
      if asks[i] < asks[i - 1]:
        let delta_a = ask_sizes[i]
      else:
        if asks[i] == asks[i - 1]:
          let delta_a = ask_sizes[i] - ask_sizes[i - 1]
        else:
          let delta_a = 0.0 - ask_sizes[i - 1]

      let ofi = delta_b - delta_a
      let res = res + [ofi]
  return res

fn realized_volatility(returns: List[Float], window: Int) -> List[Float]:
  // Computes annualized realized volatility: sqrt(252 * sum(r^2) / window).
  let res = []
  let n = len(returns)
  let annual_factor = sqrt(252.0)
  for i in range(n):
    if i < window - 1:
      let res = res + [0.0]
    else:
      let sum_sq = 0.0
      for j in range(window):
        let r = returns[i - j]
        let sum_sq = sum_sq + r * r
      let vol = sqrt(sum_sq / to_float(window)) * annual_factor
      let res = res + [vol]
  return res

fn parkinson_volatility(highs: List[Float], lows: List[Float], window: Int) -> List[Float]:
  // Parkinson Extreme-Value Volatility Estimator:
  // sigma^2 = (1 / (4 * ln(2))) * (1 / window) * sum(ln(High / Low)^2)
  // ~5x more efficient than standard close-to-close volatility.
  let res = []
  let n = len(highs)
  let factor = 1.0 / (4.0 * 0.69314718056)
  let annual_factor = sqrt(252.0)
  for i in range(n):
    if i < window - 1:
      let res = res + [0.0]
    else:
      let sum_hl = 0.0
      for j in range(window):
        let h = highs[i - j]
        let l = lows[i - j]
        let ratio = h / (l + 1e-8)
        let log_ratio = log(ratio)
        let sum_hl = sum_hl + log_ratio * log_ratio
      let var_parkinson = factor * (sum_hl / to_float(window))
      let res = res + [sqrt(var_parkinson) * annual_factor]
  return res

fn autocorr(data: List[Float], lag: Int, window: Int) -> List[Float]:
  // Causal historical sample autocorrelation at specified lag:
  // rho_k = Cov(X_t, X_{t-k}) / Var(X_t) computed strictly on historical window.
  let res = []
  let n = len(data)
  for i in range(n):
    if i < window + lag - 1:
      let res = res + [0.0]
    else:
      let sum_x = 0.0
      let sum_y = 0.0
      for j in range(window):
        let sum_x = sum_x + data[i - j]
        let sum_y = sum_y + data[i - j - lag]
      let mx = sum_x / to_float(window)
      let my = sum_y / to_float(window)

      let cov = 0.0
      let var_x = 0.0
      let var_y = 0.0
      for j in range(window):
        let dx = data[i - j] - mx
        let dy = data[i - j - lag] - my
        let cov = cov + dx * dy
        let var_x = var_x + dx * dx
        let var_y = var_y + dy * dy
      let denom = sqrt(var_x * var_y + 1e-8)
      let res = res + [cov / denom]
  return res

fn walk_forward_splits(n_samples: Int, train_len: Int, test_len: Int, step: Int) -> List[List[Int]]:
  // Generates strictly causal walk-forward folds [train_start, train_end, test_start, test_end].
  // Mathematically guarantees test_start == train_end (temporal non-interference).
  let splits = []
  let cur_start = 0
  let done = 0
  while done == 0:
    let train_end = cur_start + train_len
    let test_end = train_end + test_len
    if test_end <= n_samples:
      let fold = [cur_start, train_end, train_end, test_end]
      let splits = splits + [fold]
      let cur_start = cur_start + step
    else:
      let done = 1
  return splits

fn sharpe_ratio_uncertain(returns: List[Float], risk_free_rate: Float) -> Any:
  // Computes annualized Sharpe ratio with Andrew Lo asymptotic standard error,
  // returning an Uncertain[Float] representation for rigorous risk gating.
  let n = len(returns)
  if n < 5:
    return Uncertain(0.0, 1.0)
  let sum_r = 0.0
  for i in range(n):
    let sum_r = sum_r + returns[i]
  let mean_r = sum_r / to_float(n)
  let excess_mean = mean_r - (risk_free_rate / 252.0)

  let sum_sq = 0.0
  for i in range(n):
    let diff = returns[i] - mean_r
    let sum_sq = sum_sq + diff * diff
  let std_r = sqrt(sum_sq / (to_float(n) - 1.0) + 1e-8)

  let daily_sharpe = excess_mean / std_r
  let annual_sharpe = daily_sharpe * sqrt(252.0)

  // Asymptotic Standard Error: SE = sqrt((1 + 0.5 * S^2) / N)
  let se = sqrt((1.0 + 0.5 * daily_sharpe * daily_sharpe) / to_float(n)) * sqrt(252.0)
  return Uncertain(annual_sharpe, se)

fn max_drawdown(prices: List[Float]) -> Float:
  // Computes historical peak-to-trough maximum drawdown strictly in chronological order.
  let n = len(prices)
  if n == 0:
    return 0.0
  let peak = prices[0]
  let max_dd = 0.0
  for i in range(n):
    let p = prices[i]
    if p > peak:
      let peak = p
    let dd = (peak - p) / (peak + 1e-8)
    if dd > max_dd:
      let max_dd = dd
  return max_dd
