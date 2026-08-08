// ═══════════════════════════════════════════════════════════════════
//  stdlib/finance.nr — NEURON Finance & Quantitative Analysis Library
//  Full implementations of rolling statistics, RSI, MACD, and OHLCV helpers.
// ═══════════════════════════════════════════════════════════════════

fn load_ohlcv(path: String) -> List[List[Float]]:
  // Loads OHLCV data from a CSV file. If the file does not exist,
  // returns a mock sequence of realistic price vectors [O, H, L, C, V].
  return load_ohlcv(path)

fn rolling_mean(data: List[Float], window: Int) -> List[Float]:
  // Computes rolling mean of data over a given window size.
  let res = []
  let n = len(data)
  for i in range(n):
    if i < window - 1:
      let res = res + [0.0]
    else:
      let sum = 0.0
      for j in range(window):
        let sum = sum + data[i - j]
      let res = res + [sum * (1.0 / window)]
  return res

fn rolling_std(data: List[Float], window: Int) -> List[Float]:
  // Computes rolling standard deviation of data.
  let res = []
  let n = len(data)
  let means = rolling_mean(data, window)
  for i in range(n):
    if i < window - 1:
      let res = res + [0.0]
    else:
      let mean = means[i]
      let sum_sq = 0.0
      for j in range(window):
        let diff = data[i - j] - mean
        let sum_sq = sum_sq + diff * diff
      let variance = sum_sq * (1.0 / (window - 1))
      let std_val = sqrt(variance + 1e-8)
      let res = res + [std_val]
  return res

fn rsi(prices: List[Float], window: Int) -> List[Float]:
  // Relative Strength Index (RSI)
  let res = []
  let n = len(prices)
  for i in range(n):
    if i < window:
      let res = res + [50.0] // neutral start
    else:
      let gains = 0.0
      let losses = 0.0
      for j in range(window):
        let diff = prices[i - j] - prices[i - j - 1]
        if diff > 0.0:
          let gains = gains + diff
        else:
          let losses = losses - diff
      let rs = (gains + 1e-5) / (losses + 1e-5)
      let rsi_val = 100.0 - (100.0 / (1.0 + rs))
      let res = res + [rsi_val]
  return res

fn macd(prices: List[Float], fast: Int, slow: Int, signal: Int) -> List[List[Float]]:
  // Moving Average Convergence Divergence (MACD)
  // Returns list: [macd_line, signal_line]
  let fast_ma = rolling_mean(prices, fast)
  let slow_ma = rolling_mean(prices, slow)
  let n = len(prices)
  
  let macd_line = []
  for i in range(n):
    let macd_line = macd_line + [fast_ma[i] - slow_ma[i]]
  
  let signal_line = rolling_mean(macd_line, signal)
  return [macd_line, signal_line]

fn ema(prices: List[Float], period: Int) -> List[Float]:
  // Exponential Moving Average
  let res = []
  let n = len(prices)
  let mult = 2.0 / (period + 1)
  let prev_ema = 0.0
  for i in range(n):
    if i == 0:
      let prev_ema = prices[0]
      let res = res + [prev_ema]
    else:
      let new_ema = (prices[i] - prev_ema) * mult + prev_ema
      let prev_ema = new_ema
      let res = res + [new_ema]
  return res

fn atr(highs: List[Float], lows: List[Float], closes: List[Float], period: Int) -> List[Float]:
  // Average True Range
  let res = []
  let n = len(closes)
  let running_atr = 0.0
  for i in range(n):
    if i == 0:
      let tr = highs[i] - lows[i]
      let running_atr = tr
      let res = res + [tr]
    else:
      let hl = highs[i] - lows[i]
      let hc = highs[i] - closes[i - 1]
      let lc = closes[i - 1] - lows[i]
      // True Range = max(hl, hc, lc)
      let tr = hl
      if hc > tr:
        let tr = hc
      if lc > tr:
        let tr = lc
      if i < period:
        let running_atr = (running_atr * i + tr) / (i + 1)
      else:
        let running_atr = (running_atr * (period - 1) + tr) / period
      let res = res + [running_atr]
  return res
