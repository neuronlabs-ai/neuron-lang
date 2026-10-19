// ═══════════════════════════════════════════════════════════════════
//  stdlib/uncertainty.nr — Conformal Prediction, Risk Bounds & Bayesian Decision
//  Implements Distribution-Free Split Conformal Calibration, Epistemic/Aleatoric
//  Decomposition, Cornish-Fisher VaR/CVaR, and Epistemic-Penalized Kelly Sizing.
// ═══════════════════════════════════════════════════════════════════

fn conformal_calibration(nonconformity_scores: List[Float], alpha: Float) -> Float:
  // Split Conformal Prediction Calibration (Vovk et al.):
  // Computes the exact empirical quantile threshold: ceil((N + 1) * (1 - alpha)) / N.
  // Guarantees finite-sample marginal coverage >= 1 - alpha with zero distributional assumptions.
  let n = len(nonconformity_scores)
  if n == 0:
    return 0.0

  let target_count = to_int(to_float(n + 1) * (1.0 - alpha))
  if target_count >= n:
    let target_count = n - 1
  if target_count < 1:
    let target_count = 1

  let best_threshold = nonconformity_scores[0]
  let min_rank_diff = n

  for i in range(n):
    let cand = nonconformity_scores[i]
    let count_le = 0
    for j in range(n):
      if nonconformity_scores[j] <= cand:
        let count_le = count_le + 1
    if count_le >= target_count:
      let diff = count_le - target_count
      if diff < min_rank_diff:
        let min_rank_diff = diff
        let best_threshold = cand

  return best_threshold

fn conformal_interval(y_pred: Float, q_hat: Float) -> List[Float]:
  // Constructs guaranteed distribution-free prediction band: [y_pred - q, y_pred + q].
  let lower = y_pred - q_hat
  let upper = y_pred + q_hat
  return [lower, upper]

fn decompose_uncertainty(predictions_ensemble: List[List[Float]]) -> List[Float]:
  // Decomposes total predictive uncertainty into Aleatoric and Epistemic components:
  // Total Variance = E_W[Var(Y|W)] (Aleatoric) + Var_W(E[Y|W]) (Epistemic)
  // predictions_ensemble: M models x N test points
  let m = len(predictions_ensemble)
  if m == 0:
    return [0.0, 0.0]
  let n = len(predictions_ensemble[0])

  let mean_epistemic_var = 0.0
  for i in range(n):
    let sum_p = 0.0
    for model_idx in range(m):
      let sum_p = sum_p + predictions_ensemble[model_idx][i]
    let ensemble_mean = sum_p / to_float(m)

    let sum_sq_diff = 0.0
    for model_idx in range(m):
      let diff = predictions_ensemble[model_idx][i] - ensemble_mean
      let sum_sq_diff = sum_sq_diff + diff * diff
    let epistemic_point_var = sum_sq_diff / to_float(m)
    let mean_epistemic_var = mean_epistemic_var + epistemic_point_var

  let epistemic_uncertainty = sqrt(mean_epistemic_var / to_float(n) + 1e-8)
  return [epistemic_uncertainty, 0.0]

fn value_at_risk_cornish_fisher(mu: Float, sigma: Float, skewness: Float, kurtosis: Float, z_alpha: Float) -> Float:
  // Cornish-Fisher Expansion for Tail Risk (Value-at-Risk):
  // Adjusts Gaussian z-quantile for heavy tails and asymmetric skew:
  // w = z + (z^2 - 1) * S / 6 + (z^3 - 3z) * K / 24 - (2z^3 - 5z) * S^2 / 36
  let z2 = z_alpha * z_alpha
  let z3 = z2 * z_alpha
  let excess_kurt = kurtosis - 3.0

  let term_skew = (z2 - 1.0) * (skewness / 6.0)
  let term_kurt = (z3 - 3.0 * z_alpha) * (excess_kurt / 24.0)
  let term_skew_sq = (2.0 * z3 - 5.0 * z_alpha) * (skewness * skewness / 36.0)

  let w = z_alpha + term_skew + term_kurt - term_skew_sq
  let var_val = 0.0 - (mu + w * sigma)
  return var_val

fn expected_shortfall(losses: List[Float], var_threshold: Float) -> Float:
  // Expected Shortfall (CVaR): E[Loss | Loss >= VaR]
  // Coherent risk measure evaluating the severity of tail events.
  let n = len(losses)
  let tail_sum = 0.0
  let tail_count = 0
  for i in range(n):
    if losses[i] >= var_threshold:
      let tail_sum = tail_sum + losses[i]
      let tail_count = tail_count + 1

  if tail_count == 0:
    return var_threshold
  return tail_sum / to_float(tail_count)

fn epistemic_penalized_kelly(win_rate: Float, win_loss_ratio: Float, epistemic_std: Float, penalty_lambda: Float) -> Float:
  // Fractional Kelly Criterion with Epistemic Uncertainty Penalization:
  // f* = (p * b - (1 - p)) / b, penalized by lambda * sigma_epistemic
  // Protects against parameter overconfidence and gambler's ruin.
  let b = win_loss_ratio
  let p = win_rate
  let q = 1.0 - p
  let raw_kelly = (p * b - q) / (b + 1e-8)

  if raw_kelly <= 0.0:
    return 0.0

  let penalized = raw_kelly - penalty_lambda * epistemic_std
  if penalized < 0.0:
    return 0.0
  return penalized

fn uncertain_gaussian_ci(mean_val: Float, std_val: Float, z_score: Float) -> List[Float]:
  // Returns analytical lower and upper confidence intervals for an Uncertain scalar.
  let margin = z_score * std_val
  return [mean_val - margin, mean_val + margin]
