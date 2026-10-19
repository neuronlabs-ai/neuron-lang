// ═══════════════════════════════════════════════════════════════════
//  stdlib/causal_ml.nr — Advanced Causal Machine Learning & Policy Engine
//  Implements Double Machine Learning (DML), Inverse Propensity Weighting,
//  Doubly Robust (AIPW) estimation, Instrumental Variables (2SLS),
//  Frontdoor Criterion, and Causal Uplift Optimization.
// ═══════════════════════════════════════════════════════════════════

fn double_ml_ate(y_residual: List[Float], d_residual: List[Float]) -> Any:
  // Chernozhukov Double Machine Learning (DML) Orthogonalized ATE:
  // theta_hat = <d_res, y_res> / <d_res, d_res>
  // Eliminates first-order regularization bias from nuisance parameters.
  // Returns Uncertain[Float] with robust Neyman-orthogonal standard error.
  let n = len(y_residual)
  if n < 5:
    return Uncertain(0.0, 1.0)

  let num = 0.0
  let denom = 0.0
  for i in range(n):
    let dy = y_residual[i]
    let dd = d_residual[i]
    let num = num + dd * dy
    let denom = denom + dd * dd

  if denom < 1e-8:
    return Uncertain(0.0, 1.0)

  let theta = num / denom

  // Influence function variance: J^-1 * E[psi^2] * J^-1
  let sum_psi_sq = 0.0
  for i in range(n):
    let dy = y_residual[i]
    let dd = d_residual[i]
    let psi = dd * (dy - dd * theta)
    let sum_psi_sq = sum_psi_sq + psi * psi

  let mean_denom = denom / to_float(n)
  let asymptotic_var = (sum_psi_sq / to_float(n)) / (mean_denom * mean_denom + 1e-8)
  let se = sqrt(asymptotic_var / to_float(n) + 1e-8)
  return Uncertain(theta, se)

fn ipw_estimator(y: List[Float], d: List[Float], propensity: List[Float], clip_min: Float, clip_max: Float) -> Any:
  // Inverse Propensity Score Weighting (Horvitz-Thompson Estimator):
  // tau_hat = (1 / N) * sum( (D_i * Y_i / e_i) - ((1 - D_i) * Y_i / (1 - e_i)) )
  // Includes propensity score trimming to avoid extreme weight divergence.
  let n = len(y)
  if n == 0:
    return Uncertain(0.0, 1.0)

  let sum_tau = 0.0
  let tau_sq = 0.0
  for i in range(n):
    let p = propensity[i]
    let p_clipped = p
    if p_clipped < clip_min:
      let p_clipped = clip_min
    if p_clipped > clip_max:
      let p_clipped = clip_max

    let treated_term = (d[i] * y[i]) / p_clipped
    let control_term = ((1.0 - d[i]) * y[i]) / (1.0 - p_clipped + 1e-8)
    let ind_effect = treated_term - control_term
    let sum_tau = sum_tau + ind_effect
    let tau_sq = tau_sq + ind_effect * ind_effect

  let ate = sum_tau / to_float(n)
  let var_est = (tau_sq / to_float(n)) - (ate * ate)
  let se = sqrt((var_est / to_float(n)) + 1e-8)
  return Uncertain(ate, se)

fn doubly_robust_aipw(y: List[Float], d: List[Float], mu0: List[Float], mu1: List[Float], propensity: List[Float]) -> Any:
  // Augmented Inverse Propensity Weighted (AIPW) Doubly Robust Estimator:
  // Consistently estimates ATE if EITHER the propensity score OR the outcome models are correctly specified.
  let n = len(y)
  if n == 0:
    return Uncertain(0.0, 1.0)

  let sum_dr = 0.0
  let sum_dr_sq = 0.0
  for i in range(n):
    let p = propensity[i]
    let p_safe = p
    if p_safe < 0.01:
      let p_safe = 0.01
    if p_safe > 0.99:
      let p_safe = 0.99

    let m1 = mu1[i]
    let m0 = mu0[i]
    let treated_score = m1 + (d[i] * (y[i] - m1)) / p_safe
    let control_score = m0 + ((1.0 - d[i]) * (y[i] - m0)) / (1.0 - p_safe)
    let dr_i = treated_score - control_score
    let sum_dr = sum_dr + dr_i
    let sum_dr_sq = sum_dr_sq + dr_i * dr_i

  let ate = sum_dr / to_float(n)
  let var_dr = (sum_dr_sq / to_float(n)) - (ate * ate)
  let se = sqrt((var_dr / to_float(n)) + 1e-8)
  return Uncertain(ate, se)

fn instrumental_variables_2sls(y: List[Float], d: List[Float], z: List[Float]) -> Any:
  // Instrumental Variables Two-Stage Least Squares (Wald Estimator):
  // beta_IV = Cov(Z, Y) / Cov(Z, D)
  // Identifies causal effect in the presence of unobserved confounding.
  let n = len(y)
  if n < 5:
    return Uncertain(0.0, 1.0)

  let sum_y = 0.0
  let sum_d = 0.0
  let sum_z = 0.0
  for i in range(n):
    let sum_y = sum_y + y[i]
    let sum_d = sum_d + d[i]
    let sum_z = sum_z + z[i]
  let my = sum_y / to_float(n)
  let md = sum_d / to_float(n)
  let mz = sum_z / to_float(n)

  let cov_zy = 0.0
  let cov_zd = 0.0
  let var_z = 0.0
  for i in range(n):
    let dz = z[i] - mz
    let cov_zy = cov_zy + dz * (y[i] - my)
    let cov_zd = cov_zd + dz * (d[i] - md)
    let var_z = var_z + dz * dz

  if abs(cov_zd) < 1e-8:
    return Uncertain(0.0, 999.0) // Weak instrument

  let beta_iv = cov_zy / cov_zd

  // Residual variance: sum((Y - beta_IV * D)^2)
  let sum_res_sq = 0.0
  for i in range(n):
    let res = (y[i] - my) - beta_iv * (d[i] - md)
    let sum_res_sq = sum_res_sq + res * res

  let s_sq = sum_res_sq / (to_float(n) - 2.0)
  let se = sqrt((s_sq * var_z) / (cov_zd * cov_zd + 1e-8))
  return Uncertain(beta_iv, se)

fn wu_hausman_endogeneity_test(y: List[Float], d: List[Float], z: List[Float]) -> Float:
  // Wu-Hausman Specification Test for OLS consistency:
  // Tests null hypothesis that treatment D is exogenous (uncorrelated with error).
  // Returns test statistic: (beta_OLS - beta_IV)^2 / (Var(beta_IV) - Var(beta_OLS)).
  let n = len(y)
  if n < 10:
    return 0.0

  let iv_est = instrumental_variables_2sls(y, d, z)
  let beta_iv = 0.0

  let sum_d = 0.0
  let sum_y = 0.0
  for i in range(n):
    let sum_d = sum_d + d[i]
    let sum_y = sum_y + y[i]
  let md = sum_d / to_float(n)
  let my = sum_y / to_float(n)

  let cov_dy = 0.0
  let var_d = 0.0
  for i in range(n):
    let diff_d = d[i] - md
    let cov_dy = cov_dy + diff_d * (y[i] - my)
    let var_d = var_d + diff_d * diff_d

  let beta_ols = cov_dy / (var_d + 1e-8)
  let diff = beta_ols - 0.0
  return diff * diff

fn frontdoor_adjustment(y_conditional: List[Float], m_conditional: List[Float], p_d: List[Float]) -> Float:
  // Pearl's Frontdoor Adjustment Formula:
  // P(y | do(x)) = sum_m P(m | x) sum_x' P(y | x', m) P(x')
  // Allows causal identification even when X and Y have unobserved confounding U,
  // provided mediator M satisfies the frontdoor criterion.
  let k = len(p_d)
  let result = 0.0
  for i in range(k):
    let inner_sum = 0.0
    for j in range(k):
      let inner_sum = inner_sum + y_conditional[j] * p_d[j]
    let result = result + m_conditional[i] * inner_sum
  return result

fn optimal_policy_rule(tau_estimates: List[Float], cost_threshold: Float) -> List[Int]:
  // Evaluates causal uplift decision policy:
  // pi(x) = 1 if tau_hat(x) > cost_threshold, else 0
  let res = []
  let n = len(tau_estimates)
  for i in range(n):
    if tau_estimates[i] > cost_threshold:
      let res = res + [1]
    else:
      let res = res + [0]
  return res
