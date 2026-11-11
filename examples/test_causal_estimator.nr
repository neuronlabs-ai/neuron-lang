// Estimate finite categorical interventional distributions from observations.

causal model ConfoundedStudy:
  variables: [Z, X, Y]
  Z -> X: 0.8
  Z -> Y: 0.6

causal model ConditionalChain:
  variables: [X, Z, Y]
  X -> Z: 0.7
  Z -> Y: 0.9

fn main():
  let study = ConfoundedStudy()
  let study_data = causal_dataset(["Z", "X", "Y"], [
    ["0", "0", "0"],
    ["0", "0", "0"],
    ["0", "0", "0"],
    ["0", "1", "1"],
    ["1", "0", "0"],
    ["1", "1", "0"],
    ["1", "1", "0"],
    ["1", "1", "0"]
  ])
  let backdoor_estimate = study.estimate_distribution(
    study_data,
    ["Y"],
    [["X", "1"]],
    []
  )
  print("=== BACKDOOR ESTIMATE: P(Y | do(X=1)) ===")
  print(backdoor_estimate.method)
  print(backdoor_estimate.formula)
  print(backdoor_estimate.distribution)

  let chain = ConditionalChain()
  let chain_data = causal_dataset(["X", "Z", "Y"], [
    ["0", "0", "0"],
    ["0", "0", "0"],
    ["0", "1", "1"],
    ["0", "1", "1"],
    ["1", "0", "0"],
    ["1", "0", "0"],
    ["1", "1", "1"],
    ["1", "1", "1"]
  ])
  let conditional_estimate = chain.estimate_distribution(
    chain_data,
    ["Y"],
    [["X", "0"]],
    [["Z", "1"]]
  )
  print("=== IDC ESTIMATE: P(Y | do(X=0), Z=1) ===")
  print(conditional_estimate.method)
  print(conditional_estimate.formula)
  print(conditional_estimate.distribution)
