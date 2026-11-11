// Conditional causal identification (IDC) examples.

causal model ExchangeChain:
  variables: [X, Z, Y]
  X -> Z: 0.7
  Z -> Y: 0.8

causal model RatioWithLatentConfounding:
  variables: [X, Y, Z]
  X -> Y: 0.6
  Y <-> Z

fn main():
  let exchange = ExchangeChain()
  let exchanged = exchange.identify_conditional("Y", "X", "Z")
  print("=== IDC CONDITION EXCHANGE ===")
  print(exchanged.identifiable)
  print(exchanged.method)
  print(exchanged.formula)
  print(exchanged.formula_ast.expression.kind)

  let ratio_model = RatioWithLatentConfounding()
  let ratio = ratio_model.identify_conditional("Y", "X", "Z")
  print("=== IDC JOINT-OVER-MARGINAL ===")
  print(ratio.identifiable)
  print(ratio.method)
  print(ratio.formula)
  print(ratio.formula_ast.expression.kind)
