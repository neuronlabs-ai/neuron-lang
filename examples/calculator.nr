fn main():
  print("═══════════════════════════════════════════")
  print("      NEURON Interactive CLI Calculator    ")
  print("═══════════════════════════════════════════")

  print("Enter first number:")
  let str_a = input()
  let a = parse_float(str_a)

  print("Enter second number:")
  let str_b = input()
  let b = parse_float(str_b)

  print("-------------------------------------------")
  print("Calculations:")

  print("Sum (a + b):")
  print(a + b)

  print("Difference (a - b):")
  print(a - b)

  print("Product (a * b):")
  print(a * b)

  print("Quotient (a / b):")
  print(a / b)

  print("Mean ((a + b) / 2):")
  print((a + b) / 2.0)
  print("═══════════════════════════════════════════")
