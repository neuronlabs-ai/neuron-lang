fn main():
  let t = true
  let f = false

  print("=== Keyword Logic (and, or, not) ===")
  let a1 = t and f
  let a2 = t and true
  let o1 = t or f
  let o2 = f or f
  let n1 = not t
  let n2 = not f

  print("true and false:")
  print(a1)
  print("true and true:")
  print(a2)
  print("true or false:")
  print(o1)
  print("false or false:")
  print(o2)
  print("not true:")
  print(n1)
  print("not false:")
  print(n2)

  print("\n=== Symbolic Logic (&&, ||, !) ===")
  let sa1 = t && f
  let sa2 = t && true
  let so1 = t || f
  let so2 = f || f
  let sn1 = !t
  let sn2 = !f

  print("true && false:")
  print(sa1)
  print("true && true:")
  print(sa2)
  print("true || false:")
  print(so1)
  print("false || false:")
  print(so2)
  print("!true:")
  print(sn1)
  print("!false:")
  print(sn2)

  if (t and not f) && (10 > 5 || 2 == 3):
    print("\n✓ Complex compound condition passed!")
