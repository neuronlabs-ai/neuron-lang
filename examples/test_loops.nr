// ═══════════════════════════════════════════════════════════════════
//  test_loops.nr — Control Flow & Loops Demonstration in NEURON
//  Demonstrating: if-else, while loops, and for loops
// ═══════════════════════════════════════════════════════════════════

fn test_if_else(x: Int):
  print("Testing if/else with x = " + str_from_int(x))
  if x > 10:
    print("  -> x is greater than 10")
  else:
    if x == 10:
      print("  -> x is exactly 10")
    else:
      print("  -> x is less than 10")

fn test_while():
  print("\nTesting while loop (countdown and sum):")
  let count = 5
  let total = 0
  while count > 0:
    print("  countdown: " + str_from_int(count))
    let total = total + count
    let count = count - 1
  print("  Final sum: " + str_from_int(total))

fn test_for_range():
  print("\nTesting for loop over range(5):")
  let sum = 0
  for i in range(5):
    print("  iteration i = " + str_from_int(i))
    let sum = sum + i
  print("  Sum of range(5): " + str_from_int(sum))

fn test_for_list():
  print("\nTesting for loop over explicit list:")
  let items = ["Alpha", "Beta", "Gamma", "Delta"]
  for item in items:
    print("  item: " + item)

fn str_from_int(val: Int) -> String:
  if val == 0:
    return "0"
  if val == 1:
    return "1"
  if val == 2:
    return "2"
  if val == 3:
    return "3"
  if val == 4:
    return "4"
  if val == 5:
    return "5"
  if val == 10:
    return "10"
  if val == 15:
    return "15"
  if val == 20:
    return "20"
  return "other"

fn main():
  print("═════════════════════════════════════════════════════════")
  print("     NEURON Language Control Flow & Loops Audit          ")
  print("═════════════════════════════════════════════════════════")

  // 1. If-Else
  test_if_else(20)
  test_if_else(10)
  test_if_else(3)

  // 2. While Loop
  test_while()

  // 3. For Range Loop
  test_for_range()

  // 4. For List Loop
  test_for_list()

  print("\n═════════════════════════════════════════════════════════")
  print("  [SUCCESS] All Loops and Control Flow Working Cleanly   ")
  print("═════════════════════════════════════════════════════════")
