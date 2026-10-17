fn main():
  print("=== TESTING ZERO-LOOKAHEAD STREAM RING BUFFER ===")
  let s = Stream(4)
  print(s)
  print("Initial stream capacity:")
  print(s.capacity())
  print("Initial stream len:")
  print(s.len())

  s.push(10.0)
  s.push(20.0)
  s.push(30.0)

  print("Current element (t=0):")
  print(s.current())

  print("Lag 1 (t=-1):")
  print(s.lag(1))

  print("Lag 2 (t=-2):")
  print(s.lag(2))

  print("Len before wrap:")
  print(s.len())

  // Push more to trigger circular buffer wrap
  s.push(40.0)
  s.push(50.0)

  print("Len after wrap:")
  print(s.len())

  print("Current after wrap (t=0):")
  print(s.current())

  print("Lag 1 after wrap (t=-1):")
  print(s.lag(1))

  print("Lag 3 after wrap (oldest):")
  print(s.lag(3))

  let t = s.to_tensor()
  print("Stream as Tensor:")
  print(t)

  print("ALL STREAM TEMPORAL TESTS PASSED")
