fn main():
  let mem = EpisodicMemory(64, 1000)
  let key1 = byte_encode("France capital", 64)
  let val1 = "Paris"
  mem.store(key1, val1)

  let key2 = byte_encode("Japan capital", 64)
  let val2 = "Tokyo"
  mem.store(key2, val2)

  let query = byte_encode("What is France capital?", 64)
  let results = mem.recall(query, 1)
  print("Recalled answer:")
  print(results)
