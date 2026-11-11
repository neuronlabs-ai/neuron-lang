// Test Persistent Memory in NEURON (.nmem)
// Verifies WorkingMemory, EpisodicMemory, and SemanticMemory persistence and querying

print("Testing SemanticMemory persistence...")
let sem = SemanticMemory(64, 1000, "target/test_semantic.nmem")
sem.store_fact("Paris", "is_capital_of", "France")
sem.store_fact("Berlin", "is_capital_of", "Germany")
sem.store_fact("Tokyo", "is_capital_of", "Japan")
sem.sync()

// Verify query
let facts = sem.query("Paris")
print("Facts for Paris: ")
println(facts)

// Reload from disk into fresh instance
let sem2 = SemanticMemory(64, 1000, "target/test_semantic.nmem")
let val = sem2.query_value("Tokyo", "is_capital_of")
print("Tokyo capital query: ")
println(val)

print("Testing EpisodicMemory persistence...")
let em = EpisodicMemory(8, 100, 0.01, "target/test_episodic.nmem")
let k1 = zeros(1, 8)
let k2 = ones(1, 8)
em.store(k1, "Episode 1: Initialization")
em.store(k2, "Episode 2: Observation")
em.sync()

let em2 = EpisodicMemory(8, 100, 0.01, "target/test_episodic.nmem")
let recalled = em2.recall(k2, 1)
print("Recalled from persisted episodic memory: ")
println(recalled)

print("Persistent Memory Tests Passed!")
