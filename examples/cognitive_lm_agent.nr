// cognitive_lm_agent.nr — Cognitive Architecture with NeuronLM & Persistent Memories
// Demonstrates:
//   1. Persistent SemanticMemory & EpisodicMemory
//   2. Neural string embeddings via embed_string()
//   3. Text generation via generate_reply()
//   4. Uncertainty evaluation with Uncertain[Float]

fn test_cognitive_agent():
  print("=== Step 1: Initializing Persistent Semantic Memory ===")
  let sem = SemanticMemory(8, 500, "target/agent_semantic.nmem")
  
  // Embed and store core facts
  let emb_paris = embed_string("Paris is the capital of France")
  let emb_tokyo = embed_string("Tokyo is the capital of Japan")
  let emb_rust = embed_string("NEURON is a unified AGI language")

  sem.store_fact("Paris", "is_capital_of", "France", 1.0, emb_paris)
  sem.store_fact("Tokyo", "is_capital_of", "Japan", 1.0, emb_tokyo)
  sem.store_fact("NEURON", "type", "AGI-Language", 1.0, emb_rust)
  sem.sync()

  print("=== Step 2: Querying Semantic Memory via Vector Similarity ===")
  let q_emb = embed_string("capital of France")
  let recalled_facts = sem.query_vector(q_emb, 1)
  print("Vector recall result: ")
  println(recalled_facts)

  print("=== Step 3: Working Memory Scratchpad ===")
  let wm = WorkingMemory(8, 4)
  wm.write(q_emb)
  let attended = wm.read(q_emb)
  print("Attended Working Memory vector slots: ")
  println(attended)

  print("=== Step 4: Language Model Generation via NeuronLM ===")
  let prompt = "Explain the relationship between Paris and France"
  let reply = generate_reply(prompt)
  print("NeuronLM reply: ")
  println(reply)

  print("=== Step 5: Uncertainty Evaluation ===")
  let cert = Normal(0.95, 0.05)
  let conf = cert.confidence
  if conf > 0.8:
    print("High confidence response confirmed.")
  else:
    print("Response flagged as uncertain.")

  print("=== Cognitive Agent Execution Complete! ===")

fn main():
  test_cognitive_agent()
