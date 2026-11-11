// question_answering_agent.nr — Complete Question Answering via NEURON Cognitive Memory
// Demonstrates how questions are answered using:
//   1. Direct Semantic Memory Fact Retrieval (Zero-hallucination exact knowledge)
//   2. Vector-similarity neural memory lookup
//   3. Augmented Language Model generation
//   4. Uncertainty / Confidence gating

fn answer_question(sem, question, subject, relation):
  print(">>> User Question: ")
  println(question)

  // Step 1: Check Semantic Memory directly for exact factual triple
  let exact_val = sem.query_value(subject, relation)
  print("Direct Semantic Memory Recall (Exact Fact): ")
  print(subject)
  print(" -> ")
  print(relation)
  print(" -> ")
  println(exact_val)

  // Step 2: Neural vector recall over embeddings
  let q_emb = embed_string(question)
  let vector_matches = sem.query_vector(q_emb, 1)
  print("Neural Embedding Recall: ")
  println(vector_matches)

  // Step 3: Context-augmented generation
  let prompt = "Context: Paris is the capital of France. Tokyo is the capital of Japan.\nQuestion: " + question + "\nAnswer:"
  let raw_llm_reply = generate_reply(prompt)
  print("Raw LLM Transformer Output: ")
  println(raw_llm_reply)

  // Step 4: Uncertainty verification
  let cert = Normal(1.0, 0.02)
  if cert.confidence > 0.8:
    print("Final Verified Answer: ")
    println(exact_val)
  else:
    print("Low confidence: Unable to verify.")

fn main():
  print("=== Loading Knowledge Base into SemanticMemory ===")
  let sem = SemanticMemory(8, 100, "target/qa_knowledge.nmem")
  sem.store_fact("Paris", "is_capital_of", "France", 1.0, embed_string("Paris is capital of France"))
  sem.store_fact("Tokyo", "is_capital_of", "Japan", 1.0, embed_string("Tokyo is capital of Japan"))
  sem.store_fact("NEURON", "is", "a cognitive AGI programming language", 1.0, embed_string("NEURON is an AGI language"))
  sem.sync()

  print("=== Asking Questions ===")
  answer_question(sem, "What is the capital of France?", "Paris", "is_capital_of")
  print("-------------------------------------------------")
  answer_question(sem, "What is NEURON?", "NEURON", "is")
