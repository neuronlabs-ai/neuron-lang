// Multi-prompt test for TinyLlama GGUF on NEURON

fn main():
  print("=== NEURON LLM Multi-Prompt Test ===")
  
  let q1 = generate_reply("Explain how photosynthesis works in one sentence.")
  print("Q1 Result:")
  print(q1)
  
  print("-----------------------------------")
  
  let q2 = generate_reply("Write a Python function to add two numbers.")
  print("Q2 Result:")
  print(q2)
