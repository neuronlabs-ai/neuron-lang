// Qwen 3.5 2B — Native NEURON Chat
// Uses the built-in NeuronLM engine with real dequantized GGUF weights

fn main():
  print("=== Qwen 3.5 2B on NEURON ===")
  let response = generate_reply("What is the capital of Kenya?")
  print(response)
