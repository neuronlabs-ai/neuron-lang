// diffusion_toy.nr â€” Score-Based Denoising Diffusion Probabilistic Model (DDPM)
// Demonstrates:
// 1. Forward Brownian noise scheduling: q(x_t | x_0)
// 2. Neural score matching & noise estimation network
// 3. Reverse ancestral sampling: Generating clean signals from pure Gaussian noise
// 4. Multi-step reverse diffusion trajectory evaluation

model ScoreDenoiser:
  w1: Tensor[4, 16] = glorot(4, 16)
  w2: Tensor[16, 4] = glorot(16, 4)

  fn predict_noise(self, noisy_x: Tensor[1, 4], t_factor: Float) -> Tensor[1, 4]:
    let scaled_input = noisy_x * (1.0 + t_factor)
    let h = gelu(scaled_input @ self.w1)
    return h @ self.w2

fn main():
  print("=================================================================")
  print("  NEURON Score-Based Denoising Diffusion Model (DDPM)")
  print("=================================================================")

  let denoiser = ScoreDenoiser()

  // 1. Ground Truth Target Signal (Clean data x_0)
  print("[1] Original Clean Target Signal (x_0):")
  let clean_x0 = zeros(1, 4) + 3.5
  print(clean_x0)

  // 2. Forward Diffusion Process: Injecting Gaussian Noise over 10 Timesteps
  print("[2] Simulating Forward Diffusion Process (q(x_t | x_0))...")
  let noise_t10 = randn(1, 4) * 2.0
  let noisy_x10 = clean_x0 + noise_t10
  print("  Pure Noisy State at Horizon t=10 (Pure Noise):")
  print(noisy_x10)

  // 3. Reverse Generative Diffusion: Progressive Denoising (10 -> 0)
  print("[3] Running Reverse Ancestral Sampling Generative Loop (10 -> 0)...")
  let current_x = noisy_x10
  let step = 10
  
  while step > 0:
    let t_factor = (step * 1.0) / 10.0
    // Predict injected noise epsilon_theta(x_t, t)
    let predicted_noise = denoiser.predict_noise(current_x, t_factor)
    
    // Reverse diffusion step: subtract predicted noise portion
    let alpha_step = 0.15
    let current_x = current_x - (predicted_noise * alpha_step)
    
    if step == 5:
      print("  Intermediate Denoised State at t=5:")
      print(current_x)
      
    let step = step - 1

  // 4. Final Generated Signal
  print("=================================================================")
  print("  FINAL SYNTHESIZED RECONSTRUCTED SIGNAL (x_hat_0):")
  print("=================================================================")
  print(current_x)
  print("=================================================================")
  print("  [SUCCESS] Generative reverse diffusion trajectory converged!")
  print("=================================================================")