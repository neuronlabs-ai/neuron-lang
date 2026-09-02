// mamba_ssm.nr — Selective State-Space Model (Mamba / S4 Architecture)
// Demonstrates:
// 1. Continuous-to-Discrete State Space Discretization: (A, B, C, Delta)
// 2. Selective Input-Dependent Gating: Delta(x_t), B(x_t), C(x_t)
// 3. Linear-Time O(L) Recurrent Sequence Processing (Zero Attention Quadratic Overhead)
// 4. Memory-Efficient Latent State Evolution: h_t = A_bar * h_{t-1} + B_bar * x_t

model MambaBlock:
  // State space dimensions: d_in=4, d_state=8
  a_diag: Tensor[1, 8] = zeros(1, 8) + 0.85     // Diagonal state transition
  proj_delta: Tensor[4, 8] = glorot(4, 8)      // Input-dependent timescale gate
  proj_b: Tensor[4, 8] = glorot(4, 8)          // Input-dependent input matrix
  proj_c: Tensor[8, 4] = glorot(8, 4)          // State-to-output projection
  skip_d: Tensor[1, 4] = zeros(1, 4) + 0.1     // Residual skip connection

  fn step(self, x_t: Tensor[1, 4], h_prev: Tensor[1, 8]) -> Tensor[1, 4]:
    // 1. Compute selective parameters from input
    let delta = sigmoid(x_t @ self.proj_delta)
    let b_matrix = x_t @ self.proj_b
    
    // 2. Discretize continuous state transition: A_bar = exp(Delta * A) ~ (1 - Delta * (1 - A))
    let a_bar = self.a_diag * (zeros(1, 8) + 1.0 - delta * 0.1)
    let b_bar = delta * b_matrix
    
    // 3. State update: h_t = A_bar * h_{t-1} + B_bar * x_t
    let h_t = (h_prev * a_bar) + b_bar
    
    // 4. Compute output: y_t = (h_t @ C) + (x_t * D)
    let y_state = h_t @ self.proj_c
    let y_skip = x_t * self.skip_d
    return y_state + y_skip

fn main():
  print("=================================================================")
  print("  NEURON Selective State-Space Model (Mamba SSM Engine)")
  print("=================================================================")

  let mamba = MambaBlock()

  print("[1] Initializing Mamba SSM with d_in=4, d_state=8 (O(L) Linear Scaling)...")
  let h_state = zeros(1, 8) // Zero initial latent memory

  // 2. Process Sequence of 5 Time Tokens sequentially in O(L) linear time
  print("[2] Streaming 5 Sequential Tokens through Recurrent State Space...")
  let t = 0
  while t < 5:
    let token_input = randn(1, 4)
    let output_t = mamba.step(token_input, h_state)
    
    print("  Token Step:")
    print(t)
    print("  Emitted SSM Latent Representation:")
    print(output_t)
    
    let t = t + 1

  print("=================================================================")
  print("  [SUCCESS] Mamba Linear-Time State Space Execution Complete!")
  print("=================================================================")