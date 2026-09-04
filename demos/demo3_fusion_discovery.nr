// ═══════════════════════════════════════════════════════════════
//  STAGE DEMO 3: "NUCLEAR FUSION DISCOVERY"
//  
//  Compact stage version: 10 real MIT Alcator C-Mod shots.
//  Discovers the plasma density limit scaling law in ~30 seconds.
//
//  Real data: MIT Plasma Science & Fusion Center (PSFC)
//  264,385 time-resolved measurements from Alcator C-Mod Tokamak.
//
//  Scaling law: ln(n_GL) = α·ln(Ip) + β·ln(Bt) + γ·ln(a) + bias
// ═══════════════════════════════════════════════════════════════

model FusionScalingLaw:
  alpha: Tensor[1, 1] = zeros(1, 1)
  beta:  Tensor[1, 1] = zeros(1, 1)
  gamma: Tensor[1, 1] = zeros(1, 1)
  bias:  Tensor[1, 1] = zeros(1, 1)

  fn predict(self, ln_ip: Tensor[1, 1], ln_bt: Tensor[1, 1], ln_a: Tensor[1, 1]) -> Tensor[1, 1]:
    let term1 = ln_ip * self.alpha
    let term2 = ln_bt * self.beta
    let term3 = ln_a * self.gamma
    return term1 + term2 + term3 + self.bias

fn main():
  let discoverer = FusionScalingLaw()

  // ── 10 Real MIT Alcator C-Mod Shots (log-transformed) ──
  // ln(Ip_MA), ln(Bt_T), ln(a_m), ln(n_GL)
  let ln_ip_0  = zeros(1, 1) + -0.232817
  let ln_bt_0  = zeros(1, 1) + 1.698722
  let ln_a_0   = zeros(1, 1) + -1.512801
  let ln_ne_0  = zeros(1, 1) + 0.691176

  let ln_ip_1  = zeros(1, 1) + -0.243921
  let ln_bt_1  = zeros(1, 1) + 1.700422
  let ln_a_1   = zeros(1, 1) + -1.507683
  let ln_ne_1  = zeros(1, 1) + 0.673104

  let ln_ip_2  = zeros(1, 1) + -0.378739
  let ln_bt_2  = zeros(1, 1) + 1.682997
  let ln_a_2   = zeros(1, 1) + -1.516421
  let ln_ne_2  = zeros(1, 1) + 1.117978

  let ln_ip_3  = zeros(1, 1) + -0.678677
  let ln_bt_3  = zeros(1, 1) + 1.784360
  let ln_a_3   = zeros(1, 1) + -1.529410
  let ln_ne_3  = zeros(1, 1) + 0.389279

  let ln_ip_4  = zeros(1, 1) + -0.230795
  let ln_bt_4  = zeros(1, 1) + 1.701990
  let ln_a_4   = zeros(1, 1) + -1.541819
  let ln_ne_4  = zeros(1, 1) + 0.688160

  let ln_ip_5  = zeros(1, 1) + 0.190520
  let ln_bt_5  = zeros(1, 1) + 2.041667
  let ln_a_5   = zeros(1, 1) + -1.521926
  let ln_ne_5  = zeros(1, 1) + 0.637927

  let ln_ip_6  = zeros(1, 1) + -0.754609
  let ln_bt_6  = zeros(1, 1) + 1.735409
  let ln_a_6   = zeros(1, 1) + -1.525548
  let ln_ne_6  = zeros(1, 1) + 0.322381

  let ln_ip_7  = zeros(1, 1) + -0.682096
  let ln_bt_7  = zeros(1, 1) + 1.783441
  let ln_a_7   = zeros(1, 1) + -1.539317
  let ln_ne_7  = zeros(1, 1) + 0.631731

  let ln_ip_8  = zeros(1, 1) + -0.271682
  let ln_bt_8  = zeros(1, 1) + 1.657644
  let ln_a_8   = zeros(1, 1) + -1.494293
  let ln_ne_8  = zeros(1, 1) + 0.082815

  let ln_ip_9  = zeros(1, 1) + 0.004070
  let ln_bt_9  = zeros(1, 1) + 1.693077
  let ln_a_9   = zeros(1, 1) + -1.496249
  let ln_ne_9  = zeros(1, 1) + 0.082110

  // ── Training Loop: Discover Physical Scaling Law ──
  let epoch = 0
  while epoch < 500:
    let loss_total = zeros(1, 1)
    let pred_0 = discoverer.predict(ln_ip_0, ln_bt_0, ln_a_0)
    let loss_total = loss_total + mse(pred_0, ln_ne_0)
    let pred_1 = discoverer.predict(ln_ip_1, ln_bt_1, ln_a_1)
    let loss_total = loss_total + mse(pred_1, ln_ne_1)
    let pred_2 = discoverer.predict(ln_ip_2, ln_bt_2, ln_a_2)
    let loss_total = loss_total + mse(pred_2, ln_ne_2)
    let pred_3 = discoverer.predict(ln_ip_3, ln_bt_3, ln_a_3)
    let loss_total = loss_total + mse(pred_3, ln_ne_3)
    let pred_4 = discoverer.predict(ln_ip_4, ln_bt_4, ln_a_4)
    let loss_total = loss_total + mse(pred_4, ln_ne_4)
    let pred_5 = discoverer.predict(ln_ip_5, ln_bt_5, ln_a_5)
    let loss_total = loss_total + mse(pred_5, ln_ne_5)
    let pred_6 = discoverer.predict(ln_ip_6, ln_bt_6, ln_a_6)
    let loss_total = loss_total + mse(pred_6, ln_ne_6)
    let pred_7 = discoverer.predict(ln_ip_7, ln_bt_7, ln_a_7)
    let loss_total = loss_total + mse(pred_7, ln_ne_7)
    let pred_8 = discoverer.predict(ln_ip_8, ln_bt_8, ln_a_8)
    let loss_total = loss_total + mse(pred_8, ln_ne_8)
    let pred_9 = discoverer.predict(ln_ip_9, ln_bt_9, ln_a_9)
    let loss_total = loss_total + mse(pred_9, ln_ne_9)

    let loss = loss_total * 0.1

    if epoch % 100 == 0:
      print("Epoch:")
      print(epoch)
      print("  Loss:")
      print(loss)

    update discoverer by adam(grad(loss), lr=0.05)
    let epoch = epoch + 1

  // ── Report Discovered Physical Coefficients ──
  print("═══════════════════════════════════════════════════════")
  print("  DISCOVERED FUSION SCALING LAW (MIT Alcator C-Mod)")
  print("  ln(n_GL) = α·ln(Ip) + β·ln(Bt) + γ·ln(a) + bias")
  print("═══════════════════════════════════════════════════════")
  print("α (Plasma Current scaling):")
  print(discoverer.alpha)
  print("β (Toroidal Field scaling):")
  print(discoverer.beta)
  print("γ (Minor Radius scaling):")
  print(discoverer.gamma)
  print("bias:")
  print(discoverer.bias)
  print("═══════════════════════════════════════════════════════")
