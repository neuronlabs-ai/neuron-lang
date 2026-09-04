// NEURON Website — Live In-Browser WebAssembly Engine & UI Logic
import init, { type_check, eval_neuron, transpile_to_python } from './neuron_wasm.js';

// ══════════════════════════════════════════════════
// 1. Curated Code Snippets & Descriptions
// ══════════════════════════════════════════════════
const codeSnippets = {
  autograd: `// 1. Native Autograd — Train a Neural Network in Browser WASM
model TinyNet:
  w1: Tensor[4, 8] = glorot(4, 8)
  w2: Tensor[8, 1] = glorot(8, 1)

  fn forward(self, x: Tensor[1, 4]) -> Tensor[1, 1]:
    let h = relu(x @ self.w1)
    return h @ self.w2

fn main():
  let net = TinyNet()
  let x = zeros(1, 4) + 0.5
  let y = zeros(1, 1) + 1.0

  print("Starting backpropagation training in browser WASM...")
  let epoch = 0
  while epoch < 30:
    let pred = net.forward(x)
    let loss = mse(pred, y)
    update net by adam(grad(loss), lr=0.05)
    let epoch = epoch + 1

  let final_pred = net.forward(x)
  print("Training converged! Final prediction:")
  print(final_pred)`,

  matmul: `// 2. High-Performance Tensor Math & Activations
fn main():
  let a = glorot(4, 4)
  let b = glorot(4, 4)
  let c = a @ b
  print("Matrix A @ Matrix B:")
  print(c)
  let activated = relu(c)
  print("ReLU(A @ B):")
  print(activated)`,

  temporal: `// 3. Temporal Safety — Lookahead Leak Prevention
// Notice: prices.after(1) tries to peek +1 step into the future!
fn predict_price(prices: Temporal[Tensor[10, 1], -1]) -> Tensor[10, 1]:
  let prev_price = prices.before(1)
  
  // COMPILE ERROR: reading future timestamps (+1) is rejected at compile time!
  let future_leak: Temporal[Tensor[10, 1], 0] = prices.after(1)
  return future_leak.snapshot()

fn main():
  let history: Temporal[Tensor[10, 1], -1] = zeros(10, 1) + 100.0
  let p = predict_price(history)
  print(p)`,

  causal: `// 4. Causal Safety — Conflating Correlation with Causation
// Observational data cannot be passed to functions requiring causal interventions!
fn should_prescribe(effect: Causal[Float, intervened]) -> Float:
  return effect.extract()

fn main():
  // Observational evidence: correlation observed in clinical records
  let obs: Causal[Float, observed] = 0.85

  // TYPE ERROR: Cannot prescribe medication based solely on observational data!
  let decision = should_prescribe(obs)
  print(decision)`,

  uncertainty: `// 5. Medical Safety — Uncertain[T] with Confidence Bounds
model DosagePredictor:
  w1: Tensor[4, 8] = glorot(4, 8)
  w_dose: Tensor[8, 1] = glorot(8, 1)
  w_conf: Tensor[8, 1] = glorot(8, 1)

  fn predict(self, vitals: Tensor[1, 4]) -> Uncertain[Tensor[1, 1]]:
    let h = relu(vitals @ self.w1)
    let dose = h @ self.w_dose
    let confidence = sigmoid(h @ self.w_conf)
    return Uncertain(dose, confidence)

fn main():
  let predictor = DosagePredictor()
  let vitals = zeros(1, 4) + 0.5
  let result = predictor.predict(vitals)

  print("Estimated Dosage (mg):")
  print(result.value)
  print("Model Confidence Interval:")
  print(result.confidence)`,

  forgetting: `// 6. Provable Machine Unlearning with Fisher Scrubbing
model PatientNet:
  w: Tensor[4, 1] = glorot(4, 1)

fn main():
  let net = PatientNet()
  let sensitive_data = zeros(1, 4) + 0.9

  print("Applying Fisher Information Noise Scrubbing to model...")
  let cert = forget(net, sensitive_data, method="FisherScrubbing", strength=0.1)
  print("Unlearning verified: Certificate signed in-memory.")`
};

const snippetDescriptions = {
  autograd: "Differentiable neural network training in WebAssembly with autograd tape, MSE loss, and Adam optimizer.",
  matmul: "Sub-millisecond matrix multiplication and vectorized ReLU activation running on browser SIMD.",
  temporal: "Compile-time prevention of lookahead bias in time-series data. Click 'Type Check' to see it caught!",
  causal: "Prevents confusing observational correlation with interventional causation. Click 'Type Check' to verify!",
  uncertainty: "Tracks uncertainty distributions and confidence intervals across medical dosage predictions.",
  forgetting: "Provable machine unlearning in-place via Fisher Information Noise Scrubbing with verified bounds."
};

// ══════════════════════════════════════════════════
// 2. Application Logic
// ══════════════════════════════════════════════════
let currentTab = "autograd";
let wasmReady = false;

function initApp() {
  const tabContainer = document.getElementById("editor-tabs");
  const codeEditor = document.getElementById("code-editor");
  const terminalBody = document.getElementById("terminal-body");
  const snippetDesc = document.getElementById("snippet-desc");
  const runBtn = document.getElementById("run-btn");
  const typecheckBtn = document.getElementById("typecheck-btn");
  const transpileBtn = document.getElementById("transpile-btn");
  const copyBtn = document.getElementById("copy-btn");
  const resetBtn = document.getElementById("reset-btn");
  const wasmStatus = document.getElementById("wasm-status");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  function updateEditor() {
    if (codeEditor) {
      codeEditor.value = codeSnippets[currentTab] || "";
    }
    if (snippetDesc) {
      snippetDesc.textContent = snippetDescriptions[currentTab] || "";
    }
  }

  // ══════════════════════════════════════════════════
  // ANSI Color Formatter
  // ══════════════════════════════════════════════════
  function ansiToHtml(text) {
    if (!text) return "";
    let escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    escaped = escaped
      .replace(/\u001b\[1;31m/g, '<span style="color:var(--rose);font-weight:600">')
      .replace(/\u001b\[31m/g, '<span style="color:var(--rose)">')
      .replace(/\u001b\[1;32m/g, '<span style="color:var(--emerald);font-weight:600">')
      .replace(/\u001b\[32m/g, '<span style="color:var(--emerald)">')
      .replace(/\u001b\[1;33m/g, '<span style="color:var(--amber);font-weight:600">')
      .replace(/\u001b\[33m/g, '<span style="color:var(--amber)">')
      .replace(/\u001b\[1;34m/g, '<span style="color:var(--violet-light);font-weight:600">')
      .replace(/\u001b\[34m/g, '<span style="color:var(--violet-light)">')
      .replace(/\u001b\[1;36m/g, '<span style="color:#38bdf8;font-weight:600">')
      .replace(/\u001b\[36m/g, '<span style="color:#38bdf8">')
      .replace(/\u001b\[1m/g, '<span style="font-weight:bold;color:var(--text-primary)">')
      .replace(/\u001b\[0m/g, '</span>')
      .replace(/\u001b\[[0-9;]*m/g, '');

    return escaped;
  }

  function clearTerminal() {
    if (terminalBody) terminalBody.innerHTML = "";
  }

  function appendPrompt(cmd) {
    if (!terminalBody) return;
    const div = document.createElement("div");
    div.className = "term-line";
    div.innerHTML = `<span class="term-prompt">visitor@neuron:~$</span> ${cmd}`;
    terminalBody.appendChild(div);
  }

  function appendLine(html, type = "") {
    if (!terminalBody) return;
    const div = document.createElement("div");
    div.className = "term-line";
    if (type === "error") div.className += " term-error";
    else if (type === "warning") div.className += " term-warning";
    else if (type === "success") div.className += " term-success";
    div.innerHTML = html;
    terminalBody.appendChild(div);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  function renderWasmResult(jsonStr, command) {
    clearTerminal();
    appendPrompt(command);

    try {
      const res = JSON.parse(jsonStr);

      if (res.warnings && res.warnings.length > 0) {
        res.warnings.forEach(w => {
          appendLine(ansiToHtml(w), "warning");
        });
      }

      if (!res.success || (res.errors && res.errors.length > 0)) {
        if (res.errors && res.errors.length > 0) {
          res.errors.forEach(e => {
            appendLine(ansiToHtml(e), "error");
          });
        }
        if (res.output && res.output !== "Type check failed.") {
          appendLine(ansiToHtml(res.output));
        }
        appendLine("Compilation / Verification completed with diagnostics above.", "error");
      } else {
        if (res.output) {
          appendLine(ansiToHtml(res.output));
        }
        appendLine("✓ Succeeded (0 errors)", "success");
      }
    } catch (e) {
      appendLine(ansiToHtml(jsonStr));
    }
  }

  // ══════════════════════════════════════════════════
  // WASM Loader
  // ══════════════════════════════════════════════════
  async function loadWasm() {
    try {
      const t0 = performance.now();
      await init('neuron_wasm_bg.wasm');
      const elapsed = Math.round(performance.now() - t0);

      wasmReady = true;
      if (wasmStatus) {
        wasmStatus.textContent = `WASM ready (${elapsed}ms) ✓`;
        wasmStatus.style.color = "var(--emerald)";
      }

      if (runBtn) runBtn.disabled = false;
      if (typecheckBtn) typecheckBtn.disabled = false;
      if (transpileBtn) transpileBtn.disabled = false;
    } catch (err) {
      console.warn("Direct WASM init fallback:", err);
      try {
        await init();
        wasmReady = true;
        if (wasmStatus) {
          wasmStatus.textContent = "WASM ready ✓";
          wasmStatus.style.color = "var(--emerald)";
        }
        if (runBtn) runBtn.disabled = false;
        if (typecheckBtn) typecheckBtn.disabled = false;
        if (transpileBtn) transpileBtn.disabled = false;
      } catch (e2) {
        console.error("WASM load failed:", e2);
        if (wasmStatus) {
          wasmStatus.textContent = "WASM unavailable";
          wasmStatus.style.color = "var(--rose)";
        }
      }
    }
  }

  // ══════════════════════════════════════════════════
  // Event Listeners
  // ══════════════════════════════════════════════════
  if (tabContainer) {
    tabContainer.addEventListener("click", (e) => {
      const button = e.target.closest(".tab-btn");
      if (!button) return;

      document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
      button.classList.add("active");

      currentTab = button.dataset.tab;
      updateEditor();

      clearTerminal();
      appendPrompt(`neuronc check examples/${currentTab}.nr`);
      appendLine(`Loaded "${button.textContent.trim()}" example.`);
      appendLine(`Click "Run" to execute in WASM, "Type Check" to verify types, or "→ Python" to transpile.`);
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      if (!codeEditor) return;
      navigator.clipboard.writeText(codeEditor.value).then(() => {
        const span = copyBtn.querySelector("span");
        if (span) {
          span.textContent = "Copied!";
          setTimeout(() => { span.textContent = "Copy"; }, 2000);
        }
      }).catch(err => console.error("Clipboard copy failed:", err));
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      updateEditor();
      clearTerminal();
      appendPrompt(`git checkout examples/${currentTab}.nr`);
      appendLine(`Reset snippet to original code.`);
    });
  }

  if (typecheckBtn) {
    typecheckBtn.addEventListener("click", () => {
      if (!wasmReady || !codeEditor) return;
      const src = codeEditor.value;
      const cmd = `neuronc check ${currentTab}.nr`;
      try {
        const res = type_check(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Type check error: ${err.message || err}`, "error");
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", () => {
      if (!wasmReady || !codeEditor) return;
      const src = codeEditor.value;
      const cmd = `neuronc run ${currentTab}.nr`;
      try {
        const res = eval_neuron(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Runtime error: ${err.message || err}`, "error");
      }
    });
  }

  if (transpileBtn) {
    transpileBtn.addEventListener("click", () => {
      if (!wasmReady || !codeEditor) return;
      const src = codeEditor.value;
      const cmd = `neuronc transpile --target python ${currentTab}.nr`;
      try {
        const res = transpile_to_python(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Transpiler error: ${err.message || err}`, "error");
      }
    });
  }

  // Mobile nav
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("active");
    });
  }

  // Smooth scroll
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === "#") return;
      e.preventDefault();
      const targetElement = document.querySelector(href);
      if (targetElement) {
        if (navLinks && navLinks.classList.contains("active")) {
          navLinks.classList.remove("active");
        }
        targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Reveal animations
  const revealElements = document.querySelectorAll(".reveal");
  if ('IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
    revealElements.forEach(el => revealObserver.observe(el));
  } else {
    revealElements.forEach(el => el.classList.add("revealed"));
  }

  // Card cursor gradients
  const cards = document.querySelectorAll(".feature-card, .pricing-card, .benchmark-card, .cta-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
      card.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
    });
  });

  // Stats counter
  const statsElements = document.querySelectorAll(".hero-stat-value");
  if (statsElements.length > 0 && 'IntersectionObserver' in window) {
    const statsObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const stat = entry.target;
          const target = parseInt(stat.getAttribute("data-target"));
          if (isNaN(target)) return;
          if (target === 0) { stat.textContent = "0"; observer.unobserve(stat); return; }
          let current = 0;
          const duration = 1500, steps = 50;
          const stepTime = duration / steps, increment = target / steps;
          let step = 0;
          const timer = setInterval(() => {
            current += increment;
            step++;
            if (step >= steps) {
              clearInterval(timer);
              stat.textContent = target >= 100000 ? "100k+" : Math.round(target).toString();
            } else {
              stat.textContent = target >= 100000 ? Math.round(current / 1000) + "k+" : Math.round(current).toString();
            }
          }, stepTime);
          observer.unobserve(stat);
        }
      });
    }, { threshold: 0.2 });
    statsElements.forEach(el => statsObserver.observe(el));
  } else {
    statsElements.forEach(el => {
      const target = el.getAttribute("data-target");
      el.textContent = target === "100000" ? "100k+" : target;
    });
  }

  // FAQs
  document.querySelectorAll(".faq-question").forEach(btn => {
    btn.addEventListener("click", () => {
      const item = btn.closest(".faq-item");
      const isActive = item.classList.contains("active");
      document.querySelectorAll(".faq-item").forEach(i => i.classList.remove("active"));
      if (!isActive) item.classList.add("active");
    });
  });

  // Initialize
  updateEditor();
  loadWasm();
}

// Ensure initApp runs regardless of whether DOMContentLoaded has already fired
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
