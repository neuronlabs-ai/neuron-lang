// NEURON Website — Live In-Browser WebAssembly Engine & UI Logic
import init, { type_check, eval_neuron, transpile_to_python } from './neuron_wasm.js';

document.addEventListener("DOMContentLoaded", async () => {
  // ══════════════════════════════════════════════════
  // 1. Curated Code Snippets for Interactive Playground
  // ══════════════════════════════════════════════════
  const codeSnippets = {
    temporal: `// Temporal Safety — Lookahead Leaks Detected at Compile Time
fn predict_price(prices: Temporal[Tensor[10, 1], -1]) -> Tensor[10, 1]:
  let prev_price: Temporal[Tensor[10, 1], -2] = prices.before(1)

  // COMPILE ERROR: Lookahead violation! Reading future timestamp (+1)
  let future_leak: Temporal[Tensor[10, 1], 0] = prices.after(1)
  return future_leak.snapshot()
`,

    causal: `// Causal Safety — Cannot Conflate Observations with Interventions
fn should_prescribe(effect: Causal[Float, intervened]) -> Float:
  return effect.extract()

fn main():
  // Observational evidence: correlation from historical patient data
  let obs: Causal[Float, observed] = 0.85

  // COMPILE ERROR: Cannot pass observational data where intervention is required!
  let decision = should_prescribe(obs)
  print(decision)
`,

    uncertainty: `// Uncertainty Tracking — Warns When Accessing Unguarded Predictions
fn predict_dosage(vitals: Tensor[1, 4]) -> Uncertain[Float]:
  let dose = 50.0
  let conf = 0.65
  return Uncertain(dose, conf)

fn administer(dose: Float):
  print("Administering dose:")
  print(dose)

fn main():
  let p = predict_dosage(zeros(1, 4))
  // COMPILER WARNING: accessing uncertain value without confidence check!
  administer(p)
`,

    autograd: `// Native Autograd — Differentiable ML Model Training in Browser WASM
model TinyNet:
  w: Tensor[4, 1] = glorot(4, 1)

  fn forward(self, x: Tensor[1, 4]) -> Tensor[1, 1]:
    return x @ self.w

fn main():
  let net = TinyNet()
  let x = zeros(1, 4) + 0.5
  let y = zeros(1, 1) + 1.0

  let epoch = 0
  while epoch < 25:
    let pred = net.forward(x)
    let loss = mse(pred, y)
    update net by adam(grad(loss), lr=0.05)
    let epoch = epoch + 1

  let final_pred = net.forward(x)
  print("Training 25 steps with Adam complete!")
  print("Trained prediction:")
  print(final_pred)
`,

    forgetting: `// Provable Machine Unlearning — Fisher Information Noise Scrubbing
model SafetyNet:
  w: Tensor[4, 1] = glorot(4, 1)

fn main():
  let net = SafetyNet()
  let sensitive_data = zeros(1, 4) + 0.9

  print("Executing Fisher Information Noise Scrubbing...")
  let cert = forget(net, sensitive_data, method="FisherScrubbing", strength=0.1)
  print("Unlearning verified & certificate signed.")
`
  };

  // ══════════════════════════════════════════════════
  // 2. State & Elements
  // ══════════════════════════════════════════════════
  let currentTab = "temporal";
  let wasmReady = false;

  const tabContainer = document.getElementById("editor-tabs");
  const codeEditor = document.getElementById("code-editor");
  const terminalBody = document.getElementById("terminal-body");
  const runBtn = document.getElementById("run-btn");
  const typecheckBtn = document.getElementById("typecheck-btn");
  const transpileBtn = document.getElementById("transpile-btn");
  const copyBtn = document.getElementById("copy-btn");
  const wasmStatus = document.getElementById("wasm-status");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  // ══════════════════════════════════════════════════
  // 3. ANSI Escape Code Converter to Formatted HTML
  // ══════════════════════════════════════════════════
  function ansiToHtml(text) {
    if (!text) return "";
    let escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Color mappings
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
      .replace(/\u001b\[[0-9;]*m/g, ''); // strip any remaining

    return escaped;
  }

  function clearTerminal() {
    terminalBody.innerHTML = "";
  }

  function appendPrompt(cmd) {
    const div = document.createElement("div");
    div.className = "term-line";
    div.innerHTML = `<span class="term-prompt">visitor@neuron:~$</span> ${cmd}`;
    terminalBody.appendChild(div);
  }

  function appendLine(html, type = "") {
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
        appendLine("Compilation / Execution failed.", "error");
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
  // 4. Initialize WASM Engine
  // ══════════════════════════════════════════════════
  async function loadWasmEngine() {
    try {
      const t0 = performance.now();
      await init('neuron_wasm_bg.wasm');
      const elapsed = Math.round(performance.now() - t0);

      wasmReady = true;
      wasmStatus.textContent = `WASM engine ready (${elapsed}ms) ✓`;
      wasmStatus.style.color = "var(--emerald)";

      runBtn.disabled = false;
      typecheckBtn.disabled = false;
      transpileBtn.disabled = false;

      clearTerminal();
      appendPrompt("neuronc --version");
      appendLine("NEURON Compiler v1.0.0 (WebAssembly, 100% In-Browser Runtime)");
      appendLine("Ready. Edit code or select an example above, then click an action.", "success");
    } catch (err) {
      console.warn("Direct WASM init fallback:", err);
      try {
        await init();
        wasmReady = true;
        wasmStatus.textContent = "WASM engine ready ✓";
        wasmStatus.style.color = "var(--emerald)";
        runBtn.disabled = false;
        typecheckBtn.disabled = false;
        transpileBtn.disabled = false;
      } catch (e2) {
        console.error("WASM load failed completely:", e2);
        wasmStatus.textContent = "WASM unavailable";
        wasmStatus.style.color = "var(--rose)";
        appendPrompt("neuronc check");
        appendLine(`WebAssembly failed to initialize: ${e2.message || e2}.`, "error");
      }
    }
  }

  // ══════════════════════════════════════════════════
  // 5. Playground Events & Actions
  // ══════════════════════════════════════════════════
  function updateEditor() {
    codeEditor.value = codeSnippets[currentTab] || "";
  }

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
      appendLine(`Loaded ${currentTab} example. Click Type Check, Run, or → Python.`);
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(codeEditor.value).then(() => {
        const span = copyBtn.querySelector("span");
        span.textContent = "Copied!";
        setTimeout(() => { span.textContent = "Copy"; }, 2000);
      }).catch(err => console.error("Clipboard copy failed:", err));
    });
  }

  if (typecheckBtn) {
    typecheckBtn.addEventListener("click", () => {
      if (!wasmReady) return;
      const src = codeEditor.value;
      const cmd = `neuronc check ${currentTab}.nr`;
      try {
        const res = type_check(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Type check crashed: ${err.message || err}`, "error");
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", () => {
      if (!wasmReady) return;
      const src = codeEditor.value;
      const cmd = `neuronc run ${currentTab}.nr`;
      try {
        const res = eval_neuron(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Runtime execution crashed: ${err.message || err}`, "error");
      }
    });
  }

  if (transpileBtn) {
    transpileBtn.addEventListener("click", () => {
      if (!wasmReady) return;
      const src = codeEditor.value;
      const cmd = `neuronc transpile --target python ${currentTab}.nr`;
      try {
        const res = transpile_to_python(src);
        renderWasmResult(res, cmd);
      } catch (err) {
        clearTerminal();
        appendPrompt(cmd);
        appendLine(`Transpiler crashed: ${err.message || err}`, "error");
      }
    });
  }

  // ══════════════════════════════════════════════════
  // 6. Navigation, Scroll, and Interactive Effects
  // ══════════════════════════════════════════════════
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("active");
    });
  }

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

  const cards = document.querySelectorAll(".feature-card, .pricing-card, .benchmark-card, .cta-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
      card.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
    });
  });

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

  document.querySelectorAll(".faq-question").forEach(btn => {
    btn.addEventListener("click", () => {
      const item = btn.closest(".faq-item");
      const isActive = item.classList.contains("active");
      document.querySelectorAll(".faq-item").forEach(i => i.classList.remove("active"));
      if (!isActive) item.classList.add("active");
    });
  });

  // ══════════════════════════════════════════════════
  // 7. Startup
  // ══════════════════════════════════════════════════
  updateEditor();
  loadWasmEngine();
});
