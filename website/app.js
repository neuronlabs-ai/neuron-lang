// NEURON Website — Live WASM Playground + UI Logic

document.addEventListener("DOMContentLoaded", () => {
  // ══════════════════════════════════════════════════
  // 1. Code Snippets (raw text for editable textarea)
  // ══════════════════════════════════════════════════
  const codeSnippets = {
    temporal: `fn predict_price(prices: Temporal[Tensor, past_to_future]) -> Tensor:
    let prev_price = prices.before(1) # OK: reading historical price
    let future_leak = prices.after(2)  # COMPILE ERROR: lookahead bias!
    return future_leak`,

    causal: `fn treatment_analysis(patient_data: Dataset):
    let obs = observe(patient_data, treatment=1)  # P(Y|X)
    let intervened = intervene(treatment=1)      # P(Y|do(X))
    
    # TYPE ERROR: Cannot mix conditional observations with interventions
    let effect: Causal[Intervention] = obs`,

    uncertainty: `fn autonomous_driving(lidar: Uncertain[Tensor[1, 3]]) -> Tensor:
    let distance = preprocess(lidar)
    
    # COMPILER WARNING: returning Uncertain prediction without check
    return distance`,

    autograd: `@differentiable
model LinearNet:
  w: Tensor[4, 1] = glorot(4, 1)

  fn train(self, x: Tensor[B, 4], y: Tensor[B, 1]) [Effect[Mut[self]]]:
    let loss = mse(x @ self.w, y)
    update self.w by sgd(grad(loss), lr=0.1)`,

    forgetting: `fn patient_right_to_be_forgotten(net: DiagnosisModel, data: Tensor) [Effect[Mut[net]]]:
    # Selective unlearning using Fisher Information Noise Scrubbing
    let cert = forget(net, data, method="FisherScrubbing", strength=0.1)
    return cert`
  };

  // ══════════════════════════════════════════════════
  // 2. State
  // ══════════════════════════════════════════════════
  let currentTab = "temporal";
  let wasmReady = false;
  let wasmModule = null;

  // ══════════════════════════════════════════════════
  // 3. DOM Elements
  // ══════════════════════════════════════════════════
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
  // 4. WASM Engine Initialization
  // ══════════════════════════════════════════════════
  async function initWasm() {
    try {
      // Dynamic import of the WASM JS bindings (ES module)
      // For non-module scripts, we fetch and instantiate manually
      const wasmUrl = "neuron_wasm_bg.wasm";
      const jsUrl = "neuron_wasm.js";

      // Fetch the JS glue code as text and evaluate it
      const jsResponse = await fetch(jsUrl);
      if (!jsResponse.ok) throw new Error(`Failed to load ${jsUrl}: ${jsResponse.status}`);
      const jsCode = await jsResponse.text();

      // Create a blob URL to import as ES module
      const blob = new Blob([jsCode], { type: "application/javascript" });
      const blobUrl = URL.createObjectURL(blob);

      try {
        wasmModule = await import(/* webpackIgnore: true */ blobUrl);
        // Initialize the WASM module
        await wasmModule.default(wasmUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }

      wasmReady = true;
      wasmStatus.textContent = "WASM engine ready ✓";
      wasmStatus.style.color = "var(--emerald)";

      // Enable buttons
      runBtn.disabled = false;
      typecheckBtn.disabled = false;
      transpileBtn.disabled = false;

      terminalBody.innerHTML = `
        <div class="term-line"><span class="term-success">✓ NEURON WASM compiler loaded successfully</span></div>
        <div class="term-line" style="color:var(--text-muted)">Ready. Select an example or write your own code, then click an action.</div>
      `;
    } catch (err) {
      console.error("WASM init failed:", err);
      wasmStatus.textContent = "WASM failed — using simulation";
      wasmStatus.style.color = "var(--amber)";

      // Enable buttons with fallback simulation mode
      runBtn.disabled = false;
      typecheckBtn.disabled = false;
      transpileBtn.disabled = false;

      terminalBody.innerHTML = `
        <div class="term-line"><span class="term-warning">⚠ WASM engine failed to load: ${err.message}</span></div>
        <div class="term-line" style="color:var(--text-muted)">Falling back to simulated output. Try using a modern browser with WASM support.</div>
      `;
    }
  }

  // ══════════════════════════════════════════════════
  // 5. Simulated Fallback Outputs
  // ══════════════════════════════════════════════════
  const simulatedOutputs = {
    typecheck: {
      temporal: `[ERROR] Line 3: TemporalLeak detected.
  --> examples/temporal_leak.nr:3:21
   |
 3 |     let future_leak = prices.after(2)
   |                       ^^^^^^^^^^^^^^^ Lookahead violation: reading future timestamps.
   |
Compilation failed: 1 temporal type violation found.`,
      causal: `[ERROR] Line 6: CausalTypeMismatch
  --> examples/causal_engine.nr:6:40
   |
 6 |     let effect: Causal[Intervention] = obs
   |                                        ^^^ expected Causal[Intervention], found Causal[Observation]
   |
Compilation failed: 1 causal type violation found.`,
      uncertainty: `[WARNING] Line 5: UncheckedUncertainty
  --> examples/lidar_test.nr:5:12
   |
 5 |     return distance
   |            ^^^^^^^^ returning Uncertain value without explicit confidence threshold check.
   |
Compilation succeeded with 1 warning.`,
      autograd: `Type checking passed.
All tensor shapes verified: Tensor[B, 4] @ Tensor[4, 1] → Tensor[B, 1]
Effect types verified: Mut[self] declared and used correctly.
0 errors, 0 warnings.`,
      forgetting: `Type checking passed.
Effect types verified: Mut[net] declared and used correctly.
Return type ForgetCertificate inferred from forget() call.
0 errors, 0 warnings.`
    },
    run: {
      temporal: `[ERROR] Compilation aborted — 1 temporal type violation prevents execution.`,
      causal: `[ERROR] Compilation aborted — 1 causal type violation prevents execution.`,
      uncertainty: `Compilation succeeded with 1 warning. Running...

Autonomous driving inference:
  Input: lidar scan [0.85, 0.42, 0.91]
  Preprocessed distance: 2.18 ± 0.73
  ⚠ Warning: High uncertainty in prediction (σ = 0.73)
Execution complete.`,
      autograd: `Compilation succeeded. Running JIT interpreter...
Iter 000/100: Loss = 16.000 (starting weight = 5.0)
Iter 020/100: Loss = 5.7600
Iter 040/100: Loss = 2.0736
Iter 060/100: Loss = 0.7464
Iter 080/100: Loss = 0.2687
Iter 100/100: Loss = 0.0001 (weight converged to 3.0)
Execution complete. Tape reset, 0 memory leaks.`,
      forgetting: `Compilation succeeded. Running...
Training model on patient dataset for 3 propagation epochs...
  -> Epoch 1: Loss = 0.311934
  -> Epoch 2: Loss = 0.177842
  -> Epoch 3: Loss = 0.106163
Applying Fisher Information Noise Scrubbing (strength = 0.10)...
✓ Modifying 4 parameter tensors in-place. Rescrambled norms: 0.932155
<ForgetCertificate>
  certificate_id: CERT-AF3A67EA1F65D64A
  method: FisherScrubbing
  strength: 0.100000
  bounds_satisfied: true
</ForgetCertificate>
Execution complete. Certificate generated.`
    }
  };

  // ══════════════════════════════════════════════════
  // 6. Initialize Editor
  // ══════════════════════════════════════════════════
  function updateEditor() {
    codeEditor.value = codeSnippets[currentTab];
  }

  // ══════════════════════════════════════════════════
  // 7. Terminal Output Helpers
  // ══════════════════════════════════════════════════
  function clearTerminal() {
    terminalBody.innerHTML = "";
  }

  function appendTermLine(text, type) {
    const div = document.createElement("div");
    div.className = "term-line";
    if (type === "prompt") {
      div.innerHTML = `<span class="term-prompt">visitor@neuron:~$</span> ${escapeHtml(text)}`;
    } else if (type === "error") {
      div.innerHTML = `<span class="term-error">${escapeHtml(text)}</span>`;
    } else if (type === "warning") {
      div.innerHTML = `<span class="term-warning">${escapeHtml(text)}</span>`;
    } else if (type === "success") {
      div.innerHTML = `<span class="term-success">${escapeHtml(text)}</span>`;
    } else {
      div.textContent = text;
    }
    terminalBody.appendChild(div);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderOutput(rawOutput, command) {
    clearTerminal();
    appendTermLine(command, "prompt");

    const lines = rawOutput.split("\n");
    lines.forEach(line => {
      if (line.match(/^\[ERROR\]/) || line.includes("error") || line.includes("failed") || line.includes("aborted")) {
        appendTermLine(line, "error");
      } else if (line.match(/^\[WARNING\]/) || line.includes("⚠") || line.includes("warning")) {
        appendTermLine(line, "warning");
      } else if (line.includes("✓") || line.includes("succeeded") || line.includes("complete") || line.includes("passed") || line.includes("converged")) {
        appendTermLine(line, "success");
      } else {
        appendTermLine(line, "info");
      }
    });
  }

  // ══════════════════════════════════════════════════
  // 8. Tab Switching
  // ══════════════════════════════════════════════════
  if (tabContainer) {
    tabContainer.addEventListener("click", (e) => {
      const button = e.target.closest(".tab-btn");
      if (!button) return;

      document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
      button.classList.add("active");

      currentTab = button.dataset.tab;
      updateEditor();
    });
  }

  // ══════════════════════════════════════════════════
  // 9. Copy Button
  // ══════════════════════════════════════════════════
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const rawText = codeEditor.value;
      navigator.clipboard.writeText(rawText).then(() => {
        const span = copyBtn.querySelector("span");
        span.textContent = "Copied!";
        setTimeout(() => { span.textContent = "Copy"; }, 2000);
      }).catch(err => console.error("Clipboard copy failed:", err));
    });
  }

  // ══════════════════════════════════════════════════
  // 10. Action Buttons — Type Check
  // ══════════════════════════════════════════════════
  if (typecheckBtn) {
    typecheckBtn.addEventListener("click", () => {
      const source = codeEditor.value;
      const command = "neuronc check playground.nr";

      if (wasmReady && wasmModule) {
        try {
          const result = wasmModule.type_check(source);
          renderOutput(result, command);
        } catch (err) {
          clearTerminal();
          appendTermLine(command, "prompt");
          appendTermLine(`Internal error: ${err.message || err}`, "error");
        }
      } else {
        // Fallback: use simulated output
        const output = simulatedOutputs.typecheck[currentTab] || "Type checking completed.";
        renderOutput(output, command);
      }
    });
  }

  // ══════════════════════════════════════════════════
  // 11. Action Buttons — Run
  // ══════════════════════════════════════════════════
  if (runBtn) {
    runBtn.addEventListener("click", () => {
      const source = codeEditor.value;
      const command = "neuronc run playground.nr";

      if (wasmReady && wasmModule) {
        try {
          const result = wasmModule.eval_neuron(source);
          renderOutput(result, command);
        } catch (err) {
          clearTerminal();
          appendTermLine(command, "prompt");
          appendTermLine(`Internal error: ${err.message || err}`, "error");
        }
      } else {
        // Fallback: use simulated output
        const output = simulatedOutputs.run[currentTab] || "Execution completed.";
        renderOutput(output, command);
      }
    });
  }

  // ══════════════════════════════════════════════════
  // 12. Action Buttons — Transpile to Python
  // ══════════════════════════════════════════════════
  if (transpileBtn) {
    transpileBtn.addEventListener("click", () => {
      const source = codeEditor.value;
      const command = "neuronc transpile --target python playground.nr";

      if (wasmReady && wasmModule) {
        try {
          const result = wasmModule.transpile_to_python(source);
          renderOutput(result, command);
        } catch (err) {
          clearTerminal();
          appendTermLine(command, "prompt");
          appendTermLine(`Internal error: ${err.message || err}`, "error");
        }
      } else {
        clearTerminal();
        appendTermLine(command, "prompt");
        appendTermLine("Transpilation requires the WASM engine. Please try in a browser with WebAssembly support.", "warning");
      }
    });
  }

  // ══════════════════════════════════════════════════
  // 13. Mobile Navbar Toggle
  // ══════════════════════════════════════════════════
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("active");
    });
  }

  // ══════════════════════════════════════════════════
  // 14. Smooth Scroll for Anchor Links
  // ══════════════════════════════════════════════════
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

  // ══════════════════════════════════════════════════
  // 15. Reveal Animations on Scroll
  // ══════════════════════════════════════════════════
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

  // ══════════════════════════════════════════════════
  // 16. Premium Cursor-Tracking Hover Effect
  // ══════════════════════════════════════════════════
  const cards = document.querySelectorAll(".feature-card, .pricing-card, .benchmark-card, .cta-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
      card.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
    });
  });

  // ══════════════════════════════════════════════════
  // 17. Stats Count-Up Animation
  // ══════════════════════════════════════════════════
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

  // ══════════════════════════════════════════════════
  // 18. FAQ Accordion
  // ══════════════════════════════════════════════════
  document.querySelectorAll(".faq-question").forEach(btn => {
    btn.addEventListener("click", () => {
      const item = btn.closest(".faq-item");
      const isActive = item.classList.contains("active");
      // Close all
      document.querySelectorAll(".faq-item").forEach(i => i.classList.remove("active"));
      // Toggle clicked
      if (!isActive) item.classList.add("active");
    });
  });

  // ══════════════════════════════════════════════════
  // 19. CSS Spin keyframes injection
  // ══════════════════════════════════════════════════
  const style = document.createElement('style');
  style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
  document.head.appendChild(style);

  // ══════════════════════════════════════════════════
  // 20. Initialize
  // ══════════════════════════════════════════════════
  updateEditor();
  initWasm();
});
