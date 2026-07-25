// NEURON Website Simulation Logic

document.addEventListener("DOMContentLoaded", () => {
  // 1. Code Snippets with Pre-Highlighted Syntax
  const codeSnippets = {
    temporal: `fn predict_price(prices: <span class="hl-type">Temporal</span>[Tensor, past_to_future]) -> <span class="hl-type">Tensor</span>:
    <span class="hl-keyword">let</span> prev_price = prices.before(<span class="hl-number">1</span>) <span class="hl-comment"># OK: reading historical price</span>
    <span class="hl-keyword">let</span> future_leak = prices.after(<span class="hl-number">2</span>)  <span class="hl-comment"># COMPILE ERROR: lookahead bias!</span>
    <span class="hl-keyword">return</span> future_leak`,

    causal: `<span class="hl-keyword">fn</span> treatment_analysis(patient_data: <span class="hl-type">Dataset</span>):
    <span class="hl-keyword">let</span> obs = observe(patient_data, treatment=<span class="hl-number">1</span>)  <span class="hl-comment"># P(Y|X)</span>
    <span class="hl-keyword">let</span> intervened = intervene(treatment=<span class="hl-number">1</span>)      <span class="hl-comment"># P(Y|do(X))</span>
    
    <span class="hl-comment"># TYPE ERROR: Cannot mix conditional observations with interventions</span>
    <span class="hl-keyword">let</span> effect: <span class="hl-type">Causal</span>[Intervention] = obs`,

    uncertainty: `<span class="hl-keyword">fn</span> autonomous_driving(lidar: <span class="hl-type">Uncertain</span>[Tensor[1, 3]]) -> <span class="hl-type">Tensor</span>:
    <span class="hl-keyword">let</span> distance = preprocess(lidar)
    
    <span class="hl-comment"># COMPILER WARNING: returning Uncertain prediction without check</span>
    <span class="hl-keyword">return</span> distance`,

    autograd: `<span class="hl-decorator">@differentiable</span>
<span class="hl-keyword">model</span> LinearNet:
  w: <span class="hl-type">Tensor</span>[4, 1] = glorot(<span class="hl-number">4</span>, <span class="hl-number">1</span>)

  <span class="hl-keyword">fn</span> train(self, x: <span class="hl-type">Tensor</span>[B, 4], y: <span class="hl-type">Tensor</span>[B, 1]) [<span class="hl-type">Effect</span>[Mut[self]]]:
    <span class="hl-keyword">let</span> loss = mse(x @ self.w, y)
    <span class="hl-keyword">update</span> self.w <span class="hl-keyword">by</span> sgd(grad(loss), lr=<span class="hl-number">0.1</span>)`,

    forgetting: `<span class="hl-keyword">fn</span> patient_right_to_be_forgotten(net: <span class="hl-type">DiagnosisModel</span>, data: <span class="hl-type">Tensor</span>) [<span class="hl-type">Effect</span>[Mut[net]]]:
    <span class="hl-comment"># Selective unlearning using Fisher Information Noise Scrubbing</span>
    <span class="hl-keyword">let</span> cert = forget(net, data, method=<span class="hl-string">"FisherScrubbing"</span>, strength=<span class="hl-number">0.1</span>)
    <span class="hl-keyword">return</span> cert`,

    wasm: `<span class="hl-comment"># WebAssembly (WASM) & Multi-GPU Distributed Training</span>
<span class="hl-keyword">fn</span> main() -> <span class="hl-type">Tensor</span>[2, 4]:
    <span class="hl-keyword">let</span> cluster = distribute(devices=[<span class="hl-number">0</span>, <span class="hl-number">1</span>, <span class="hl-number">2</span>, <span class="hl-number">3</span>]) <span class="hl-comment"># Ring-AllReduce</span>
    <span class="hl-keyword">let</span> w = glorot(<span class="hl-number">2</span>, <span class="hl-number">4</span>)
    <span class="hl-keyword">return</span> relu(w)`
  };

  // 1b. Raw Code Snippets for Copy-to-Clipboard
  const rawSnippets = {
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
    return cert`,

    wasm: `# WebAssembly (WASM) & Multi-GPU Distributed Training
fn main() -> Tensor[2, 4]:
    let cluster = distribute(devices=[0, 1, 2, 3]) # Ring-AllReduce
    let w = glorot(2, 4)
    return relu(w)`
  };

  // 2. Line counts for each snippet
  const lineCounts = {
    temporal: 4,
    causal: 6,
    uncertainty: 5,
    autograd: 8,
    forgetting: 4,
    wasm: 5
  };

  // 3. Simulated Compiler Outputs
  const compilerLogs = {
    temporal: [
      { text: "visitor@neuron:~$ neuronc check examples/temporal_leak.nr", type: "prompt" },
      { text: "Analyzing temporal data dependencies...", type: "info" },
      { text: "[ERROR] Line 3: TemporalLeak detected.", type: "error" },
      { text: "  --> examples/temporal_leak.nr:3:21", type: "info" },
      { text: "   |", type: "info" },
      { text: " 3 |     let future_leak = prices.after(2)", type: "info" },
      { text: "   |                       ^^^^^^^^^^^^^^^ Lookahead violation: reading future timestamps.", type: "error" },
      { text: "   |", type: "info" },
      { text: "Compilation failed: 1 temporal type violation found.", type: "error" }
    ],
    causal: [
      { text: "visitor@neuron:~$ neuronc check examples/causal_engine.nr", type: "prompt" },
      { text: "Type-checking structural causal model variables...", type: "info" },
      { text: "[ERROR] Line 6: CausalTypeMismatch", type: "error" },
      { text: "  --> examples/causal_engine.nr:6:40", type: "info" },
      { text: "   |", type: "info" },
      { text: " 6 |     let effect: Causal[Intervention] = obs", type: "info" },
      { text: "   |                                        ^^^ expected Causal[Intervention], found Causal[Observation]", type: "error" },
      { text: "   |", type: "info" },
      { text: "Compilation failed: 1 causal type violation found.", type: "error" }
    ],
    uncertainty: [
      { text: "visitor@neuron:~$ neuronc check examples/lidar_test.nr", type: "prompt" },
      { text: "Analyzing uncertainty propagation pathways...", type: "info" },
      { text: "[WARNING] Line 5: UncheckedUncertainty", type: "warning" },
      { text: "  --> examples/lidar_test.nr:5:12", type: "info" },
      { text: "   |", type: "info" },
      { text: " 5 |     return distance", type: "info" },
      { text: "   |            ^^^^^^^^ returning Uncertain value without explicit confidence threshold check.", type: "warning" },
      { text: "   |", type: "info" },
      { text: "Compilation succeeded with 1 warning.", type: "success" }
    ],
    autograd: [
      { text: "visitor@neuron:~$ neuronc run examples/linear_regression.nr", type: "prompt" },
      { text: "Initializing AD tape & allocating tensors on JIT backend...", type: "info" },
      { text: "Compilation succeeded. Running JIT interpreter...", type: "success" },
      { text: "Iter 000/100: Loss = 16.000 (starting weight = 5.0)", type: "info" },
      { text: "Iter 020/100: Loss = 5.7600", type: "info" },
      { text: "Iter 040/100: Loss = 2.0736", type: "info" },
      { text: "Iter 060/100: Loss = 0.7464", type: "info" },
      { text: "Iter 080/100: Loss = 0.2687", type: "info" },
      { text: "Iter 100/100: Loss = 0.0001 (weight converged to 3.0)", type: "success" },
      { text: "Execution complete. Tape reset, 0 memory leaks.", type: "success" }
    ],
    forgetting: [
      { text: "visitor@neuron:~$ neuronc run examples/unlearning_demo.nr", type: "prompt" },
      { text: "Resolving model dependency tree for DiagnosisModel...", type: "info" },
      { text: "Training model on patient dataset for 3 propagation epochs...", type: "info" },
      { text: "  -> Epoch 1: Loss = 0.311934", type: "info" },
      { text: "  -> Epoch 2: Loss = 0.177842", type: "info" },
      { text: "  -> Epoch 3: Loss = 0.106163", type: "info" },
      { text: "Measuring baseline model parameter norms before modification...", type: "info" },
      { text: "  -> baseline total norm: 1.158016", type: "info" },
      { text: "Computing Fisher Information diagonal expectation on target dataset...", type: "info" },
      { text: "Executing tape-based backward pass for targeted noise injection...", type: "info" },
      { text: "Applying Fisher Information Noise Scrubbing (strength = 0.50)...", type: "success" },
      { text: "✓ Modifying 4 parameter tensors in-place. Rescrambled norms: 0.932155", type: "success" },
      { text: "Evaluating residual capabilities & safety bounds: bounds satisfied.", type: "success" },
      { text: "<ForgetCertificate>", type: "info" },
      { text: "  certificate_id: CERT-AF3A67EA1F65D64A", type: "info" },
      { text: "  forgotten_loss_before: 0.469637", type: "info" },
      { text: "  forgotten_loss_after: 0.567157", type: "info" },
      { text: "  method: FisherScrubbing", type: "info" },
      { text: "  param_norm_before: 1.158016", type: "info" },
      { text: "  param_norm_after: 0.932155", type: "info" },
      { text: "  params_modified: 4", type: "info" },
      { text: "  residual_loss_retained: 0.195042", type: "info" },
      { text: "  bounds_satisfied: true", type: "info" },
      { text: "  strength: 0.500000", type: "info" },
      { text: "</ForgetCertificate>", type: "info" },
      { text: "Execution complete. Certificate generated.", type: "success" }
    ],
    wasm: [
      { text: "visitor@neuron:~$ neuronc run examples/wasm_distributed.nr", type: "prompt" },
      { text: "Compiling NEURON IR with WebAssembly (neuron-wasm) & Ring-AllReduce...", type: "info" },
      { text: "✓ WebAssembly C-ABI Target: [type_check, compile_to_ir, eval_neuron, transpile]", type: "success" },
      { text: "✓ Multi-GPU Topology initialized across 4 CUDA devices [0, 1, 2, 3]", type: "success" },
      { text: "✓ Ring-AllReduce scatter-reduce and all-gather gradient synchronization active", type: "success" },
      { text: "Tensor[2, 4]", type: "success" },
      { text: "  [[ 0.1425,  0.8912,  0.0000,  0.4120 ],", type: "info" },
      { text: "   [ 0.0000,  0.6514,  0.3129,  0.9810 ]]", type: "info" },
      { text: "✓ Executed cleanly in WebAssembly (wasm32-unknown-unknown) target.", type: "success" }
    ]
  };

  // 4. State Management
  let currentTab = "temporal";
  let isRunning = false;

  // 5. DOM Elements
  const tabContainer = document.getElementById("editor-tabs");
  const codeContainer = document.getElementById("code-container");
  const codeTextarea = document.getElementById("code-textarea");
  const lineNumbersContainer = document.getElementById("line-numbers");
  const terminalBody = document.getElementById("terminal-body");
  const runBtn = document.getElementById("run-btn");
  const copyBtn = document.getElementById("copy-btn");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  // 6. Initialize UI
  function initPlayground() {
    updateEditor();
  }

  function updateLineNumbers() {
    const val = codeTextarea ? codeTextarea.value : rawSnippets[currentTab];
    const lines = val.split('\n').length;
    let lineHtml = "";
    for (let i = 1; i <= Math.max(lines, 4); i++) {
      lineHtml += `${i}<br>`;
    }
    lineNumbersContainer.innerHTML = lineHtml;
  }

  function updateEditor() {
    if (codeTextarea) {
      codeTextarea.value = rawSnippets[currentTab];
    }
    updateLineNumbers();
  }

  if (codeTextarea) {
    codeTextarea.addEventListener("input", updateLineNumbers);
    codeTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const start = codeTextarea.selectionStart;
        const end = codeTextarea.selectionEnd;
        codeTextarea.value = codeTextarea.value.substring(0, start) + "  " + codeTextarea.value.substring(end);
        codeTextarea.selectionStart = codeTextarea.selectionEnd = start + 2;
        updateLineNumbers();
      }
    });
  }

  // 7. Tab Switching Event Listeners
  tabContainer.addEventListener("click", (e) => {
    if (isRunning) return;
    
    const button = e.target.closest(".tab-btn");
    if (!button) return;
    
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    button.classList.add("active");
    
    currentTab = button.dataset.tab;
    updateEditor();
    
    let cmd = `check examples/${currentTab}_leak.nr`;
    if (currentTab === 'causal') cmd = "check examples/causal_engine.nr";
    else if (currentTab === 'uncertainty') cmd = "check examples/lidar_test.nr";
    else if (currentTab === 'autograd') cmd = "run examples/linear_regression.nr";
    else if (currentTab === 'forgetting') cmd = "run examples/unlearning_demo.nr";
    else if (currentTab === 'wasm') cmd = "run examples/wasm_distributed.nr";

    terminalBody.innerHTML = `
      <div class="term-line"><span class="term-prompt">visitor@neuron:~$</span> neuronc ${cmd}</div>
      <div class="term-line">Ready to compile. Click "Compile & Run" above to execute.</div>
    `;
  });

  // 8. Copy Snippet Event Listener
  copyBtn.addEventListener("click", () => {
    const rawText = codeTextarea ? codeTextarea.value : rawSnippets[currentTab];
    navigator.clipboard.writeText(rawText).then(() => {
      const span = copyBtn.querySelector("span");
      span.textContent = "Copied!";
      setTimeout(() => {
        span.textContent = "Copy";
      }, 2000);
    }).catch(err => {
      console.error("Clipboard copy failed: ", err);
    });
  });

  // 9. Run Live Parser / Evaluator
  runBtn.addEventListener("click", () => {
    if (isRunning) return;
    
    isRunning = true;
    runBtn.style.opacity = "0.6";
    runBtn.innerHTML = `
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="animation: spin 1s linear infinite"><circle cx="12" cy="12" r="10"></circle><path d="M12 2v4"></path></svg>
      Compiling...
    `;
    
    terminalBody.innerHTML = "";

    const userCode = codeTextarea ? codeTextarea.value : rawSnippets[currentTab];
    let logQueue = [];

    // Live Parser & Type Checker logic based on user input
    if (userCode.includes(".after(")) {
      const lineNo = userCode.split("\n").findIndex(l => l.includes(".after(")) + 1;
      logQueue = [
        { text: `visitor@neuron:~$ neuronc check user_input.nr`, type: "prompt" },
        { text: "Analyzing temporal data dependencies...", type: "info" },
        { text: `[ERROR] Line ${lineNo}: TemporalLeak detected.`, type: "error" },
        { text: `  --> user_input.nr:${lineNo}:21`, type: "info" },
        { text: "   |", type: "info" },
        { text: ` ${lineNo} | ${userCode.split("\n")[lineNo-1] || ""}`, type: "info" },
        { text: "   |                       ^^^^^^^^^^^^^^^ Lookahead violation: reading future timestamps.", type: "error" },
        { text: "Type check failed: 1 error found.", type: "error" }
      ];
    } else if (userCode.includes("observe(") && userCode.includes("intervene(")) {
      const lineNo = userCode.split("\n").findIndex(l => l.includes("Causal")) + 1 || 4;
      logQueue = [
        { text: `visitor@neuron:~$ neuronc check user_input.nr`, type: "prompt" },
        { text: "Analyzing structural causal model graphs...", type: "info" },
        { text: `[ERROR] Line ${lineNo}: CausalTypeMismatch detected.`, type: "error" },
        { text: `  --> user_input.nr:${lineNo}:18`, type: "info" },
        { text: "   |", type: "info" },
        { text: "   | Cannot assign observational P(Y|X) to intervention target P(Y|do(X)).", type: "error" },
        { text: "Type check failed: 1 error found.", type: "error" }
      ];
    } else {
      logQueue = [
        { text: `visitor@neuron:~$ neuronc run user_input.nr`, type: "prompt" },
        { text: "Compiling NEURON IR via WebAssembly (neuron-wasm)...", type: "info" },
        { text: "✓ Type Checker: 0 errors, 0 warnings", type: "success" },
        { text: "✓ Generated IR SSA Basic Blocks (1 func, 4 ops)", type: "info" },
        { text: "Tensor[2, 4]", type: "success" },
        { text: "  [[ 0.1425,  0.8912,  0.0000,  0.4120 ],", type: "info" },
        { text: "   [ 0.0000,  0.6514,  0.3129,  0.9810 ]]", type: "info" },
        { text: "✓ WebAssembly Execution completed in 0.42 ms.", type: "success" }
      ];
    }

    let logIndex = 0;
    function printNextLine() {
      if (logIndex >= logQueue.length) {
        isRunning = false;
        runBtn.style.opacity = "1";
        runBtn.innerHTML = `
          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Compile & Run
        `;
        return;
      }
      
      const log = logQueue[logIndex];
      const div = document.createElement("div");
      div.className = "term-line";
      
      if (log.type === "prompt") {
        div.innerHTML = `<span class="term-prompt">visitor@neuron:~$</span> ${log.text.replace("visitor@neuron:~$ ", "")}`;
      } else if (log.type === "error") {
        div.innerHTML = `<span class="term-error">${log.text}</span>`;
      } else if (log.type === "warning") {
        div.innerHTML = `<span class="term-warning">${log.text}</span>`;
      } else if (log.type === "success") {
        div.innerHTML = `<span class="term-success">${log.text}</span>`;
      } else {
        div.textContent = log.text;
      }
      
      terminalBody.appendChild(div);
      terminalBody.scrollTop = terminalBody.scrollHeight;
      
      logIndex++;
      setTimeout(printNextLine, 120);
    }
    
    printNextLine();
  });

  // 10. Mobile Navbar Toggle
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
      navToggle.classList.toggle("active");
    });
  }

  // 11. FAQ Accordion Event Listeners
  const faqItems = document.querySelectorAll(".faq-item");
  faqItems.forEach(item => {
    const question = item.querySelector(".faq-question");
    question.addEventListener("click", () => {
      const isActive = item.classList.contains("active");
      
      // Close other items
      faqItems.forEach(el => el.classList.remove("active"));
      
      // Toggle clicked item
      if (!isActive) {
        item.classList.add("active");
      }
    });
  });

  // 12. SaaS Pricing Billing Toggle Switch (Removed in public launch)

  // 13. Waitlist Modal Handling
  const waitlistModal = document.getElementById("waitlist-modal");
  const openWaitlistBtn = document.getElementById("open-waitlist-btn");
  const closeWaitlistBtn = document.getElementById("close-waitlist-btn");
  const successCloseBtn = document.getElementById("success-close-btn");
  const waitlistForm = document.getElementById("waitlist-form");
  const waitlistEmailInput = document.getElementById("waitlist-email");
  const waitlistSubmitBtn = document.getElementById("waitlist-submit-btn");
  const waitlistFormContainer = document.getElementById("waitlist-form-container");
  const waitlistSuccessContainer = document.getElementById("waitlist-success-container");
  const waitlistSuccessEmail = document.getElementById("waitlist-success-email");

  function openModal() {
    if (waitlistModal) {
      waitlistFormContainer.style.display = "block";
      waitlistSuccessContainer.style.display = "none";
      waitlistEmailInput.value = "";
      waitlistModal.classList.add("active");
    }
  }

  function closeModal() {
    if (waitlistModal) {
      waitlistModal.classList.remove("active");
    }
  }

  if (openWaitlistBtn) {
    openWaitlistBtn.addEventListener("click", openModal);
  }

  if (closeWaitlistBtn) {
    closeWaitlistBtn.addEventListener("click", closeModal);
  }

  if (successCloseBtn) {
    successCloseBtn.addEventListener("click", closeModal);
  }

  if (waitlistModal) {
    waitlistModal.addEventListener("click", (e) => {
      if (e.target === waitlistModal) {
        closeModal();
      }
    });
  }

  if (waitlistForm) {
    waitlistForm.addEventListener("submit", (e) => {
      e.preventDefault();
      
      const email = waitlistEmailInput.value;
      if (!email) return;
      
      const originalBtnText = waitlistSubmitBtn.textContent;
      waitlistSubmitBtn.disabled = true;
      waitlistSubmitBtn.textContent = "Joining...";
      waitlistSubmitBtn.style.opacity = "0.7";
      
      fetch("https://formsubmit.co/ajax/contact@neuron-lab.org", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          email: email,
          _subject: "New NEURON Cloud Platform Waitlist Signup!",
          _message: `Developer email: ${email} registered for the NEURON SaaS Cloud Beta.`
        })
      })
      .then(response => response.json())
      .then(data => {
        waitlistSuccessEmail.textContent = email;
        waitlistFormContainer.style.display = "none";
        waitlistSuccessContainer.style.display = "block";
      })
      .catch(error => {
        console.error("Waitlist submission error:", error);
        waitlistSuccessEmail.textContent = email;
        waitlistFormContainer.style.display = "none";
        waitlistSuccessContainer.style.display = "block";
      })
      .finally(() => {
        waitlistSubmitBtn.disabled = false;
        waitlistSubmitBtn.textContent = originalBtnText;
        waitlistSubmitBtn.style.opacity = "1";
      });
    });
  }

  // 14. Scroll Reveal Observer
  const revealElements = document.querySelectorAll(".reveal");
  if (revealElements.length > 0 && 'IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: "0px 0px -40px 0px"
    });
    revealElements.forEach(el => revealObserver.observe(el));
  } else {
    revealElements.forEach(el => el.classList.add("revealed"));
  }

  // 15. Premium Cursor-Tracking Hover Effect
  const cards = document.querySelectorAll(".feature-card, .pricing-card, .benchmark-card, .cta-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);
    });
  });

  // 16. Stats Count-Up Animation
  const statsElements = document.querySelectorAll(".hero-stat-value");
  if (statsElements.length > 0 && 'IntersectionObserver' in window) {
    const statsObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const stat = entry.target;
          const target = parseInt(stat.getAttribute("data-target"));
          if (isNaN(target)) return;
          if (target === 0) {
            stat.textContent = "0";
            observer.unobserve(stat);
            return;
          }
          let current = 0;
          const duration = 1500;
          const steps = 50;
          const stepTime = duration / steps;
          const increment = target / steps;
          
          let step = 0;
          const timer = setInterval(() => {
            current += increment;
            step++;
            if (step >= steps) {
              clearInterval(timer);
              if (target >= 100000) {
                stat.textContent = "100k+";
              } else {
                stat.textContent = Math.round(target).toString();
              }
            } else {
              if (target >= 100000) {
                stat.textContent = Math.round(current / 1000) + "k+";
              } else {
                stat.textContent = Math.round(current).toString();
              }
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

  // Initialize playground on startup
  initPlayground();

  // CSS Spin keyframes injection for runner loading icon
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
});
