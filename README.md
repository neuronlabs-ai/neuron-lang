<p align="center">
  <h1 align="center">◈ NEURON</h1>
  <p align="center"><strong>The world's first AI-native programming language.</strong></p>
  <p align="center">
    A statically typed, natively differentiable programming language with compile-time temporal safety checks,<br>
    causal inference types, uncertainty tracking, and empirical machine unlearning certificates.
  </p>
</p>

> [!NOTE]
> **Status**: Research prototype in active development. NEURON enforces type-level compile-time safety checks for temporal offsets, causal intervention modes, and effect boundaries. See the technical paper for formal boundary definitions and scope.

<p align="center">
  <a href="#why-neuron">Why NEURON?</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#language-tour">Language Tour</a> •
  <a href="#pycheck">PyCheck Linter</a> •
  <a href="#execution-backends">Backends</a> •
  <a href="#examples">Examples</a>
</p>

---

## Why NEURON?

Existing ML frameworks bolt differentiation and safety onto Python as runtime libraries. When things break — whether it's a silent dimension mismatch, a $100M lookahead bias leak, or an inability to legally prove model unlearning — they fail silently in production.

**NEURON makes differentiation, temporal safety, causality, and forgetting first-class citizens of the compiler and type system.**

```python
# NEURON — A self-attention transformer in pure, differentiable syntax
model Attention(d_model: Int, head_dim: Int):
  wq: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wk: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wv: Tensor[d_model, head_dim] = glorot(d_model, head_dim)
  wo: Tensor[head_dim, d_model] = glorot(head_dim, d_model)

  fn forward(self, x: Tensor) -> Tensor:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = q @ transpose(k, 0, 1)
    let attn = softmax(scores * (1.0 / sqrt(16.0)))
    return (attn @ v) @ self.wo

fn main() [Effect[Mut[model]]]:
  let model = Attention(64, 16)
  let x = randn(1, 64)
  let target = randn(1, 64)
  
  let loss = mse(model.forward(x), target)
  update model by adam(grad(loss), lr=0.001)
```

**No `torch.nn.Module`. No `@tf.function`. No `.backward()` boilerplate.** Just clean mathematical expressions verified at compile time.

---

## What Makes NEURON Different

| Feature | Python / PyTorch / JAX | NEURON |
|---|---|---|
| **Gradients** | Library runtime call (`.backward()`) | **Built into type system** — every `fn` is differentiable |
| **Temporal Data Leakage** | Silent runtime failure (fake 99% accuracy) | **Type-level compile-time checks** (`error[TemporalLeak]`) & offset bounds |
| **Machine Unlearning** | Full retraining required ($$$) | **Built-in `forget()` primitive** with empirical `ForgetCertificate` |
| **Causality** | Separate libraries (DoWhy) | **First-class causal types** (`observe`, `intervene`) |
| **Uncertainty** | Manual propagation | **Type-tracked** (`Uncertain[Tensor]`, confidence warnings) |
| **Execution Targets** | Python runtime only | **Interpreter, Rust JIT, Native AOT Binary, WASM, Python** |

---

## Quickstart

### 1. Build from Source (Rust 1.70+)

```bash
git clone https://github.com/neuronlabs-ai/neuron-lang
cd neuron-lang
cargo build --release
```

The `neuronc` compiler binary will be at `target/release/neuronc`.

### 2. Run Interactive REPL Terminal

```bash
neuronc repl
```

```
  ╔══════════════════════════════════════════════════╗
  ║   ◈ NEURON  Interactive Terminal                 ║
  ║   v1.0.0 — The AI-Native Programming Language    ║
  ╚══════════════════════════════════════════════════╝

  nr │ randn(3, 3)
  Tensor[3, 3]

  nr │ softmax(randn(1, 5))
  [0.0462, 0.5159, 0.0513, 0.1799, 0.2066]

  nr │ :load examples/transformer.nr
  ✓ Loaded and executed examples/transformer.nr
```

### 3. Launch Desktop Terminal / Web IDE

NEURON comes with a standalone Web IDE powered by client-side WebAssembly:

```bash
python run_desktop.py
```
Opens `http://localhost:8080/desktop/` with syntax highlighting, live execution, and built-in examples.

---

## Language Tour

### 1. Compile-Time Temporal Safety (Lookahead Bias Prevention)

In time-series forecasting and sequence modeling, lookahead bias is fatal. NEURON's type checker verifies temporal data flow at compile time:

```python
temporal fn predict_signal(data: Temporal[Tensor, past_to_future]) -> Tensor:
  return sigmoid(data @ glorot(10, 1))

fn main():
  let future_prices = Temporal[randn(100, 10), future_to_past]
  return predict_signal(future_prices)  # COMPILE ERROR: lookahead bias detected
```

```text
error[TemporalLeak]: Temporal direction violation: data flows future_to_past
but context expects past_to_future — lookahead bias detected
  --> demo_million_dollar_bug.nr:23:10
23 |   return predict_signal(future_prices)
             ^^^^^^^^^^^^^^
  expected: past_to_future
  got:      future_to_past
  help: Use .before(t) to restrict temporal data to the past
```

#### Offset-Based Temporal Algebra & Multi-Horizon Alignment (§3.1.1)

NEURON also tracks exact signed integer step horizons with compile-time algebraic composition:

```python
fn safe_strategy(x: Temporal[Tensor, 0]) -> Tensor:
  return x

fn main():
  let prices: Temporal[Tensor, 0] = randn(20, 4)
  
  // Lookback 5 periods, then forward shift by 2 -> (-5) + 2 = -3 (Safe past data)
  let signal = prices.shift(-5).shift(2)
  safe_strategy(signal) # PASS: -3 <= 0 (Zero lookahead)

  // Unsafe forward shift: (-5) + 8 = +3 (Leaking 3 steps into future)
  let bad = signal.shift(6)
  safe_strategy(bad)    # COMPILE ERROR: Temporal offset violation (+3 > 0)
```

---

### 2. Machine Unlearning (`forget()`) with Verifiable Certificates

Erase specific training data representations from model weights on command:

```python
// Erase copyrighted or sensitive dataset using Fisher Scrubbing or Gradient Ascent
let cert = forget(model, sensitive_data, "FisherScrubbing", 0.5)
print(cert)
```

**Output on a 120,832-parameter Transformer:**
```text
<ForgetCertificate>
  certificate_id:          CERT-75A0D2BFBE344562
  method:                  FisherScrubbing
  strength:                0.500000
  params_modified:         120832
  task_alignment_before:   0.991036
  task_alignment_after:    0.990978
  forgotten_loss_before:   0.008964
  forgotten_loss_after:    0.009022
  bounds_satisfied:        true
  forgetting_successful:   true
```

---

### 3. Causal Inference (Observation vs. Intervention)

```python
causal fn estimate_ate(x: Causal[Tensor]) -> Tensor:
  let obs = observe(x, condition=1.0)
  let act = intervene(x, do_value=1.0)  // do-calculus
  return act - obs
```

---

### 4. Uncertainty Propagation

```python
fn prescribe_dosage(patient_weight: Uncertain[Float]) -> Uncertain[Float]:
  let dose = patient_weight * 0.12  // Uncertainty automatically propagates
  return dose
```

---

## PyCheck — Python ML Safety Analyzer

NEURON ships with **PyCheck** (`pycheck-neuron` on PyPI), an AST static analyzer and 2-pass taint engine that scans **existing Python ML & quant code** for temporal leaks, causal confusion, and uncertainty bugs:

```bash
pip install pycheck-neuron
pycheck examples/leaky_transformer_pipeline.py
```

```text
=================================================================
  PyCheck — NEURON ML Safety Analyzer
  Scanning: leaky_transformer_pipeline.py
  Rules: 30 active
=================================================================

  ERROR[T001]: .shift(-1) accesses data 1 rows INTO THE FUTURE
  ERROR[T014]: .diff(-5) computes difference using future data
  ERROR[T010]: .fit_transform() on full dataset leaks test statistics into training
  ERROR[T012]: KFold() shuffles time-series data across folds, leaking future into training

  Summary: 4 error(s), 0 warning(s)
  These 4 error(s) would be COMPILE-TIME ERRORS in NEURON
```

- **30 Specialized Rules**: 15 Temporal (`T001`–`T015`), 7 Causal (`C001`–`C007`), 6 Uncertainty (`U001`–`U006`), 2 Data Quality (`D002`–`D003`).
- **Data Flow Taint Engine**: Traces how future data propagates through variable assignments into `.fit()` sinks.
- **VS Code Extension**: Real-time inline red/yellow squiggles on temporal leaks as you write Python.

---

## Execution Backends

NEURON compiles a single Intermediate Representation (IR) to 5 backends:

```bash
neuronc run       file.nr   # 1. Stack-based VM Interpreter with autograd tape
neuronc jit       file.nr   # 2. Native Rust JIT compiler (cdylib dynamic loading)
neuronc aot       file.nr   # 3. Ahead-Of-Time standalone native binary (.exe)
neuronc transpile file.nr   # 4. PyTorch Python transpiler
wasm-pack build   wasm/     # 5. In-browser WebAssembly target
```

---

## 6-Pass Compiler Optimizer

NEURON includes an enterprise-grade IR optimization pipeline:

1. **Constant Folding**: Evaluates constant tensor and scalar sub-trees at compile time.
2. **Algebraic Simplification**: Identity elimination ($X + 0 \to X$, $X \times 1 \to X$, $X \times 0 \to 0$).
3. **Common Subexpression Elimination (CSE)**: Eliminates redundant matrix multiplications and tensor passes.
4. **Dead Code Elimination (DCE)**: Prunes unreferenced IR nodes and side-effect-free instructions.
5. **Loop-Invariant Code Motion (LICM)**: Hoists static loop expressions outside loop bodies.
6. **Tensor Operator Fusion**: Fuses element-wise chains (e.g. `relu(matmul(x, w) + b)`) to minimize memory allocation.

---

## Standard Library

```
stdlib/
├── nn.nr            # Linear, LayerNorm, RMSNorm, MultiHeadAttention, Transformer
├── optim.nr         # Adam, SGD, AdamW, learning rate schedulers
├── causal.nr        # Causal graphs, observe, intervene, counterfactuals
├── distributions.nr # Gaussian, Bernoulli, Dirichlet, Categorical sampling
├── data.nr          # DataLoader, Dataset, sequence batching
├── rl.nr            # PPO, DQN, GAE, replay buffers
└── agi.nr           # Working memory, episodic memory, planning primitives
```

---

## Verification & Testing

NEURON is backed by a rigorous multi-tier test suite:

```bash
# Run all end-to-end example verifications
powershell -ExecutionPolicy Bypass -File .\test_all.ps1

# Run core Rust test suite
cargo test

# Run PyCheck test suite
cd pycheck && pytest
```

| Test Suite | Total Tests | Pass Rate |
|---|---|---|
| **Core Rust Test Suite** (Compiler, Runtime, WASM) | 126 tests | **100% PASS** |
| **PyCheck Linter Test Suite** | 73 unit tests | **100% PASS** |
| **End-to-End Example Suite** (`test_all.ps1`) | 19 verifications | **100% PASS** |
| **Endurance Suite** | 100,000 iterations | **0 memory leaks, 0 NaN** |
| **Property Tests (VM == JIT parity)** | 100 random programs | **100% Parity** |

---

## Tools & Integrations

- **`tools/py2nr.py`**: Python → NEURON transpiler converting PyTorch/numpy scripts into `.nr` source.
- **`editors/vscode`**: Language Server Protocol (LSP) client for `.nr` syntax, diagnostics, and hover types.
- **`pycheck/pycheck-vscode`**: VS Code extension for Python ML safety diagnostics.
- **`desktop/`**: Standalone browser IDE with WebAssembly runtime.

---

## License & Commercial Inquiries

NEURON is licensed under the [Community and Commercial Enterprise License (BSL 1.1)](LICENSE).
- **Free for non-commercial research, education, and evaluation.**
- **Production use, live algorithmic trading, and commercial cloud hosting require an enterprise commercial license.**

For commercial enterprise licensing, hedge fund execution licenses, compliance audit certificates, or custom AOT hardware backends, contact: **`licensing@neuron-lab.org`** | **`sales@neuron-lab.org`**.

---

<p align="center">
  <strong>NEURON</strong> — <em>Because the language you think in shapes the intelligence you create.</em>
</p>
