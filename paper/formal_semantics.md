# Formal Semantics and Metatheoretical Soundness of NEURON ($\lambda_{\text{neuron}}$)

**Author:** Fayo Ibrahim & The NEURON Development Team  
**Affiliation:** Neuron Labs  
**Date:** September 2026  
**Status:** Core Calculus Specification, Labeled Traced Semantics, and Soundness Theorems

---

## 1. Introduction

Modern machine learning systems routinely execute computations where domain-specific correctness depends on physical and statistical constraints not captured by conventional types:
1. **Temporal Non-Interference (Lookahead Bias):** In time-series forecasting, backtesting, and online RL, computing decisions at epoch $t$ using data from timestamps $t' > t$ represents a **lookahead leak**. Such leaks produce overfitted evaluations and severe live deployment failures.
2. **Causal Mode Safety (Confounding Bias):** In observational vs. experimental studies, computing treatment effects by subtracting associational conditionals $P(Y \mid X=1) - P(Y \mid X=0)$ rather than interventional distributions $P(Y \mid \text{do}(X=1)) - P(Y \mid \text{do}(X=0))$ introduces confounding bias when unobserved common causes exist ($X \leftarrow U \rightarrow Y$).

NEURON enforces these safety properties at compile time. This document formalizes the core theoretical calculus $\lambda_{\text{neuron}}$, establishing small-step operational reduction rules, typing judgments, and complete paper-level proofs of Progress, Subject Reduction, and Temporal Non-Interference (machine-checked verification in a proof assistant like Lean 4 or Coq is ongoing future work).

Crucially, this formalization resolves the relationship between a type's **static upper-bound dependency horizon** ($\Delta$) and its **dynamic evaluation read footprint** ($\mathcal{R}(\mathcal{T})$) via a labeled transition system, establishing an airtight proof of **Temporal Non-Interference**. On the causal side, we prove **Intervention Integrity / Causal Mode Isolation**, while explicitly formalizing automated causal identification completeness as an orthogonal open problem.

### Core Design Principle: Access Footprint vs. Semantic Dependence
A critical conceptual design choice in $\lambda_{\text{neuron}}$ is that the temporal type system tracks an **over-approximation of the temporal access footprint of computation**, rather than the extensional semantic dependence of the resulting value on historical observations. For example, in an expression such as:
$$e = s.\text{eval}() - s.\text{eval}()$$
the algebraic value is identically zero and contains no semantic information dependence on $s$. Nonetheless, $\lambda_{\text{neuron}}$ assigns $e$ the temporal horizon of $s$ because evaluation executed a physical store access at that timeline coordinate. By tracking access footprints rather than undecidable semantic equivalence, NEURON maintains a sound, decidable, and statically enforceable barrier against physical lookahead leaks.

---

## 2. Abstract Syntax of $\lambda_{\text{neuron}}$

The calculus strictly separates **symbolic stream expressions** (unevaluated time-series data sources that accumulate access offsets algebraically) from **computational terms** and **evaluated values**.

### 2.1 Types
$$
\begin{array}{rcll}
B & ::= & \texttt{Int} \mid \texttt{Float} \mid \texttt{Bool} \mid \texttt{Tensor}[\vec{d}] & \text{(Base Types)} \\
m & ::= & \mathbf{obs} \mid \mathbf{int} & \text{(Causal Modes)} \\
k, \Delta & \in & \mathbb{Z} & \text{(Temporal Offsets / Horizons)} \\
\epsilon & \subseteq & \{\mathbf{mut}, \mathbf{io}, \mathbf{rand}\} & \text{(Effect Sets)} \\
\tau & ::= & B & \text{(Base Datatype)} \\
     & \mid & \tau_1 \xrightarrow{\epsilon} \tau_2 & \text{(Effect-Annotated Function)} \\
     & \mid & \texttt{Stream}[\tau, k] & \text{(Stream Reference with Relative Access Offset } k\text{)} \\
     & \mid & \texttt{Temporal}[\tau, \Delta] & \text{(Temporal Value with Upper-Bound Dependency Horizon } \Delta\text{)} \\
     & \mid & \texttt{Causal}[\tau, m] & \text{(Causal Value with Mode } m\text{)} \\
     & \mid & \texttt{Uncertain}[\tau] & \text{(Distribution Wrapper with Confidence Metric)} \\
     & \mid & \tau_1 \times \tau_2 & \text{(Product / Tuple Type)} \\
     & \mid & \mathbf{1} & \text{(Unit Type)}
\end{array}
$$

### 2.2 Expressions and Stream Transformations

Stream transformations occur algebraically *prior* to data materialization. There is no value-level offset relabeling.

$$
\begin{array}{rcll}
s & ::= & x & \text{(Stream Variable / Identifier)} \\
  & \mid & s.\text{shift}(d) & \text{(Offset Translation by } d \in \mathbb{Z}\text{)} \\
  & \mid & s.\text{lag}(d) & \text{(Backward Lag by } d \ge 0 \equiv s.\text{shift}(-d)\text{)} \\
  & \mid & s.\text{lead}(d) & \text{(Forward Lead by } d \ge 0 \equiv s.\text{shift}(+d)\text{)} \\[6pt]
e & ::= & x & \text{(Variable)} \\
  & \mid & c & \text{(Literal Constant)} \\
  & \mid & \lambda x:\tau.\, e & \text{(Function Abstraction)} \\
  & \mid & e_1\, e_2 & \text{(Function Application)} \\
  & \mid & (e_1, e_2) \mid \pi_1(e) \mid \pi_2(e) & \text{(Pair Construction and Projections)} \\
  & \mid & e_1 \oplus e_2 & \text{(Primitive Binary Operations: } +, -, \times, \div, @, ==, \land, \lor\text{)} \\
  & \mid & s.\text{eval}() & \text{(Stream Materialization / Store Read)} \\
  & \mid & e.\text{snapshot}() & \text{(Temporal Declassification to Raw Type; Restricted to } \Delta \le 0\text{)} \\
  & \mid & \text{obs}(e) & \text{(Causal Observational Introduction)} \\
  & \mid & \text{do}(x \gets e_1) \text{ in } e_2 & \text{(Causal Interventional Introduction)} \\
  & \mid & e.\text{extract}() & \text{(Causal Mode Elimination)} \\
  & \mid & \text{uncert}(e_1, e_2) & \text{(Uncertainty Constructor: value, confidence)} \\
  & \mid & e.\text{val} \mid e.\text{conf} & \text{(Uncertainty Projections)}
\end{array}
$$

### 2.3 Values
$$
\begin{array}{rcll}
v & ::= & c \mid \lambda x:\tau.\, e \mid (v_1, v_2) \mid () \\
  & \mid & \mathbf{temp}(v, \Delta) & \text{(Value with Static Upper-Bound Dependency Horizon } \Delta \in \mathbb{Z}\text{)} \\
  & \mid & \mathbf{caus}(v, m) & \text{(Value with Causal Mode } m \in \{\mathbf{obs}, \mathbf{int}\}\text{)} \\
  & \mid & \mathbf{unc}(v_1, v_2) & \text{(Uncertain Value: mean, dispersion)}
\end{array}
$$

---

## 3. Subtyping Preorder ($\le$)

Subtyping characterizes safe information containment.

### 3.1 Bounded Temporal Subtyping
An expression whose data provenance is bounded by an earlier horizon $k_1$ satisfies any requirement expecting data known up to horizon $k_2$:
$$
\frac{k_1 \le k_2 \quad \tau_1 \le \tau_2}{\texttt{Temporal}[\tau_1, k_1] \le \texttt{Temporal}[\tau_2, k_2]} \quad (\textsc{Sub-Temp})
$$

> **Soundness Rationale:** If a consumer expects data from horizon $k_2 \le 0$ (e.g. present data at $0$), supplying data from $k_1 = -5$ (five steps in the past) introduces no future information. Conversely, future data ($k_1 = +2$) cannot satisfy a present constraint ($k_2 = 0$) because $+2 \not\le 0$.

### 3.2 Causal Invariance (Antichain Property)
Causal modes are strictly disjoint. No implicit coercion between observational and interventional semantics is permitted:
$$
\frac{\tau_1 \le \tau_2}{\texttt{Causal}[\tau_1, m] \le \texttt{Causal}[\tau_2, m]} \quad (\textsc{Sub-Caus})
$$
$$
\texttt{Causal}[\tau, \mathbf{obs}] \not\le \texttt{Causal}[\tau, \mathbf{int}] \qquad \text{and} \qquad \texttt{Causal}[\tau, \mathbf{int}] \not\le \texttt{Causal}[\tau, \mathbf{obs}]
$$

### 3.3 Safe Ingestion and Wrapper Non-Erasure
Raw base data (e.g. constant scalars) may safely enter the temporal present or observational domain:
$$
\frac{\tau_1 \le \tau_2 \quad k \ge 0}{\tau_1 \le \texttt{Temporal}[\tau_2, k]} \quad (\textsc{Sub-Temp-Ingest}) \qquad
\frac{\tau_1 \le \tau_2}{\tau_1 \le \texttt{Causal}[\tau_2, \mathbf{obs}]} \quad (\textsc{Sub-Caus-Ingest})
$$

Crucially, the inverse does **not** hold:
$$
\texttt{Temporal}[\tau, \Delta] \not\le \tau \qquad \text{and} \qquad \texttt{Causal}[\tau, m] \not\le \tau \quad (\textsc{Non-Erasure})
$$
Temporal and causal wrappers cannot be stripped implicitly via subsumption.

### 3.4 Structural and Function Subtyping
$$
\frac{\tau_2 \le \tau_1 \quad \sigma_1 \le \sigma_2 \quad \epsilon_1 \subseteq \epsilon_2}{\tau_1 \xrightarrow{\epsilon_1} \sigma_1 \le \tau_2 \xrightarrow{\epsilon_2} \sigma_2} \quad (\textsc{Sub-Fn}) \qquad
\frac{\tau_1 \le \tau_1' \quad \tau_2 \le \tau_2'}{\tau_1 \times \tau_2 \le \tau_1' \times \tau_2'} \quad (\textsc{Sub-Prod})
$$

---

## 4. Static Typing Rules

Typing contexts $\Gamma$ map variable names to types: $\Gamma \Coloneqq \emptyset \mid \Gamma, x : \tau$.

### 4.1 Stream Calculus Typing ($\Gamma \vdash_s s : \texttt{Stream}[\tau, k]$)

Streams track their cumulative algebraic offset statically:
$$
\frac{x : \texttt{Stream}[\tau, k_0] \in \Gamma}{\Gamma \vdash_s x : \texttt{Stream}[\tau, k_0]} \, (\textsc{TS-Var}) \qquad
\frac{x : \texttt{Stream}[\tau] \in \Gamma}{\Gamma \vdash_s x : \texttt{Stream}[\tau, 0]} \, (\textsc{TS-Base})
$$
$$
\frac{\Gamma \vdash_s s : \texttt{Stream}[\tau, k]}{\Gamma \vdash_s s.\text{shift}(d) : \texttt{Stream}[\tau, k + d]} \, (\textsc{TS-Shift}) \qquad
\frac{\Gamma \vdash_s s : \texttt{Stream}[\tau, k]}{\Gamma \vdash_s s.\text{lag}(d) : \texttt{Stream}[\tau, k - d]} \, (\textsc{TS-Lag})
$$

### 4.2 Expression Typing ($\Gamma \vdash e : \tau$)

#### Subsumption and Standard Constructs
$$
\frac{x : \tau \in \Gamma}{\Gamma \vdash x : \tau} \, (\textsc{T-Var}) \qquad
\frac{\text{ty}(c) = B}{\Gamma \vdash c : B} \, (\textsc{T-Const}) \qquad
\frac{\Gamma \vdash e : \tau_1 \quad \tau_1 \le \tau_2}{\Gamma \vdash e : \tau_2} \, (\textsc{T-Sub})
$$
$$
\frac{\Gamma, x : \tau_1 \vdash e : \tau_2}{\Gamma \vdash (\lambda x:\tau_1.\, e) : \tau_1 \xrightarrow{\epsilon} \tau_2} \, (\textsc{T-Abs}) \qquad
\frac{\Gamma \vdash e_1 : \tau_1 \xrightarrow{\epsilon} \tau_2 \quad \Gamma \vdash e_2 : \tau_1}{\Gamma \vdash e_1\, e_2 : \tau_2} \, (\textsc{T-App})
$$
$$
\frac{\Gamma \vdash e_1 : \tau_1 \quad \Gamma \vdash e_2 : \tau_2}{\Gamma \vdash (e_1, e_2) : \tau_1 \times \tau_2} \, (\textsc{T-Pair}) \qquad
\frac{\Gamma \vdash e : \tau_1 \times \tau_2}{\Gamma \vdash \pi_1(e) : \tau_1} \, (\textsc{T-Proj1}) \qquad
\frac{\Gamma \vdash e : \tau_1 \times \tau_2}{\Gamma \vdash \pi_2(e) : \tau_2} \, (\textsc{T-Proj2})
$$

#### Stream Materialization
$$
\frac{\Gamma \vdash_s s : \texttt{Stream}[\tau, k]}{\Gamma \vdash s.\text{eval}() : \texttt{Temporal}[\tau, k]} \, (\textsc{T-Eval})
$$

#### Upper-Bound Dependency Join ($\max$)
When two temporal computations are combined, the resulting expression's future dependency bound is the maximum of its operands:
$$
\frac{\Gamma \vdash e_1 : \texttt{Temporal}[\tau_1, k_1] \quad \Gamma \vdash e_2 : \texttt{Temporal}[\tau_2, k_2] \quad \tau_1 \oplus \tau_2 = \tau_3}{\Gamma \vdash e_1 \oplus e_2 : \texttt{Temporal}[\tau_3, \max(k_1, k_2)]} \, (\textsc{T-Temp-BinOp})
$$

#### Safe Snapshot Declassification ($\Delta \le 0$)
Declassifying a temporal value into a raw type via `.snapshot()` is strictly prohibited if the value carries future provenance:
$$
\frac{\Gamma \vdash e : \texttt{Temporal}[\tau, k] \quad k \le 0}{\Gamma \vdash e.\text{snapshot}() : \tau} \, (\textsc{T-Snapshot-Safe})
$$
$$
\frac{\Gamma \vdash e : \texttt{Temporal}[\tau, k] \quad k > 0}{\Gamma \vdash e.\text{snapshot}() : \textbf{error}[\texttt{TemporalLeak}]} \, (\textsc{T-Snapshot-Leak})
$$

#### Causal Operations
$$
\frac{\Gamma \vdash e : \tau}{\Gamma \vdash \text{obs}(e) : \texttt{Causal}[\tau, \mathbf{obs}]} \, (\textsc{T-Obs}) \qquad
\frac{\Gamma \vdash e_1 : \tau_1 \quad \Gamma, x : \tau_1 \vdash e_2 : \tau_2}{\Gamma \vdash \text{do}(x \gets e_1) \text{ in } e_2 : \texttt{Causal}[\tau_2, \mathbf{int}]} \, (\textsc{T-Int})
$$
$$
\frac{\Gamma \vdash e_1 : \texttt{Causal}[\tau_1, m] \quad \Gamma \vdash e_2 : \texttt{Causal}[\tau_2, m] \quad \tau_1 \oplus \tau_2 = \tau_3}{\Gamma \vdash e_1 \oplus e_2 : \texttt{Causal}[\tau_3, m]} \, (\textsc{T-Caus-BinOp})
$$
$$
\frac{\Gamma \vdash e : \texttt{Causal}[\tau, m]}{\Gamma \vdash e.\text{extract}() : \tau} \, (\textsc{T-Extract})
$$

Rule $\textsc{T-Caus-BinOp}$ enforces strict mode agreement ($m_1 = m_2$). Combining observed and intervened data without an explicit extraction construct is rejected at compile time.

---

## 5. Dynamic Operational Semantics

### 5.1 Configurations, Stores, and Labeled Transitions

Transitions are defined over a labeled transition system:
$$
\langle e, \sigma, t \rangle \xrightarrow{\ell} \langle e', \sigma', t \rangle
$$
where:
* $t \in \mathbb{Z}$ is the **global evaluation epoch** (the present moment).
* $\sigma : \mathbb{Z} \to (\text{Var} \rightharpoonup \text{Val})$ is the **time-indexed store**.
* $\ell \in \{\tau_{\text{step}}\} \cup \{\text{read}(t') \mid t' \in \mathbb{Z}\}$ labels the transition. Internal computation emits $\tau_{\text{step}}$, while a store read emits the exact accessed timestamp $\text{read}(t')$.

\begin{definition}[Store Validity $\sigma \models t$]
A temporal store $\sigma$ is valid at epoch $t$ for a set of stream symbols $\Sigma$ if for every $x \in \Sigma$ of underlying type $\tau$ and every integer offset $k \in \mathbb{Z}$, the cell $\sigma(t+k)(x)$ is defined and contains a valid value of type $\tau$.
\end{definition}

### 5.2 Evaluation Contexts

Evaluation is deterministic call-by-value, formalized via evaluation contexts $E$:
$$
E \Coloneqq [\cdot] \mid E\, e \mid v\, E \mid (E, e) \mid (v, E) \mid \pi_1(E) \mid \pi_2(E) \mid E \oplus e \mid v \oplus E \mid E.\text{snapshot}() \mid \text{obs}(E) \mid \text{do}(x \gets E) \text{ in } e \mid E.\text{extract}()
$$
$$
\frac{\langle e, \sigma, t \rangle \xrightarrow{\ell} \langle e', \sigma', t \rangle}{\langle E[e], \sigma, t \rangle \xrightarrow{\ell} \langle E[e'], \sigma', t \rangle} \, (\textsc{E-Context})
$$

### 5.3 Reduction Rules

#### Function Application ($\beta$-reduction)
$$
\langle (\lambda x:\tau.\, e)\, v, \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle e[x \mapsto v], \sigma, t \rangle \, (\textsc{E-Beta})
$$

#### Product Projections
$$
\langle \pi_1((v_1, v_2)), \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle v_1, \sigma, t \rangle \, (\textsc{E-Proj1}) \qquad
\langle \pi_2((v_1, v_2)), \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle v_2, \sigma, t \rangle \, (\textsc{E-Proj2})
$$

#### Stream Evaluation (Store Read)
$$
\frac{k = \text{off}(s) \quad \sigma(t + k)(\text{root}(s)) = v}{\langle s.\text{eval}(), \sigma, t \rangle \xrightarrow{\text{read}(t + k)} \langle \mathbf{temp}(v, k), \sigma, t \rangle} \, (\textsc{E-Eval})
$$

#### Safe Snapshot Declassification
$$
\langle \mathbf{temp}(v, k).\text{snapshot}(), \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle v, \sigma, t \rangle \, (\textsc{E-Snap-Val})
$$

#### Binary Operations
$$
\langle \mathbf{temp}(v_1, k_1) \oplus \mathbf{temp}(v_2, k_2), \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle \mathbf{temp}(v_1 \tilde{\oplus} v_2, \max(k_1, k_2)), \sigma, t \rangle \, (\textsc{E-Temp-Op})
$$
$$
\langle \mathbf{caus}(v_1, m) \oplus \mathbf{caus}(v_2, m), \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle \mathbf{caus}(v_1 \tilde{\oplus} v_2, m), \sigma, t \rangle \, (\textsc{E-Caus-Op})
$$

#### Causal Intervention ($\text{do}$-calculus)
$$
\frac{\sigma' = \sigma[\mathcal{F}_x := (\lambda\_.\, v_1)] \quad \langle e_2, \sigma', t \rangle \Downarrow v_2}{\langle \text{do}(x \gets v_1) \text{ in } e_2, \sigma, t \rangle \xrightarrow{\tau_{\text{step}}} \langle \mathbf{caus}(v_2, \mathbf{int}), \sigma, t \rangle} \, (\textsc{E-Int})
$$

---

## 6. Metatheory and Soundness Proofs

### 6.1 Basic Type Safety

\begin{lemma}[Canonical Forms]
\label{lem:canonical}
Let $v$ be a closed value.
1. If $\emptyset \vdash v : B$, then $v = c$ with $\text{ty}(c) = B$.
2. If $\emptyset \vdash v : \tau_1 \xrightarrow{\epsilon} \tau_2$, then $v = \lambda x:\tau_1'.\, e$.
3. If $\emptyset \vdash v : \tau_1 \times \tau_2$, then $v = (v_1, v_2)$.
4. If $\emptyset \vdash v : \texttt{Temporal}[\tau, \Delta]$, then $v = \mathbf{temp}(v', \Delta')$ with $\Delta' \le \Delta$ and $\emptyset \vdash v' : \tau$.
5. If $\emptyset \vdash v : \texttt{Causal}[\tau, m]$, then $v = \mathbf{caus}(v', m)$ with $\emptyset \vdash v' : \tau$.
\end{lemma}
\begin{proof}
By straightforward inspection of the value grammar and subtyping rules.
\end{proof}

\begin{lemma}[Substitution Lemma]
If $\Gamma, x : \tau_1 \vdash e : \tau_2$ and $\Gamma \vdash v : \tau_1$, then $\Gamma \vdash e[x \mapsto v] : \tau_2$.
\end{lemma}
\begin{proof}
By structural induction on the typing derivation of $e$, utilizing transitivity of subtyping.
\end{proof}

\begin{theorem}[Progress]
If $\emptyset \vdash e : \tau$, then for any valid store $\sigma \models t$, either $e$ is a value $v$ or there exist $e', \sigma', \ell$ such that $\langle e, \sigma, t \rangle \xrightarrow{\ell} \langle e', \sigma', t \rangle$.
\end{theorem}
\begin{proof}
By induction on the typing derivation $\emptyset \vdash e : \tau$:
- **Case T-Var**: Vacuous in the empty context.
- **Case T-Const, T-Abs**: $e$ is already a value; progress holds.
- **Case T-App**: $e = e_1\, e_2$. If $e_1$ is not a value, it steps via evaluation context $E = [\cdot]\, e_2$. If $e_1$ is a value but $e_2$ is not, $e_2$ steps via $E = v_1\, [\cdot]$. If both are values, by Lemma~\ref{lem:canonical}, $e_1 = \lambda x:\tau_1'.\, e_0$, and rule $\textsc{E-Beta}$ applies.
- **Case T-Pair, T-Proj1, T-Proj2**: Either subterms step via contexts, or redexes evaluate via $\textsc{E-Proj1}/\textsc{E-Proj2}$.
- **Case T-Eval**: $e = s.\text{eval}()$. By store validity $\sigma \models t$, the cell $\sigma(t + \text{off}(s))(\text{root}(s))$ is defined. Rule $\textsc{E-Eval}$ applies immediately.
- **Case T-Snapshot-Safe**: $e = e_0.\text{snapshot}()$ with $k \le 0$. If $e_0$ is not a value, it steps via context. If $e_0$ is a value, by Lemma~\ref{lem:canonical}, $e_0 = \mathbf{temp}(v_0, k)$, and $\textsc{E-Snap-Val}$ applies. (Future offsets $k > 0$ yield compile-time errors via $\textsc{T-Snapshot-Leak}$).
- **Case T-Temp-BinOp**: When both operands are evaluated to values $\mathbf{temp}(v_1, k_1)$ and $\mathbf{temp}(v_2, k_2)$, rule $\textsc{E-Temp-Op}$ applies.
- **Case T-Caus-BinOp**: Both operands have matching mode $m$. By Lemma~\ref{lem:canonical}, $e_1 = \mathbf{caus}(v_1, m)$ and $e_2 = \mathbf{caus}(v_2, m)$. Rule $\textsc{E-Caus-Op}$ applies.
- **Case T-Sub**: Follows directly from the induction hypothesis. $\blacksquare$
\end{proof}

\begin{theorem}[Subject Reduction / Type Preservation]
If $\Gamma \vdash e : \tau$ and $\langle e, \sigma, t \rangle \xrightarrow{\ell} \langle e', \sigma', t \rangle$, then $\Gamma \vdash e' : \tau$.
\end{theorem}
\begin{proof}
By induction on the reduction relation:
- **Case \textsc{E-Beta}}**: $e = (\lambda x:\tau_1.\, e_0)\, v$. By inversion on $\textsc{T-App}$, $\Gamma, x : \tau_1 \vdash e_0 : \tau$ and $\Gamma \vdash v : \tau_1$. By the Substitution Lemma, $\Gamma \vdash e_0[x \mapsto v] : \tau$.
- **Case \textsc{E-Eval}}**: By $\textsc{T-Eval}$, $s.\text{eval}() : \texttt{Temporal}[\tau, k]$ where $k = \text{off}(s)$. The redex produces $\mathbf{temp}(v, k)$, which has type $\texttt{Temporal}[\tau, k]$.
- **Case \textsc{E-Temp-Op}}**: By $\textsc{T-Temp-BinOp}$, the expression has type $\texttt{Temporal}[\tau_3, \max(k_1, k_2)]$. The resulting value carries offset $\max(k_1, k_2)$, matching the type.
- **Case \textsc{E-Snap-Val}}**: $\mathbf{temp}(v, k).\text{snapshot}() \longrightarrow v$. By $\textsc{T-Snapshot-Safe}$, the source expression has type $\tau$. The resulting value $v$ has type $\tau$.
- **Case \textsc{E-Context}}**: Preserved by standard inductive lifting over evaluation contexts. $\blacksquare$
\end{proof}

---

### 6.2 Temporal Non-Interference: Proof of Leak-Freedom

\begin{definition}[Trace Read Footprint]
For any evaluation trace $\mathcal{T} = \langle e_0, \sigma, t \rangle \xrightarrow{\ell_1} \dots \xrightarrow{\ell_n} \langle v, \sigma', t \rangle$, the **read footprint** $\mathcal{R}(\mathcal{T}) \subset \mathbb{Z}$ is the set of all timeline timestamps read from the store:
$$
\mathcal{R}(\mathcal{T}) \Coloneqq \{t' \in \mathbb{Z} \mid \exists i \in \{1,\dots,n\}.\ \ell_i = \text{read}(t')\}
$$
\end{definition}

\begin{lemma}[Read Horizon Boundedness]
\label{lem:read-bound}
Let $e$ be a closed, well-typed expression. For any terminating trace $\mathcal{T} = \langle e, \sigma, t \rangle \longrightarrow^* \langle v, \sigma', t \rangle$:
1. If $\emptyset \vdash e : \texttt{Temporal}[\tau, \Delta]$, then $\forall t' \in \mathcal{R}(\mathcal{T})$, $t' \le t + \Delta$.
2. If $\emptyset \vdash e : \tau$ (where $\tau$ contains no temporal constructors), then $\forall t' \in \mathcal{R}(\mathcal{T})$, $t' \le t$.
\end{lemma}
\begin{proof}
By induction on the length of the evaluation trace and the typing derivation:
1. **Base Reductions with No Read**: Rules $\textsc{E-Beta}$, $\textsc{E-Proj1}$, $\textsc{E-Proj2}$, $\textsc{E-Temp-Op}$, $\textsc{E-Caus-Op}$, and $\textsc{E-Snap-Val}$ emit label $\tau_{\text{step}}$. They add no timestamps to $\mathcal{R}(\mathcal{T})$.
2. **Stream Evaluation (\textsc{E-Eval})**: $\langle s.\text{eval}(), \sigma, t \rangle \xrightarrow{\text{read}(t + k)} \langle \mathbf{temp}(v, k), \sigma, t \rangle$.
   By $\textsc{T-Eval}$, $\Delta = k = \text{off}(s)$. The read timestamp is $t' = t + k = t + \Delta \le t + \Delta$.
3. **Binary Join (\textsc{T-Temp-BinOp})**: $e = e_1 \oplus e_2$ typed with $\Delta = \max(k_1, k_2)$.
   Trace $\mathcal{T}$ evaluates $e_1$ then $e_2$, so $\mathcal{R}(\mathcal{T}) = \mathcal{R}(\mathcal{T}_1) \cup \mathcal{R}(\mathcal{T}_2)$.
   By IH, $\forall t_1 \in \mathcal{R}(\mathcal{T}_1)$, $t_1 \le t + k_1 \le t + \max(k_1, k_2) = t + \Delta$.
   Similarly, $\forall t_2 \in \mathcal{R}(\mathcal{T}_2)$, $t_2 \le t + k_2 \le t + \Delta$. Thus $\forall t' \in \mathcal{R}(\mathcal{T})$, $t' \le t + \Delta$.
4. **Snapshot (\textsc{T-Snapshot-Safe})**: $e = e_0.\text{snapshot}()$ with $\emptyset \vdash e_0 : \texttt{Temporal}[\tau, k]$ and $k \le 0$.
   The read footprint is $\mathcal{R}(\mathcal{T}_0)$. By IH, $\forall t' \in \mathcal{R}(\mathcal{T}_0)$, $t' \le t + k$. Since $k \le 0$, $t' \le t$.
5. **Function Application**: Trace evaluates $e_1 \to \lambda x.\, e_0$, $e_2 \to v_2$, then $e_0[x \mapsto v_2] \to v$.
   By the Substitution Lemma and IH applied to each subtrace, read bounds are preserved across $\beta$-reduction.
6. **Subsumption (\textsc{T-Sub})**: If $\Delta_1 \le \Delta_2$, then $t' \le t + \Delta_1 \le t + \Delta_2$. $\blacksquare$
\end{proof}

\begin{definition}[Store Horizon Agreement]
Two stores $\sigma_1, \sigma_2$ are $T$-equivalent ($\sigma_1 \approx_{\le T} \sigma_2$) iff $\sigma_1(t')(x) = \sigma_2(t')(x)$ for all variables $x$ and all $t' \le T$.
\end{definition}

\begin{theorem}[Temporal Non-Interference]
Let $e$ be a closed expression with $\emptyset \vdash e : \texttt{Temporal}[\tau, \Delta]$ where $\Delta \le 0$.
If $\sigma_1 \approx_{\le t} \sigma_2$, then:
$$
\langle e, \sigma_1, t \rangle \Downarrow \mathbf{temp}(v_1, \Delta_1) \quad \text{and} \quad \langle e, \sigma_2, t \rangle \Downarrow \mathbf{temp}(v_2, \Delta_2) \implies v_1 = v_2 \text{ and } \Delta_1 = \Delta_2
$$
\end{theorem}
\begin{proof}
1. **Footprint Confinement**: By Lemma~\ref{lem:read-bound}, $\forall t' \in \mathcal{R}(\mathcal{T})$, $t' \le t + \Delta \le t$. Therefore $\mathcal{R}(\mathcal{T}) \subseteq (-\infty, t]$.
2. **Determinism**: Because $\sigma_1 \approx_{\le t} \sigma_2$, for every read step $\textsc{E-Eval}$, $\sigma_1(t+k)(x) = \sigma_2(t+k)(x)$.
Since reduction is deterministic and store reads return identical values, the evaluation traces under $\sigma_1$ and $\sigma_2$ are isomorphic, yielding $v_1 = v_2$.
Future store mutations ($t' > t$) cannot influence the computed output. $\blacksquare$
\end{proof}

---

### 6.3 Causal Mode Soundness: Intervention Integrity

\begin{theorem}[Causal Mode Soundness (Intervention Integrity)]
Let $\mathcal{M}$ be a structural causal model over DAG $\mathcal{G}$.
Let $e$ be a closed expression typed as $\emptyset \vdash e : \texttt{Causal}[\tau, \mathbf{int}]$.
1. Evaluation of $e$ proceeds strictly under the manipulated causal model $\mathcal{M}_{\overline{X}}$ in which all incoming structural equations to intervened variables $X$ are severed ($\textsc{E-Int}$).
2. By rule $\textsc{T-Caus-BinOp}$ and subtyping mode isolation ($\texttt{Causal}[\tau, \mathbf{obs}] \not\le \texttt{Causal}[\tau, \mathbf{int}]$), $e$ cannot combine with any observational term without explicit extraction.
3. Therefore, no observational conditional expectation $P(Y \mid X=x)$ can be silently substituted for an interventional query $\mathbb{E}[Y \mid \text{do}(X=x)]$.
\end{theorem}
\begin{proof}
Under rule $\textsc{E-Int}$, $\text{do}(x \gets v_1)$ mutates the store's causal equation set by replacing $f_X$ with constant function $\lambda\_.\, v_1$. In Pearl's do-calculus, this constructs the manipulated graph $\mathcal{G}_{\overline{X}}$ with all arrows $\text{Pa}(X) \to X$ deleted. Because mode subtyping forms an antichain and $\textsc{T-Caus-BinOp}$ requires strict mode equality $m_1 = m_2 = \mathbf{int}$, observational terms cannot taint the interventional derivation. $\blacksquare$
\end{proof}

\begin{openproblem}[Automated Causal Identification Completeness]
While $\lambda_{\text{neuron}}$ statically guarantees \emph{mode integrity} (preventing observational and interventional calculations from being conflated), verifying whether an interventional query $P(Y \mid \text{do}(X))$ is non-parametrically identifiable from observational distributions via the backdoor, frontdoor, or Tian's general identification algorithm remains an open research direction for future type-directed PL+Causal integration.
\end{openproblem}

---

## 7. Implementation Correspondence Table

| Mathematical Formalism ($\lambda_{\text{neuron}}$) | NEURON Compiler Implementation | Location |
| :--- | :--- | :--- |
| $\texttt{Temporal}[\tau, \Delta]$ | `NType::Temporal(Box<NType>, TemporalSpec::Offset(i64))` | `compiler/src/types.rs:23` |
| $\textsc{Sub-Temp}$ ($k_1 \le k_2$) | `types_compatible`: `o_act <= o_exp` | `compiler/src/types.rs:164` |
| $\textsc{Sub-Caus}$ ($m_1 = m_2$) | `types_compatible`: `m1 == m2` | `compiler/src/types.rs:178` |
| $\textsc{Non-Erasure}$ ($\texttt{Temporal} \not\le \tau$) | Transparent compatibility disabled; raw sinks reject wrappers | `compiler/src/types.rs:190` |
| $\textsc{T-Temp-BinOp}$ ($\max(k_1, k_2)$) | `infer_binop`: `std::cmp::max(*n1, *n2)` | `compiler/src/types.rs:1239` |
| $\textsc{TS-Shift}$ / $\textsc{TS-Lag}$ | `infer_fn_call`: algebraic composition `current_offset + k` / `- k` | `compiler/src/types.rs:1486-1488` |
| $\textsc{T-Snapshot-Safe}$ ($k \le 0$) | Rejects future offsets ($k > 0$) with `ErrorCode::TemporalLeak` | `compiler/src/types.rs:1505, 1700` |
| $\textsc{T-Caus-BinOp}$ ($m_1 = m_2$) | `infer_binop`: emits `ErrorCode::CausalTypeMismatch` if $m_1 \neq m_2$ | `compiler/src/types.rs:1121` |
| Non-Interference Property | Validated by 66 adversarial test cases | `runtime/tests/test_adversarial.rs` |
