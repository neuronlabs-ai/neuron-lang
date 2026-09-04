"""
═══════════════════════════════════════════════════════════════════════
  NEURON — STAGE DEMO RUNNER
  
  The master script that runs all 5 demos in sequence.
  Open a terminal on stage, run this, and watch the room go silent.
  
  Usage:
    python demos/stage_runner.py           (run all demos)
    python demos/stage_runner.py --demo 1  (run specific demo)
═══════════════════════════════════════════════════════════════════════
"""

import subprocess
import sys
import time
import os
import shutil

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Configuration ──
NEURONC = os.path.join("target", "release", "neuronc.exe")
DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DEMOS_DIR)

# ANSI colors
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"
RESET   = "\033[0m"

def banner(title, subtitle=""):
    width = 65
    print()
    print(f"{MAGENTA}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    if subtitle:
        print(f"{DIM}  {subtitle}{RESET}")
    print(f"{MAGENTA}{'═' * width}{RESET}")
    print()

def separator():
    print(f"\n{DIM}{'─' * 65}{RESET}\n")

def pause(msg="Press ENTER to continue to the next demo..."):
    """Pause between demos so you control the pace on stage."""
    print(f"\n{YELLOW}{BOLD}  ▸ {msg}{RESET}")
    input()

def run_neuronc(cmd, filepath, expect_error=False):
    """Run neuronc with a subcommand and display results."""
    full_path = os.path.join(DEMOS_DIR, filepath)
    exe = os.path.join(ROOT_DIR, NEURONC)
    
    print(f"{DIM}  $ neuronc {cmd} {filepath}{RESET}")
    print()
    
    t0 = time.perf_counter()
    result = subprocess.run(
        [exe, cmd, full_path],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=ROOT_DIR
    )
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    
    # Print output
    output = (result.stdout or "") + (result.stderr or "")
    for line in output.strip().split("\n"):
        if "error" in line.lower() or "Error" in line:
            print(f"  {RED}{line}{RESET}")
        elif "warning" in line.lower():
            print(f"  {YELLOW}{line}{RESET}")
        elif "✓" in line or "PASS" in line.upper() or "CONFIRMED" in line:
            print(f"  {GREEN}{line}{RESET}")
        elif "✗" in line or "ANOMALY" in line or "BLOCKED" in line:
            print(f"  {RED}{BOLD}{line}{RESET}")
        elif "═" in line or "───" in line:
            print(f"  {CYAN}{line}{RESET}")
        else:
            print(f"  {line}")
    
    # Timing
    if expect_error and result.returncode != 0:
        print(f"\n  {RED}{BOLD}✗ COMPILE ERROR (as expected!) — Bug caught in {elapsed:.0f} ms{RESET}")
    elif result.returncode == 0:
        print(f"\n  {GREEN}{BOLD}✓ Completed in {elapsed:.0f} ms{RESET}")
    else:
        print(f"\n  {RED}Exit code: {result.returncode} ({elapsed:.0f} ms){RESET}")
    
    return result.returncode

def run_benchmark():
    """Run the Rust sub-millisecond benchmark test."""
    exe = "cargo"
    print(f"{DIM}  $ cargo test --release --test sub_millisecond_proof -p neuron-runtime -- --nocapture{RESET}")
    print()
    
    # Write the test file
    test_code = r'''use std::time::Instant;
use neuron_runtime::tensor::{Tensor, tensor_matmul};
use neuron_runtime::autograd::GradTape;
use neuron_runtime::vm::VM;
use neuron_compiler::compile;

#[test]
fn benchmark_sub_millisecond_proof() {
    let a = Tensor::glorot(&[128, 128]);
    let b = Tensor::glorot(&[128, 128]);
    let _ = tensor_matmul(&a, &b);

    let iters = 10_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = tensor_matmul(&a, &b);
    }
    let elapsed = t0.elapsed();
    let per_op_us = (elapsed.as_nanos() as f64) / (iters as f64) / 1_000.0;
    let per_op_ms = per_op_us / 1_000.0;

    println!("\n[1] 128x128 MatMul (10,000 iterations):");
    println!("    Total time:       {:.2?}", elapsed);
    println!("    Per operation:    {:.2} us ({:.4} ms)", per_op_us, per_op_ms);
    assert!(per_op_ms < 1.0, "Matmul must be sub-millisecond");

    let mut tape = GradTape::new();
    let mut w = Tensor::glorot(&[32, 1]);
    w.id = tape.alloc_id();
    w.requires_grad = true;
    let x = Tensor::zeros(&[1, 32]);
    let y = Tensor::zeros(&[1, 1]);

    let t1 = Instant::now();
    for _ in 0..iters {
        let pred = tape.matmul(&x, &w);
        let loss = tape.mse(&pred, &y);
        tape.backward(loss.id);
    }
    let elapsed_grad = t1.elapsed();
    let per_grad_us = (elapsed_grad.as_nanos() as f64) / (iters as f64) / 1_000.0;
    let per_grad_ms = per_grad_us / 1_000.0;

    println!("\n[2] Full Autograd Pass (Forward + MSE + Backward):");
    println!("    Total time:       {:.2?}", elapsed_grad);
    println!("    Per gradient step: {:.2} us ({:.4} ms)", per_grad_us, per_grad_ms);
    assert!(per_grad_ms < 1.0, "Autograd pass must be sub-millisecond");

    let src = "fn main() -> Tensor[1]:\n  let w = glorot(16, 1)\n  let x = zeros(1, 16)\n  return x @ w\n";
    let comp = compile(src, "bench.nr").unwrap();
    let mut vm = VM::new();
    vm.load(&comp.ir);

    let t2 = Instant::now();
    for _ in 0..iters {
        let _ = vm.run_main();
    }
    let elapsed_vm = t2.elapsed();
    let per_vm_us = (elapsed_vm.as_nanos() as f64) / (iters as f64) / 1_000.0;
    let per_vm_ms = per_vm_us / 1_000.0;

    println!("\n[3] End-to-End VM Execution:");
    println!("    Total time:       {:.2?}", elapsed_vm);
    println!("    Per VM execution: {:.2} us ({:.4} ms)", per_vm_us, per_vm_ms);
    assert!(per_vm_ms < 1.0, "VM execution must be sub-millisecond");

    println!("\n>>> ALL CORE OPERATIONS CONFIRMED SUB-MILLISECOND <<<");
}
'''
    test_path = os.path.join(ROOT_DIR, "runtime", "tests", "sub_millisecond_proof.rs")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_code)
    
    t0 = time.perf_counter()
    result = subprocess.run(
        ["cargo", "test", "--release", "--test", "sub_millisecond_proof",
         "-p", "neuron-runtime", "--", "--nocapture"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=ROOT_DIR
    )
    elapsed = (time.perf_counter() - t0) * 1000
    
    output = (result.stdout or "") + (result.stderr or "")
    for line in output.strip().split("\n"):
        if "Per " in line or "Total " in line:
            print(f"  {GREEN}{BOLD}{line}{RESET}")
        elif "SUB-MILLISECOND" in line or "CONFIRMED" in line:
            print(f"  {CYAN}{BOLD}{line}{RESET}")
        elif "Compiling" in line or "Finished" in line:
            print(f"  {DIM}{line}{RESET}")
        elif "ok" in line and "test" in line:
            print(f"  {GREEN}{line}{RESET}")
        else:
            print(f"  {line}")
    
    # Clean up
    try:
        os.remove(test_path)
    except:
        pass
    
    if result.returncode == 0:
        print(f"\n  {GREEN}{BOLD}✓ SUB-MILLISECOND PROOF VERIFIED in {elapsed:.0f} ms{RESET}")
    
    return result.returncode


# ═══════════════════════════════════════════════
#  DEMO SEQUENCE
# ═══════════════════════════════════════════════

def demo_1():
    banner(
        "DEMO 1: THE MILLION-DOLLAR BUG KILLER",
        "Wall Street's Most Expensive Bug — Caught at Compile Time"
    )
    print(f"  {BOLD}Scenario:{RESET} A quant accidentally leaks future price data")
    print(f"  into a backtest. In Python, this silently passes.")
    print(f"  In NEURON, the compiler catches it {RED}BEFORE{RESET} deployment.\n")
    
    print(f"  {YELLOW}Step 1: Compile the BUGGY version...{RESET}\n")
    run_neuronc("check", "demo1_wall_street_bug_killer.nr", expect_error=True)
    
    separator()
    
    print(f"  {YELLOW}Step 2: Now compile the SAFE version...{RESET}\n")
    run_neuronc("check", "demo1b_wall_street_safe.nr")

def demo_2():
    banner(
        "DEMO 2: THE MEDICAL SAFETY NET",
        "Uncertain[T] Blocks Low-Confidence Drug Dosages"
    )
    print(f"  {BOLD}Scenario:{RESET} An AI recommends chemotherapy dosage.")
    print(f"  If the model is unsure, NEURON {RED}BLOCKS{RESET} the dose")
    print(f"  and escalates to a human oncologist.\n")
    
    run_neuronc("run", "demo2_medical_safety.nr")

def demo_3():
    banner(
        "DEMO 3: NUCLEAR FUSION DISCOVERY",
        "Real MIT Alcator C-Mod Tokamak Data — Scaling Law Discovery"
    )
    print(f"  {BOLD}Data:{RESET} 20 real experimental shots from MIT PSFC")
    print(f"  {BOLD}Goal:{RESET} Discover α, β, γ coefficients of the plasma")
    print(f"  density limit scaling law: n_GL = α·Ip + β·Bt + γ·a\n")
    
    run_neuronc("run", "demo3_fusion_discovery.nr")

def demo_4():
    banner(
        "DEMO 4: EDGE AI — REAL-TIME ANOMALY DETECTION",
        "Zero Cloud. Zero Python. 1.19 MB Engine. Microsecond Latency."
    )
    print(f"  {BOLD}Scenario:{RESET} An industrial sensor array running on an")
    print(f"  edge device detects anomalies in real-time with")
    print(f"  {GREEN}zero internet connection{RESET} and {GREEN}zero cloud calls{RESET}.\n")
    
    run_neuronc("run", "demo4_edge_anomaly_detection.nr")

def demo_5_benchmark():
    banner(
        "DEMO 5: SUB-MILLISECOND EXECUTION PROOF",
        "10,000 Iterations — Raw Microsecond Measurements"
    )
    print(f"  {BOLD}Benchmark:{RESET} 128×128 MatMul, Full Autograd, VM Dispatch")
    print(f"  Running 10,000 iterations each in release mode...\n")
    
    run_benchmark()

def demo_6_browser():
    banner(
        "DEMO 6: IN-BROWSER AI ENGINE",
        "The Entire Compiler + Runtime Running in a Browser Tab"
    )
    print(f"  {BOLD}How:{RESET} Open demos/demo5_browser_engine.html in any browser.")
    print(f"  The 1.19 MB WebAssembly binary loads the full NEURON")
    print(f"  compiler, type checker, and autograd runtime.")
    print(f"  {GREEN}Zero install. Zero cloud. Zero Python.{RESET}")
    print()
    print(f"  {CYAN}→ Open: file://{os.path.join(DEMOS_DIR, 'demo5_browser_engine.html')}{RESET}")


# ═══════════════════════════════════════════════
#  MAIN ENTRY
# ═══════════════════════════════════════════════

def main():
    specific = None
    if "--demo" in sys.argv:
        idx = sys.argv.index("--demo")
        if idx + 1 < len(sys.argv):
            specific = int(sys.argv[idx + 1])
    
    print(f"\n{BOLD}{MAGENTA}")
    print(f"  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███╗   ██╗")
    print(f"  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗████╗  ██║")
    print(f"  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║██╔██╗ ██║")
    print(f"  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██║╚██╗██║")
    print(f"  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝██║ ╚████║")
    print(f"  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{RESET}")
    print()
    print(f"{BOLD}  NEURON LABS — LIVE STAGE DEMONSTRATION{RESET}")
    print(f"{DIM}  Founder: Fayo Ibrahim | Engine: 1.19 MB | Latency: <100 µs{RESET}")
    print(f"{DIM}  Built from scratch. Zero dollars. One laptop.{RESET}")
    print()
    
    demos = [
        (1, demo_1),
        (2, demo_2),
        (3, demo_3),
        (4, demo_4),
        (5, demo_5_benchmark),
        (6, demo_6_browser),
    ]
    
    if specific:
        for num, fn in demos:
            if num == specific:
                fn()
                break
    else:
        for i, (num, fn) in enumerate(demos):
            fn()
            if i < len(demos) - 1:
                pause()
    
    print()
    banner(
        "DEMONSTRATION COMPLETE",
        "NEURON — Sovereign AI Infrastructure, Built from First Principles."
    )
    print(f"  {BOLD}Contact:  licensing@neuron-lab.org{RESET}")
    print(f"  {BOLD}Website:  https://neuron-lab.org{RESET}")
    print(f"  {BOLD}Founder:  Fayo Ibrahim{RESET}")
    print()

if __name__ == "__main__":
    main()
