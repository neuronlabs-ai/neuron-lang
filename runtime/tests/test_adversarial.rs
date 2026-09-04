/// Adversarial test suite for NEURON
/// Tests edge cases, crash scenarios, and type system bypass attempts

use neuron_compiler::compile;
use neuron_runtime::vm::VM;

fn run(src: &str) -> Result<String, String> {
    neuron_runtime::device::set_force_cpu(true);
    let out = compile(src, "test.nr").map_err(|e| format!("{:?}", e))?;
    let mut vm = VM::new();
    vm.load(&out.ir);
    
    // Catch panics gracefully to prevent tests themselves from panicking on runtime errors
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        vm.run_main()
    }));
    
    match result {
        Ok(Ok(val)) => Ok(format!("{:?}", val)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Runtime panicked".to_string()),
    }
}

fn should_compile_error(src: &str) -> bool {
    compile(src, "test.nr").is_err()
}

fn should_run_ok(src: &str) -> Result<String, String> {
    run(src)
}

// ═══════════════════════════════════════════════════════════════
//  1. EMPTY / DEGENERATE PROGRAMS
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_empty_source() {
    // Compiling empty file succeeds (as a library/module), but running it fails due to no main()
    assert!(run("").is_err());
}

#[test]
fn adversarial_whitespace_only() {
    assert!(run("   \n\n  \n").is_err());
}

#[test]
fn adversarial_comment_only() {
    assert!(run("# just a comment\n").is_err());
}

#[test]
fn adversarial_minimal_fn() {
    should_run_ok("fn main() -> Int:\n  return 0\n").expect("minimal function should work");
}

// ═══════════════════════════════════════════════════════════════
//  2. TENSOR EDGE CASES
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_zero_tensor() {
    // zero-sized tensor — should not crash/panic
    let result = run("fn main() -> Float:\n  let x = zeros(0, 0)\n  return 0.0\n");
    match result {
        Ok(_) => {},
        Err(e) => { assert!(!e.contains("panic") && !e.contains("Runtime panicked"), "should not panic: {}", e); }
    }
}

#[test]
fn adversarial_shape_mismatch_matmul() {
    let result = run("fn main() -> Float:\n  let a = zeros(2, 3)\n  let b = zeros(4, 2)\n  let c = a @ b\n  return 0.0\n");
    assert!(result.is_err(), "mismatched matmul should error");
}

#[test]
fn adversarial_large_tensor() {
    should_run_ok("fn main() -> Float:\n  let x = zeros(100, 100)\n  return 0.0\n")
        .expect("large tensor should work");
}

// ═══════════════════════════════════════════════════════════════
//  3. ARITHMETIC EDGE CASES
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_float_div_zero() {
    // Should produce Inf, not crash
    let r = should_run_ok("fn main() -> Float:\n  let x = 1.0 / 0.0\n  return x\n");
    match r {
        Ok(v) => { assert!(v.contains("inf") || v.contains("Inf") || v.contains("NaN") || v.parse::<f64>().is_ok(), "unexpected: {}", v); },
        Err(_) => {} // controlled error is fine too
    }
}

#[test]
fn adversarial_zero_div_zero() {
    // Should produce NaN, not crash
    let r = run("fn main() -> Float:\n  let x = 0.0 / 0.0\n  return x\n");
    match r {
        Ok(v) => { assert!(v.contains("NaN") || v.contains("nan"), "expected NaN, got: {}", v); },
        Err(_) => {} // controlled error is fine too
    }
}

#[test]
fn adversarial_negative_values() {
    should_run_ok("fn main() -> Float:\n  let x = -1.0\n  let y = x * x\n  return y\n")
        .expect("negative multiplication should work");
}

// ═══════════════════════════════════════════════════════════════
//  4. ACTIVATION EDGE CASES
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_relu_negative() {
    let r = should_run_ok("fn main() -> Tensor:\n  let x = zeros(2, 2) - 100.0\n  let y = relu(x)\n  return y.sum()\n");
    match r {
        Ok(v) => {
            assert!(v.contains("0.0"), "relu of all negatives should be 0, got: {}", v);
        },
        Err(e) => panic!("relu on negatives crashed: {}", e),
    }
}

#[test]
fn adversarial_optimizer_matmul_relu_fusion_direct() {
    // R16 regression test: MatMul -> ReLU must NOT erase the ReLU activation.
    // [1.0] @ [-1.0] = -1.0. relu(-1.0) must return 0.0, NOT -1.0.
    let src = r#"
fn main() -> Tensor[1, 1]:
  let x = zeros(1, 1) + 1.0
  let w = zeros(1, 1) - 1.0
  let y = x @ w
  let z = relu(y)
  return z
"#;
    let r = should_run_ok(src).expect("matmul followed by relu should run cleanly");
    assert!(r.contains("0.0"), "relu of negative matmul must produce 0.0, got: {}", r);
}

#[test]
fn adversarial_sigmoid_extreme_positive() {
    let r = should_run_ok("fn main() -> Tensor:\n  let x = zeros(2, 2) + 1000.0\n  let y = sigmoid(x)\n  return y.sum()\n");
    match r {
        Ok(v) => {
            assert!(v.contains("4.0") || v.contains("4"), "sigmoid(1000) * 4 elements should sum to ~4.0, got: {}", v);
        },
        Err(e) => panic!("sigmoid extreme crashed: {}", e),
    }
}

#[test]
fn adversarial_sigmoid_extreme_negative() {
    let r = should_run_ok("fn main() -> Tensor:\n  let x = zeros(2, 2) - 1000.0\n  let y = sigmoid(x)\n  return y.sum()\n");
    match r {
        Ok(v) => {
            assert!(v.contains("0.0"), "sigmoid(-1000) * 4 elements should sum to ~0.0, got: {}", v);
        },
        Err(e) => panic!("sigmoid extreme negative crashed: {}", e),
    }
}

#[test]
fn adversarial_softmax_zeros() {
    let r = should_run_ok("fn main() -> Tensor:\n  let x = zeros(1, 4)\n  let y = softmax(x)\n  return y.sum()\n");
    match r {
        Ok(v) => {
            // softmax of zeros should sum to 1.0 (per row) * num_rows
            assert!(v.contains("1.0") || v.contains("1"), "softmax sum should be ~1.0, got: {}", v);
        },
        Err(e) => panic!("softmax zeros crashed: {}", e),
    }
}

// ═══════════════════════════════════════════════════════════════
//  5. MALFORMED SYNTAX
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_unclosed_paren() {
    assert!(should_compile_error("fn main() -> Int:\n  return (1 + 2\n"), "unclosed paren should fail");
}

#[test]
fn adversarial_garbage_tokens() {
    assert!(should_compile_error("asdf !@#$ %^&*"), "garbage should fail");
}

#[test]
fn adversarial_incomplete_fn() {
    assert!(should_compile_error("fn"), "incomplete fn should fail");
}

#[test]
fn adversarial_fn_no_body() {
    // fn with declaration but no body
    let _result = should_compile_error("fn main() -> Int:\n");
    // This might parse as valid with implicit return — either way, should not panic
    assert!(true, "did not panic");
}

#[test]
fn adversarial_double_arrow() {
    assert!(should_compile_error("fn main() -> -> Int:\n  return 0\n"), "double arrow should fail");
}

// ═══════════════════════════════════════════════════════════════
//  6. DEEP COMPUTATION
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_deep_nesting() {
    should_run_ok("fn main() -> Float:\n  let x = 1.0\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  let x = x + x\n  return x\n")
        .expect("deep nesting should work");
}

// ═══════════════════════════════════════════════════════════════
//  7. MODEL OPERATIONS
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_model_basic() {
    let src = r#"
model MLP:
  w: Tensor[2, 2] = zeros(2, 2) + 1.0

  fn forward(self, x: Tensor[2, 2]) -> Tensor[2, 2]:
    return self.w @ x

fn main() -> Tensor:
  let m = MLP()
  let x = zeros(2, 2) + 0.5
  let y = m.forward(x)
  return y.sum()
"#;
    should_run_ok(src).expect("basic model should work");
}

#[test]
fn adversarial_forget_basic() {
    let src = r#"
model Tiny:
  w: Tensor[2, 2] = zeros(2, 2) + 1.0

  fn forward(self, x: Tensor[2, 2]) -> Tensor[2, 2]:
    return self.w @ x

fn main() -> Float:
  let m = Tiny()
  let task_data = zeros(2, 2) + 0.5
  let cert = forget(m, task_data, "GradientAscent", 0.5)
  return 0.0
"#;
    should_run_ok(src).expect("forget should work");
}

#[test]
fn adversarial_r14_forget_zero_strength_not_successful() {
    let src = r#"
model Tiny:
  w: Tensor[2, 2] = zeros(2, 2) + 1.0

  fn forward(self, x: Tensor[2, 2]) -> Tensor[2, 2]:
    return self.w @ x

fn main() -> Bool:
  let m = Tiny()
  let task_data = zeros(2, 2) + 0.5
  let cert = forget(m, task_data, "GradientAscent", 0.0)
  return cert.forgetting_successful
"#;
    let r = should_run_ok(src).expect("forget with zero strength should return certificate");
    assert!(r.contains("false"), "zero strength forget certificate must have forgetting_successful=false, got: {}", r);
}

#[test]
fn adversarial_r17_stop_grad_backward_safety() {
    let src = r#"
fn main() -> Tensor:
  let x = zeros(2, 2) + 2.0
  let y = stop_grad(x)
  let z = y + y
  return z
"#;
    should_run_ok(src).expect("stop_grad graph severance must not crash during backward");
}

// ═══════════════════════════════════════════════════════════════
//  8. CHAINED OPERATIONS STRESS
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_chained_matmul() {
    let src = r#"
@opaque
fn main() -> Tensor[4, 4]:
  let w = glorot(4, 4)
  let x = glorot(4, 4)
  let x = x @ w
  let x = x @ w
  let x = x @ w
  let x = x @ w
  let x = x @ w
  return x
"#;
    should_run_ok(src).expect("chained matmul should work");
}

#[test]
fn adversarial_chained_activations() {
    let src = r#"
fn main() -> Tensor:
  let x = zeros(4, 4) + 0.5
  let x = relu(x)
  let x = sigmoid(x)
  let x = gelu(x)
  let x = relu(x)
  let x = sigmoid(x)
  return x.sum()
"#;
    should_run_ok(src).expect("chained activations should work");
}

// ═══════════════════════════════════════════════════════════════
//  9. VERY LONG SOURCE
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_many_variables() {
    let mut src = String::from("fn main() -> Float:\n");
    for i in 0..200 {
        src.push_str(&format!("  let v{} = {}.0\n", i, i));
    }
    src.push_str("  return v199\n");
    should_run_ok(&src).expect("200 variables should work");
}

// ═══════════════════════════════════════════════════════════════
//  10. TRAINING LOOP EDGE CASES
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_training_zero_lr() {
    let src = r#"
model Net:
  w: Tensor[2, 2] = glorot(2, 2)

  fn forward(self, x: Tensor[2, 2]) -> Tensor[2, 2]:
    return self.w @ x

  fn update_weights(self, loss: Loss) [Effect[Mut[self]]]:
    update self.w by sgd(grad(loss), lr=0.0)

fn main() -> Tensor:
  let m = Net()
  let x = zeros(2, 2) + 1.0
  let y = m.forward(x)
  let loss = mse(y, zeros(2, 2))
  m.update_weights(loss)
  return loss
"#;
    // Zero learning rate should work — just no parameter update
    should_run_ok(src).expect("training with zero lr should not crash");
}

// ═══════════════════════════════════════════════════════════════
//  11. CODEX AUDIT REPRODUCTIONS
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_deep_parentheses() {
    let mut src = "fn main() -> Int:\n  return ".to_string();
    for _ in 0..1000 {
        src.push('(');
    }
    src.push_str("0");
    for _ in 0..1000 {
        src.push(')');
    }
    src.push('\n');
    let res = compile(&src, "test.nr");
    let err = match res {
        Ok(_) => panic!("expected compilation failure"),
        Err(e) => format!("{:?}", e),
    };
    assert!(err.contains("recursion depth exceeded") || err.contains("ParseError"), "unexpected error: {}", err);
}

#[test]
fn adversarial_negative_dimension() {
    let src = "fn main() -> Tensor:\n  return zeros(2, -3)\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("Negative dimension size"), "unexpected error: {}", err);
}

#[test]
fn adversarial_oversized_tensor() {
    let src = "fn main() -> Tensor:\n  return zeros(4294967296, 4294967296)\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("overflow") || err.contains("too large"), "unexpected error: {}", err);
}

#[test]
fn adversarial_negative_index() {
    let src = "fn main() -> Float:\n  let x = zeros(5)\n  let y = x[-1]\n  return y\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("Negative index"), "unexpected error: {}", err);
}

#[test]
fn adversarial_path_traversal_ohlcv() {
    let src = "fn main() -> List[List[Float]]:\n  return load_ohlcv(\"../secret.csv\")\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("Security Error"), "unexpected error: {}", err);
}

// ═══════════════════════════════════════════════════════════════
//  12. PHASE 2 CODEX AUDIT REPRODUCTIONS
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_windows_backslash_path() {
    let src = "fn main() -> List[List[Float]]:\n  return load_ohlcv(\"\\\\Users\\\\secret.csv\")\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("Security Error"), "Windows backslash path should be blocked: {}", err);
}

#[test]
fn adversarial_modulo_operation() {
    let src = "fn main() -> Int:\n  let x = 10 % 3\n  return x\n";
    let res = run(src);
    assert!(res.is_ok(), "modulo should succeed: {:?}", res);
    let val = res.unwrap();
    assert!(val.contains("1"), "10 % 3 should be 1, got: {}", val);
}

#[test]
fn adversarial_modulo_div_by_zero() {
    let src = "fn main() -> Int:\n  let x = 10 % 0\n  return x\n";
    let res = run(src);
    assert!(res.is_err(), "modulo by zero should error");
}

#[test]
fn adversarial_logical_and_or() {
    let src = "fn main() -> Bool:\n  let x = true and false\n  let y = true or false\n  return y\n";
    let res = run(src);
    assert!(res.is_ok(), "logical and/or should work: {:?}", res);
    let val = res.unwrap();
    assert!(val.contains("true") || val.contains("True") || val.contains("Bool(true)"), "true or false should be true, got: {}", val);
}

#[test]
fn adversarial_deep_nested_type() {
    // Generate Uncertain[Uncertain[...Uncertain[Int]...]] at depth 40 (exceeds limit of 30)
    let mut type_str = "Int".to_string();
    for _ in 0..40 {
        type_str = format!("Uncertain[{}]", type_str);
    }
    let src = format!("fn foo(x: {}) -> Int:\n  return 0\n\nfn main() -> Int:\n  return foo(0)\n", type_str);
    let res = compile(&src, "test.nr");
    assert!(res.is_err(), "deeply nested type should cause compile error");
    let err = match res {
        Ok(_) => panic!("expected compilation failure"),
        Err(e) => format!("{:?}", e),
    };
    assert!(err.contains("recursion depth exceeded") || err.contains("ParseError"), "unexpected error: {}", err);
}

#[test]
fn adversarial_update_row_width_mismatch() {
    let src = r#"
fn main() -> Tensor:
  let t = zeros(3, 4)
  let row = zeros(1, 2)
  let t = update_row(t, 0, row)
  return t
"#;
    let res = run(src);
    assert!(res.is_err(), "update_row with mismatched row width should error");
    let err = res.unwrap_err();
    assert!(err.contains("Row width mismatch") || err.contains("Runtime panicked"), "unexpected error: {}", err);
}

#[test]
fn adversarial_concat_exceeds_ceiling() {
    // Each tensor is 60M elements, concatenating two should exceed the 100M ceiling (120M total)
    let src = r#"
fn main() -> Tensor:
  let a = zeros(10000, 6000)
  let b = zeros(10000, 6000)
  let c = concat([a, b])
  return c
"#;
    let res = run(src);
    // Should either fail at tensor creation or at concat
    assert!(res.is_err(), "concat exceeding element ceiling should error");
}

#[test]
fn adversarial_transpose_oob_dim() {
    let src = r#"
fn main() -> Tensor:
  let t = zeros(2, 3)
  let r = t.sum(5)
  return r
"#;
    let res = run(src);
    assert!(res.is_err(), "sum with out-of-bounds dim should error");
}

#[test]
fn adversarial_mean_oob_dim() {
    let src = r#"
fn main() -> Tensor:
  let t = zeros(2, 3)
  let r = t.mean(10)
  return r
"#;
    let res = run(src);
    assert!(res.is_err(), "mean with out-of-bounds dim should error");
}

#[test]
fn adversarial_drive_letter_path() {
    let src = "fn main() -> List[List[Float]]:\n  return load_ohlcv(\"C:\\\\Windows\\\\System32\\\\config.csv\")\n";
    let res = run(src);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("Security Error"), "drive letter path should be blocked: {}", err);
}

// ═══════════════════════════════════════════════════════════════
//  13. PHASE 3 ADVOCATED HARDENING TESTS
// ═══════════════════════════════════════════════════════════════

fn run_jit_helper(src: &str) -> Result<neuron_runtime::vm::Value, String> {
    let compile_res = neuron_compiler::compile(src, "test_jit_input.nr")
        .map_err(|e| format!("{:?}", e))?;
    
    let mut rust_code = neuron_compiler::transpiler::Transpiler::transpile(&compile_res.ir);
    rust_code = format!("#![allow(warnings)]\n{}", rust_code);
    
    let temp_dir = std::env::temp_dir().join(format!(
        "neuron_jit_test_project_adv_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let src_dir = temp_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    let runtime_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .to_string_lossy()
        .replace('\\', "/");
    let cargo_toml_content = format!(r#"[package]
name = "neuron_jit_test_adv"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);
    std::fs::write(temp_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();
    
    let compile_status = std::process::Command::new("cargo")
        .arg("build")
        .current_dir(&temp_dir)
        .status()
        .map_err(|e| format!("Failed to run cargo: {:?}", e))?;
        
    if !compile_status.success() {
        return Err("JIT compilation failed".to_string());
    }
    
    let lib_path = if cfg!(target_os = "windows") {
        temp_dir.join("target").join("debug").join("neuron_jit_test_adv.dll")
    } else if cfg!(target_os = "macos") {
        temp_dir.join("target").join("debug").join("libneuron_jit_test_adv.dylib")
    } else {
        temp_dir.join("target").join("debug").join("libneuron_jit_test_adv.so")
    };
    
    let unique_dll_name = format!(
        "neuron_jit_test_adv_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = temp_dir.join("target").join("debug").join(&unique_dll_name);
    std::fs::copy(&lib_path, &load_lib_path).unwrap();

    let lib = unsafe { libloading::Library::new(&load_lib_path) }
        .map_err(|e| format!("Failed to load JIT library: {:?}", e))?;
        
    let result = unsafe {
        let run_main: libloading::Symbol<fn(&mut VM) -> neuron_runtime::vm::Value> = lib
            .get(b"run_main")
            .map_err(|e| format!("Failed to resolve run_main: {:?}", e))?;
        let mut vm = VM::new();
        run_main(&mut vm)
    };
    
    drop(lib);
    let _ = std::fs::remove_dir_all(&temp_dir);
    
    Ok(result)
}

#[test]
fn adversarial_python_string_escaping() {
    let src = r#"
fn main() -> String:
  let s = "{__import__('builtins').print('NEURON_FSTRING_EXECUTED')}"
  return s
"#;
    let compile_res = compile(src, "test_py.nr").unwrap();
    let py_code = neuron_compiler::py_transpiler::PyTranspiler::transpile(&compile_res.ir);
    
    assert!(!py_code.contains("f\"\"\""), "Python transpiler should not emit f-strings for string literals");
    assert!(py_code.contains("\"{__import__"), "Python transpiler should escape or output literal strings safely");
}

#[test]
fn adversarial_jit_panic_safety() {
    let src = "fn main() -> Tensor:\n  return zeros(-1)\n";
    let res = run_jit_helper(src);
    assert!(res.is_ok(), "JIT run should succeed (no crash)");
    let val = res.unwrap();
    match val {
        neuron_runtime::vm::Value::Err(msg) => {
            assert!(msg.contains("Negative dimension size"), "Unexpected error: {}", msg);
        }
        _ => panic!("Expected Value::Err, got {:?}", val),
    }
}

#[test]
fn adversarial_jit_imports() {
    let temp_dir = std::env::temp_dir().join(format!(
        "neuron_import_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();
    
    let helper_code = r#"
fn helper() -> Int:
  return 7
"#;
    let main_code = r#"
from helper import helper

fn main() -> Int:
  let x = helper()
  return x
"#;
    
    let helper_path = temp_dir.join("helper.nr");
    let main_path = temp_dir.join("main.nr");
    std::fs::write(&helper_path, helper_code).unwrap();
    std::fs::write(&main_path, main_code).unwrap();
    
    let source = std::fs::read_to_string(&main_path).unwrap();
    let compile_res = neuron_compiler::compile_with_imports(&source, main_path.to_str().unwrap());
    assert!(compile_res.is_ok(), "Compilation with imports failed: {:?}", compile_res.err());
    
    let ir = compile_res.unwrap().ir;
    let mut rust_code = neuron_compiler::transpiler::Transpiler::transpile(&ir);
    rust_code = format!("#![allow(warnings)]\n{}", rust_code);
    
    // Now compile and run JIT
    let jit_dir = std::env::temp_dir().join(format!(
        "neuron_jit_test_project_import_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let src_dir = jit_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    let runtime_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .to_string_lossy()
        .replace('\\', "/");
    let cargo_toml_content = format!(r#"[package]
name = "neuron_jit_test_import"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuron-runtime = {{ path = "{}" }}
"#, runtime_path);
    std::fs::write(jit_dir.join("Cargo.toml"), cargo_toml_content).unwrap();
    std::fs::write(src_dir.join("lib.rs"), rust_code).unwrap();
    
    let compile_status = std::process::Command::new("cargo")
        .arg("build")
        .current_dir(&jit_dir)
        .status()
        .unwrap();
        
    assert!(compile_status.success());
    
    let lib_path = if cfg!(target_os = "windows") {
        jit_dir.join("target").join("debug").join("neuron_jit_test_import.dll")
    } else if cfg!(target_os = "macos") {
        jit_dir.join("target").join("debug").join("libneuron_jit_test_import.dylib")
    } else {
        jit_dir.join("target").join("debug").join("libneuron_jit_test_import.so")
    };
    
    let unique_dll_name = format!(
        "neuron_jit_test_import_{}.dll",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let load_lib_path = jit_dir.join("target").join("debug").join(&unique_dll_name);
    std::fs::copy(&lib_path, &load_lib_path).unwrap();

    let lib = unsafe { libloading::Library::new(&load_lib_path) }.unwrap();
    let result = unsafe {
        let run_main: libloading::Symbol<fn(&mut VM) -> neuron_runtime::vm::Value> = lib
            .get(b"run_main")
            .unwrap();
        let mut vm = VM::new();
        run_main(&mut vm)
    };
    
    drop(lib);
    let _ = std::fs::remove_dir_all(&jit_dir);
    let _ = std::fs::remove_dir_all(&temp_dir);
    
    match result {
        neuron_runtime::vm::Value::Int(val) => assert_eq!(val, 7),
        _ => panic!("Expected Int(7), got {:?}", result),
    }
}

// ═══════════════════════════════════════════════════════════════
//  13. CODEX AUDIT TYPE SOUNDNESS REGRESSIONS (R1–R9)
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_r6_undefined_variable_fails() {
    let src = r#"
fn main() -> Float:
  return missing_value + 1.0
"#;
    assert!(should_compile_error(src), "undefined variables must cause compile error");
}

#[test]
fn adversarial_r5_return_type_mismatch_fails() {
    let src = r#"
fn lie() -> Tensor[1, 1]:
  return 1.0

fn main() -> Tensor[1, 1]:
  return lie()
"#;
    assert!(should_compile_error(src), "return type mismatch must cause compile error");
}

#[test]
fn adversarial_r1_temporal_raw_tensor_bypass_fails() {
    let src = r#"
fn accepts_raw_tensor(x: Tensor) -> Tensor:
  return x

fn main() -> Tensor:
  let prices: Temporal[Tensor, 0] = randn(4, 4)
  let future = prices.shift(3)
  return accepts_raw_tensor(future)
"#;
    assert!(should_compile_error(src), "passing Temporal to raw Tensor must be rejected");
}

#[test]
fn adversarial_r4_causal_raw_float_bypass_fails() {
    let src = r#"
fn add_raw(a: Float, b: Float) -> Float:
  return a + b

fn main() -> Float:
  let observed: Causal[Float, observed] = 1.0
  let intervened: Causal[Float, intervened] = 2.0
  return add_raw(observed, intervened)
"#;
    assert!(should_compile_error(src), "passing Causal to raw Float must be rejected");
}

#[test]
fn adversarial_r2_temporal_before_laundering_fails() {
    let src = r#"
fn requires_safe(x: Temporal[Tensor, 0]) -> Tensor:
  return x.snapshot()

fn main() -> Tensor:
  let prices: Temporal[Tensor, 0] = randn(4, 4)
  let future = prices.shift(10)
  let laundered = future.before(1)
  return requires_safe(laundered)
"#;
    assert!(should_compile_error(src), "laundered temporal offset +9 passed to 0 must be rejected");
}

#[test]
fn adversarial_r3_temporal_binary_composition_preserves_future() {
    let src = r#"
fn requires_safe(x: Temporal[Tensor, 0]) -> Tensor:
  return x.snapshot()

fn main() -> Tensor:
  let prices: Temporal[Tensor, 0] = randn(4, 4)
  let past = prices.shift(-5)
  let future = prices.shift(3)
  let mixed = past + future
  return requires_safe(mixed)
"#;
    assert!(should_compile_error(src), "temporal binary composition with future (+3) passed to 0 must be rejected");
}

#[test]
fn adversarial_r7_if_block_scoping_prevents_escape() {
    let src = r#"
fn main() -> Int:
  let flag = false
  if flag:
    let x = 1
  return x
"#;
    assert!(should_compile_error(src), "variables defined inside if block must not escape");
}

#[test]
fn adversarial_r8_effect_loop_scope_propagated() {
    let src = r#"
model Tiny:
  w: Tensor[1, 1] = zeros(1, 1) + 1.0

  fn mutate_in_loop(self):
    for i in range(1):
      update self.w by sgd(grad(self.w), lr=0.1)
"#;
    assert!(should_compile_error(src), "mutations inside loop must require Effect declaration");
}

#[test]
fn adversarial_r9_effect_wrong_target_rejected() {
    let src = r#"
model Tiny:
  w: Tensor[1, 1] = zeros(1, 1) + 1.0

  fn mutate_wrong_target(self) [Effect[Mut[other]]]:
    update self.w by sgd(grad(self.w), lr=0.1)
"#;
    assert!(should_compile_error(src), "mutating self.w when only Mut[other] is declared must be rejected");
}

// ═══════════════════════════════════════════════════════════════
//  ROUND 2 ADVERSARIAL AUDIT REGRESSIONS (R2-1 through R2-7)
// ═══════════════════════════════════════════════════════════════

#[test]
fn adversarial_r2_1_temporal_shift_variable_leak() {
    let src = r#"
fn requires_safe(x: Temporal[Tensor, 0]) -> Tensor:
  return x.snapshot()

fn main() -> Tensor:
  let prices: Temporal[Tensor, 0] = randn(2, 2)
  let k = 5
  let future = prices.shift(k)
  return requires_safe(future)
"#;
    assert!(should_compile_error(src), "variable shift into future must be rejected at compile time");
}

#[test]
fn adversarial_r2_3_interprocedural_effects_propagated() {
    let src = r#"
model Tiny:
  w: Tensor[1, 1] = zeros(1, 1) + 1.0

  fn mutate(self) [Effect[Mut[self]]]:
    update self.w by sgd(grad(self.w), lr=0.1)

  fn pure_wrapper(self):
    self.mutate()

fn main():
  let t = Tiny()
  t.pure_wrapper()
"#;
    assert!(should_compile_error(src), "caller must declare Effect when calling a method that has Effect[Mut[self]]");
}

#[test]
fn adversarial_r2_5_uncertain_arithmetic_evaluation() {
    let src = r#"
fn main() -> Float:
  let u = Normal(10.0, 1.0)
  let res = u + 5.0
  let res2 = res * 2.0
  let res3 = res2 / 2.0
  let res4 = res3 - 5.0
  if res4.confidence > 0.5:
    return res4.value
  return 0.0
"#;
    let r = should_run_ok(src).expect("Uncertain arithmetic operations should evaluate without runtime panics");
    assert!(r.contains("10.0") || r.contains("10"), "Result should be around 10.0, got: {}", r);
}

#[test]
fn adversarial_r2_6_forget_negative_strength_rejected() {
    let src = r#"
model SafetyNet:
  w: Tensor[4, 1] = glorot(4, 1)

fn main():
  let net = SafetyNet()
  let sensitive_data = zeros(1, 4) + 0.9
  let cert = forget(net, sensitive_data, "GradientAscent", -1.0)
"#;
    assert!(run(src).is_err(), "negative unlearning strength must be rejected with runtime error");
}

#[test]
fn adversarial_r2_7_forget_unknown_method_rejected() {
    let src = r#"
model SafetyNet:
  w: Tensor[4, 1] = glorot(4, 1)

fn main():
  let net = SafetyNet()
  let sensitive_data = zeros(1, 4) + 0.9
  let cert = forget(net, sensitive_data, "NoSuchMethod", 0.5)
"#;
    assert!(run(src).is_err(), "unknown unlearning method must be rejected with runtime error");
}

#[test]
fn adversarial_tuple_causal_escape_prevented() {
    let src = r#"
fn unwrap_pair(pair: (Float, Int)) -> Float:
  return pair[0]

fn main() -> Float:
  let c: Causal[Float, "observed"] = 1.0
  let p = (c, 42)
  return unwrap_pair(p)
"#;
    assert!(should_compile_error(src), "Causal wrapper inside a tuple must not escape to raw Float");
}



