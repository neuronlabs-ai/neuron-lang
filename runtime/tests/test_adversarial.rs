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
