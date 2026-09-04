/// Integration tests for Offset-Based Temporal Types (§3.1.1).
/// Verifies algebraic composition (.shift, .lag, .lead), horizon alignment, and compile-time rejection.

use neuron_compiler::lexer::Lexer;
use neuron_compiler::parser::Parser;
use neuron_compiler::types::TypeChecker;

fn check_src(src: &str) -> Result<(), Vec<String>> {
    let tokens = Lexer::new(src).tokenize().map_err(|e| vec![e.to_string()])?;
    let ast = Parser::new(tokens, "test.nr").parse().map_err(|e| vec![e.to_string()])?;
    let mut checker = TypeChecker::new("test.nr");
    checker.check(&ast);
    if checker.result.has_errors() {
        Err(checker.result.errors.iter().map(|e| e.to_string()).collect())
    } else {
        Ok(())
    }
}

#[test]
fn test_temporal_offset_valid_algebra() {
    let code = r#"
fn process_past(data: Temporal[Tensor, 0]) -> Tensor:
    return data.snapshot()

fn main():
    let prices: Temporal[Tensor, 0] = zeros(10, 4)
    let sma: Temporal[Tensor, -5] = prices.shift(-5)
    let signal = sma.shift(2)
    let out = process_past(signal)
"#;
    let res = check_src(code);
    assert!(res.is_ok(), "Expected valid offset shift algebra to pass: {:?}", res);
}

#[test]
fn test_temporal_offset_lag_and_lead() {
    let code = r#"
fn safe_strategy(x: Temporal[Tensor, 0]) -> Tensor:
    return x.snapshot()

fn main():
    let data: Temporal[Tensor, 0] = zeros(10, 4)
    let lagged = data.lag(5)
    let res = safe_strategy(lagged)
"#;
    let res = check_src(code);
    assert!(res.is_ok(), "Expected lagged data to satisfy safe strategy: {:?}", res);
}

#[test]
fn test_temporal_offset_negative_rejection() {
    let code = r#"
fn safe_strategy(x: Temporal[Tensor, 0]) -> Tensor:
    return x.snapshot()

fn main():
    let data: Temporal[Tensor, 0] = zeros(10, 4)
    let future_leak = data.shift(3)
    let res = safe_strategy(future_leak)
"#;
    let res = check_src(code);
    assert!(res.is_err(), "Expected future offset (+3) to be rejected by safe_strategy");
    let errors = res.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Temporal offset violation") || e.contains("TemporalLeak") || e.contains("+3")),
        "Expected error mentioning temporal offset violation: {:?}", errors);
}

#[test]
fn test_temporal_offset_lead_rejection() {
    let code = r#"
fn safe_strategy(x: Temporal[Tensor, 0]) -> Tensor:
    return x.snapshot()

fn main():
    let data: Temporal[Tensor, 0] = zeros(10, 4)
    let lead_leak = data.lead(2)
    let res = safe_strategy(lead_leak)
"#;
    let res = check_src(code);
    assert!(res.is_err(), "Expected lead data (+2) to be rejected");
}

#[test]
fn test_temporal_offset_binary_conservative_composition() {
    let code = r#"
fn require_lagged_2(x: Temporal[Tensor, -2]) -> Tensor:
    return x.snapshot()

fn main():
    let a: Temporal[Tensor, -2] = zeros(5, 5)
    let b: Temporal[Tensor, -5] = zeros(5, 5)
    let c = a + b
    let res = require_lagged_2(c)
"#;
    let res = check_src(code);
    assert!(res.is_ok(), "Expected binary op max(-2, -5) to yield offset -2: {:?}", res);

    let code_invalid = r#"
fn require_lagged_5(x: Temporal[Tensor, -5]) -> Tensor:
    return x.snapshot()

fn main():
    let a: Temporal[Tensor, -2] = zeros(5, 5)
    let b: Temporal[Tensor, -5] = zeros(5, 5)
    let c = a + b
    let res = require_lagged_5(c)
"#;
    assert!(check_src(code_invalid).is_err(), "Expected binary op max(-2, -5) = -2 to reject requirement -5");
}

#[test]
fn test_temporal_offset_multi_horizon_matching() {
    let valid_code = r#"
fn evaluate_horizon(pred: Temporal[Tensor, 5], target: Temporal[Tensor, 5]) -> Tensor:
    return (pred - target).snapshot()

fn main():
    let p: Temporal[Tensor, 5] = zeros(10, 1)
    let t: Temporal[Tensor, 5] = zeros(10, 1)
    let loss = evaluate_horizon(p, t)
"#;
    assert!(check_src(valid_code).is_ok(), "Expected matching +5 step horizons to pass");

    let invalid_code = r#"
fn evaluate_horizon(pred: Temporal[Tensor, 5], target: Temporal[Tensor, 5]) -> Tensor:
    return (pred - target).snapshot()

fn main():
    let p: Temporal[Tensor, 5] = zeros(10, 1)
    let t_mismatch: Temporal[Tensor, 1] = zeros(10, 1)
    let loss = evaluate_horizon(p, t_mismatch)
"#;
    assert!(check_src(invalid_code).is_err(), "Expected horizon mismatch (+5 vs +1) to fail");
}
