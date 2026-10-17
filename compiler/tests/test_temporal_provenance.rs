use neuron_compiler::check_only;
use neuron_compiler::errors::ErrorCode;

#[test]
fn test_legitimate_future_target_allowed() {
    let code = r#"
fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = build_labels(x)
    print(y)
"#;
    let res = check_only(code, "test_target.nr");
    assert!(!res.has_errors(), "Expected valid target labeling to compile cleanly, but got errors: {:?}", res.errors);
}

#[test]
fn test_direct_future_target_leak_rejected() {
    let code = r#"
fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn train_model(features: Temporal[Tensor[5000], 0]):
    print(features)

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = build_labels(x)
    train_model(y)
"#;
    let res = check_only(code, "test_direct_leak.nr");
    assert!(res.has_errors(), "Expected direct target leak to be rejected!");
    let has_temporal_leak = res.errors.iter().any(|e| e.code == ErrorCode::TemporalLeak);
    assert!(has_temporal_leak, "Expected ErrorCode::TemporalLeak in errors: {:?}", res.errors);
}

#[test]
fn test_target_laundering_method_rejected() {
    let code = r#"
fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn enrich_features(features: Temporal[Tensor[5000], 0], labels: Temporal[Tensor[5000], 3]) -> Temporal[Tensor[5000], 0]:
    let m = labels.mean()
    let laundered = features * m
    return laundered

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = build_labels(x)
    let f = enrich_features(x, y)
    print(f)
"#;
    let res = check_only(code, "test_laundering_method.nr");
    assert!(res.has_errors(), "Expected mean-laundering method to be rejected!");
    let has_temporal_leak = res.errors.iter().any(|e| e.code == ErrorCode::TemporalLeak);
    assert!(has_temporal_leak, "Expected ErrorCode::TemporalLeak in errors: {:?}", res.errors);
    let has_global_sample_mention = res.errors.iter().any(|e| e.message.contains("global_sample"));
    assert!(has_global_sample_mention, "Expected error to mention global_sample filtration provenance: {:?}", res.errors);
}

#[test]
fn test_target_laundering_function_call_rejected() {
    let code = r#"
fn build_labels(prices: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 3]:
    return prices.shift(3)

fn enrich_features(features: Temporal[Tensor[5000], 0], labels: Temporal[Tensor[5000], 3]) -> Temporal[Tensor[5000], 0]:
    let m = mean(labels)
    let laundered = features * m
    return laundered

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = build_labels(x)
    let f = enrich_features(x, y)
    print(f)
"#;
    let res = check_only(code, "test_laundering_fn.nr");
    assert!(res.has_errors(), "Expected mean(labels) laundering to be rejected!");
    let has_temporal_leak = res.errors.iter().any(|e| e.code == ErrorCode::TemporalLeak);
    assert!(has_temporal_leak, "Expected ErrorCode::TemporalLeak in errors: {:?}", res.errors);
    let has_global_sample_mention = res.errors.iter().any(|e| e.message.contains("global_sample"));
    assert!(has_global_sample_mention, "Expected error to mention global_sample filtration provenance: {:?}", res.errors);
}

#[test]
fn test_causal_rolling_aggregation_allowed() {
    let code = r#"
fn enrich_rolling(features: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 0]:
    return features.rolling(24).mean()

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let f = enrich_rolling(x)
    print(f)
"#;
    let res = check_only(code, "test_causal_rolling.nr");
    assert!(!res.has_errors(), "Expected causal rolling mean to compile cleanly, but got errors: {:?}", res.errors);
}

#[test]
fn test_causal_expanding_aggregation_allowed() {
    let code = r#"
fn enrich_expanding(features: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 0]:
    return features.expanding().mean()

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let f = enrich_expanding(x)
    print(f)
"#;
    let res = check_only(code, "test_causal_expanding.nr");
    assert!(!res.has_errors(), "Expected causal expanding mean to compile cleanly, but got errors: {:?}", res.errors);
}

#[test]
fn test_centered_rolling_aggregation_rejected() {
    let code = r#"
fn enrich_centered(features: Temporal[Tensor[5000], 0]) -> Temporal[Tensor[5000], 0]:
    return features.rolling(24, center=true).mean()

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let f = enrich_centered(x)
    print(f)
"#;
    let res = check_only(code, "test_centered_rolling.nr");
    assert!(res.has_errors(), "Expected centered rolling mean to be rejected as a future leak!");
    let has_temporal_leak = res.errors.iter().any(|e| e.code == ErrorCode::TemporalLeak);
    assert!(has_temporal_leak, "Expected ErrorCode::TemporalLeak in errors: {:?}", res.errors);
}

#[test]
fn test_snapshot_escape_hatch_rejected() {
    let code = r#"
fn declassify_leak(labels: Temporal[Tensor[5000], 3]) -> Tensor[5000]:
    // Attempting to bypass temporal type safety by stripping to raw tensor
    return labels.snapshot()

fn main():
    let x: Temporal[Tensor[5000], 0] = zeros(5000)
    let y = x.shift(3)
    let raw = declassify_leak(y)
    print(raw)
"#;
    let res = check_only(code, "test_snapshot_leak.nr");
    assert!(res.has_errors(), "Expected snapshot escape hatch on future data to be rejected!");
    let has_temporal_leak = res.errors.iter().any(|e| e.code == ErrorCode::TemporalLeak);
    assert!(has_temporal_leak, "Expected ErrorCode::TemporalLeak on snapshot() escape hatch: {:?}", res.errors);
}


