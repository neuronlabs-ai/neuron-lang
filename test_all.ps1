#!/usr/bin/env pwsh
# NEURON End-to-End Verification Suite

$neuronc = "$PSScriptRoot\target\release\neuronc.exe"
if (-not (Test-Path $neuronc)) {
    Write-Host "Building neuronc release binary..." -ForegroundColor Cyan
    cargo build --release --bin neuronc
}

$runExamples = @(
    "simple_shapes.nr",
    "training_demo.nr",
    "transformer.nr",
    "demo_million_dollar_safe.nr",
    "demo_forget.nr",
    "demo_imports.nr",
    "medical_dosage.nr",
    "clinical_trial.nr",
    "autonomous_vehicle.nr",
    "advanced_data_pipeline.nr",
    "advanced_causal_rl.nr",
    "advanced_uncertainty_routing.nr",
    "micro_gpt.nr",
    "agi_agent.nr",
    "qwen_mini_transpiled.nr",
    "stress_test.nr",
    "demo_offset_temporal.nr",
    "neuro_causal_quant.nr"
)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NEURON End-to-End Verification Suite" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

foreach ($ex in $runExamples) {
    $path = "$PSScriptRoot\examples\$ex"
    Write-Host -NoNewline "  Testing run: $ex ... "
    $output = & $neuronc run $path 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PASS" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "FAIL" -ForegroundColor Red
        $failed++
        $output | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }
}

# Negative Compile Test 1: Temporal Leak Rejection
Write-Host -NoNewline "  Testing temporal leak rejection: demo_million_dollar_bug.nr ... "
$checkOutput1 = & $neuronc check "$PSScriptRoot\examples\demo_million_dollar_bug.nr" 2>&1
if ($LASTEXITCODE -ne 0 -and ($checkOutput1 -match "TemporalLeak|lookahead bias")) {
    Write-Host "PASS (Compile-time rejected as expected)" -ForegroundColor Green
    $passed++
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $failed++
}

# Negative Compile Test 2: Causal Type Mismatch Rejection
Write-Host -NoNewline "  Testing causal mismatch rejection: demo_causal.nr ... "
$checkOutput2 = & $neuronc check "$PSScriptRoot\examples\demo_causal.nr" 2>&1
if ($LASTEXITCODE -ne 0 -and ($checkOutput2 -match "CausalTypeMismatch|intervened")) {
    Write-Host "PASS (Compile-time rejected as expected)" -ForegroundColor Green
    $passed++
} else {
    Write-Host "FAIL" -ForegroundColor Red
    $failed++
}

Write-Host ""
Write-Host "----------------------------------------------------------------" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "  Total: $($passed + $failed) | Passed: $passed | Failed: $failed" -ForegroundColor Green
} else {
    Write-Host "  Total: $($passed + $failed) | Passed: $passed | Failed: $failed" -ForegroundColor Red
}
Write-Host "================================================================" -ForegroundColor Cyan

if ($failed -gt 0) {
    exit 1
}
