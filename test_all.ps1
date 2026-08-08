$examples = @(
    "stress_test.nr",
    "transformer.nr",
    "training_demo.nr",
    "simple_shapes.nr",
    "demo_causal.nr",
    "demo_forget.nr",
    "agi_agent.nr",
    "clinical_trial.nr",
    "autonomous_vehicle.nr",
    "micro_gpt.nr",
    "demo_million_dollar_bug.nr",
    "demo_million_dollar_safe.nr",
    "smc_live_demo.nr",
    "clone_robotics_demo.nr",
    "test_while.nr"
)

foreach ($ex in $examples) {
    $path = "examples/$ex"
    Write-Host "--- TESTING: $ex ---"
    $result = & cargo run --release --bin neuronc -- run $path 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "RESULT: PASS"
    } else {
        # Extract the error line
        $errLines = $result | Select-String -Pattern "ERROR|error|panic|not found" | Select-Object -First 2
        Write-Host "RESULT: FAIL (exit $exitCode)"
        foreach ($line in $errLines) {
            Write-Host "  $($line.Line.Trim())"
        }
    }
    Write-Host ""
}
