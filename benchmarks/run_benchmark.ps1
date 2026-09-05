# NEURON Prime Benchmark Runner
# Compares: NEURON VM vs Python vs Rust (compiled)
# Usage: powershell -File benchmarks/run_benchmark.ps1

$ErrorActionPreference = "SilentlyContinue"

Write-Host "`n" -NoNewline
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  NEURON Prime Hunt Benchmark Suite                   ║" -ForegroundColor Cyan
Write-Host "║  Lucas-Lehmer Mersenne Test, p ≤ 61                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$results = @{}

# --- NEURON VM ---
Write-Host "▸ Running NEURON VM..." -ForegroundColor Yellow
$neuron_start = Get-Date
$neuron_out = & cargo run -p neuronc --release -- run benchmarks/prime_hunt_neuron.nr 2>&1
$neuron_elapsed = ((Get-Date) - $neuron_start).TotalSeconds
$results["NEURON VM"] = $neuron_elapsed
Write-Host "  NEURON VM: $([math]::Round($neuron_elapsed, 4))s" -ForegroundColor Green
Write-Host ""

# --- Python ---
Write-Host "▸ Running Python..." -ForegroundColor Yellow
$python_start = Get-Date
$python_out = & python benchmarks/prime_hunt_python.py 61 2>&1
$python_elapsed = ((Get-Date) - $python_start).TotalSeconds
$results["Python"] = $python_elapsed
Write-Host "  Python: $([math]::Round($python_elapsed, 4))s" -ForegroundColor Green
Write-Host ""

# --- Rust (compile + run) ---
Write-Host "▸ Compiling Rust benchmark..." -ForegroundColor Yellow
$rust_compile_out = & rustc -O benchmarks/prime_hunt_rust.rs -o benchmarks/prime_hunt_rust.exe 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "▸ Running Rust..." -ForegroundColor Yellow
    $rust_start = Get-Date
    $rust_out = & .\benchmarks\prime_hunt_rust.exe 61 2>&1
    $rust_elapsed = ((Get-Date) - $rust_start).TotalSeconds
    $results["Rust"] = $rust_elapsed
    Write-Host "  Rust: $([math]::Round($rust_elapsed, 4))s" -ForegroundColor Green
} else {
    Write-Host "  Rust compilation failed, skipping" -ForegroundColor Red
    $rust_elapsed = 0
}
Write-Host ""

# --- NEURON JIT ---
Write-Host "▸ Running NEURON JIT..." -ForegroundColor Yellow
$jit_start = Get-Date
$jit_out = & cargo run -p neuronc --release -- jit benchmarks/prime_hunt_neuron.nr 2>&1
$jit_elapsed = ((Get-Date) - $jit_start).TotalSeconds
$results["NEURON JIT"] = $jit_elapsed
Write-Host "  NEURON JIT: $([math]::Round($jit_elapsed, 4))s" -ForegroundColor Green
Write-Host ""

# --- Results ---
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  BENCHMARK RESULTS                                   ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════════════════╣" -ForegroundColor Cyan

$sorted = $results.GetEnumerator() | Sort-Object Value
$fastest = ($sorted | Select-Object -First 1).Value

foreach ($entry in $sorted) {
    $name = $entry.Key.PadRight(15)
    $time = "$([math]::Round($entry.Value, 4))s".PadRight(12)
    if ($fastest -gt 0) {
        $ratio = [math]::Round($entry.Value / $fastest, 2)
        $bar = "█" * [math]::Min([int]($ratio * 10), 40)
        Write-Host "║  $name $time ${ratio}x  $bar" -ForegroundColor White
    } else {
        Write-Host "║  $name $time" -ForegroundColor White
    }
}

Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: NEURON times include compilation overhead." -ForegroundColor DarkGray
Write-Host "      Rust time is execution only (pre-compiled with -O)." -ForegroundColor DarkGray
Write-Host "      Python time includes interpreter startup." -ForegroundColor DarkGray
