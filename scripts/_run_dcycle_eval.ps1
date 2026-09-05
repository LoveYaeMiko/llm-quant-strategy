# Self-healing wrapper for the D model-cycle evidence experiment.
# Docker Desktop keeps dropping on this machine; before each attempt ensure the
# PIT container is healthy, then run the (resumable) eval script. Retries.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
# Cap BLAS/OpenMP threads: the 337-formula zoo build runs OpenBLAS ops from 12
# process workers — unrestricted thread buffers OOM 32 GB machines (observed
# 2026-09-04). LightGBM sets its own num_threads independently.
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "1"
$log = "outputs\_dcycle_eval.log"
$evalArgs = $args -join " "

function Test-PitHealthy {
    $h = docker inspect --format "{{.State.Health.Status}}" fqa-pit-db 2>$null
    return ($h -eq "healthy")
}

for ($attempt = 1; $attempt -le 6; $attempt++) {
    if (-not (Test-PitHealthy)) {
        $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dd) { Start-Process $dd | Out-Null }
        $deadline = (Get-Date).AddMinutes(120)
        while (-not (Test-PitHealthy) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 10 }
    }
    if (Test-PitHealthy) {
        "attempt ${attempt} $(Get-Date -Format 'HH:mm:ss'): PIT healthy - running eval $evalArgs" | Tee-Object -FilePath $log -Append
        python scripts\d_model_cycle_eval.py $evalArgs *>> $log
        $code = $LASTEXITCODE
        if ($code -eq 0) { "attempt ${attempt}: eval completed OK" | Tee-Object -FilePath $log -Append; exit 0 }
        "attempt ${attempt}: eval exit $code - will retry" | Tee-Object -FilePath $log -Append
    } else {
        "attempt ${attempt} $(Get-Date -Format 'HH:mm:ss'): PIT not healthy after wait" | Tee-Object -FilePath $log -Append
    }
    Start-Sleep -Seconds 30
}
"all attempts failed" | Tee-Object -FilePath $log -Append
exit 1
