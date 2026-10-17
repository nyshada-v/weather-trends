$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}
Write-Host "Using Python: $python"

$summary = Join-Path $root 'reports\metrics\run_summary.json'
if (-not (Test-Path $summary)) {
    Write-Host 'No results found - building data and models (first run may take a few minutes)...'
    & $python (Join-Path $root 'scripts\run_pipeline.py')
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed with exit code $LASTEXITCODE"
    }
}

Write-Host 'Starting dashboard (press Ctrl+C to stop). Open the Local URL shown below.'
Start-Process 'http://localhost:8501'
& $python -m streamlit run (Join-Path $root 'app\dashboard.py') --server.headless true
