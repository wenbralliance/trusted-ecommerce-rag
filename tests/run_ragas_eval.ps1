param(
    [int]$Limit = 10,
    [int]$Workers = 2,
    [string]$Metrics = "faithfulness,answer_relevancy",
    [switch]$SkipGenerate
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mainPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$ragasPython = Join-Path $projectRoot ".venv-ragas\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env"

Set-Location $projectRoot

if (-not (Test-Path -LiteralPath $mainPython)) {
    throw "Main environment not found. Run 'uv sync --dev' first."
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw ".env is missing. Create it from .env.example."
}

if (-not $SkipGenerate) {
    $milvus = New-Object Net.Sockets.TcpClient
    try {
        $milvus.Connect("127.0.0.1", 19530)
    }
    catch {
        throw "Milvus is not available on 127.0.0.1:19530. Start it before generating samples."
    }
    finally {
        $milvus.Close()
    }

    & $mainPython tests\generate_ragas_dataset.py --limit $Limit --workers $Workers
    if ($LASTEXITCODE -ne 0) {
        throw "Ragas sample generation failed."
    }
}

if (-not (Test-Path -LiteralPath $ragasPython)) {
    uv venv .venv-ragas --python 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the isolated Ragas environment."
    }
    uv pip install --python $ragasPython -r requirements-ragas.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Ragas dependencies."
    }
}

$env:RAGAS_DO_NOT_TRACK = "true"
& $ragasPython tests\evaluate_ragas.py --metrics $Metrics
if ($LASTEXITCODE -ne 0) {
    throw "Ragas evaluation failed."
}
