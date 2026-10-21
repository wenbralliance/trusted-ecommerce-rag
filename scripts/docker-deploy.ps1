param(
    [switch]$NoBuild,
    [switch]$NoWait,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker-compose.yml"
$envFile = Join-Path $projectRoot ".env"

Set-Location $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install and start Docker Desktop first."
}

docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is not available. Run 'docker compose version' to check."
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $envFile
    throw "Created .env from .env.example. Fill in the API keys and run this script again."
}

$envText = Get-Content -LiteralPath $envFile -Raw
if ($envText -match "EMBEDDING_API_KEY\s*=\s*your_") {
    throw "EMBEDDING_API_KEY is still a placeholder in .env."
}
$appPort = 8000
$portMatch = [regex]::Match($envText, "(?m)^APP_PORT\s*=\s*(\d+)\s*$")
if ($portMatch.Success) {
    $appPort = [int]$portMatch.Groups[1].Value
}

# Remove a container created by the old manual docker run command.
$existing = docker ps -aq --filter "name=^/rag-container$"
if ($existing) {
    docker rm -f rag-container | Out-Null
}

$composeArgs = @(
    "compose",
    "-f",
    $composeFile,
    "up",
    "-d",
    "--force-recreate",
    "--remove-orphans"
)
if (-not $NoBuild) {
    $composeArgs += "--build"
}

& docker @composeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose deployment failed."
}

if ($NoWait) {
    Write-Host "Container started. Skipping health wait because -NoWait was supplied."
    exit 0
}

Write-Host "Waiting for rag-container health check..."
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $health = docker inspect --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" rag-container 2>$null
    if ($health -eq "healthy") {
        Write-Host "Deployment complete: http://localhost:$appPort"
        exit 0
    }
    if ($health -eq "unhealthy") {
        docker logs --tail 100 rag-container
        throw "Container became unhealthy."
    }
    Start-Sleep -Seconds 3
}

docker logs --tail 100 rag-container
throw "Health check timed out after $TimeoutSeconds seconds."
