$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker-compose.yml"

Set-Location $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found."
}

docker compose -f $composeFile down --remove-orphans
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stop the Docker Compose application."
}
