#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"
BUILD_FLAG="--build"
WAIT_SECONDS="${WAIT_SECONDS:-120}"

while (($#)); do
  case "$1" in
    --no-build)
      BUILD_FLAG=""
      ;;
    --no-wait)
      WAIT_SECONDS=0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

cd "$PROJECT_ROOT"

command -v docker >/dev/null || {
  echo "Docker CLI was not found. Install Docker first." >&2
  exit 1
}

docker compose version >/dev/null

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
  echo "Created .env from .env.example. Fill in API keys and rerun." >&2
  exit 1
fi

if grep -Eq '^EMBEDDING_API_KEY[[:space:]]*=[[:space:]]*your_' "$ENV_FILE"; then
  echo "EMBEDDING_API_KEY is still a placeholder in .env." >&2
  exit 1
fi

APP_PORT="${APP_PORT:-$(sed -n 's/^APP_PORT[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$ENV_FILE" | tail -n 1)}"
APP_PORT="${APP_PORT:-8000}"

# Remove a container created by the previous manual docker run command.
if docker ps -aq --filter 'name=^/rag-container$' | grep -q .; then
  docker rm -f rag-container >/dev/null
fi

docker compose -f "$COMPOSE_FILE" up -d --force-recreate --remove-orphans $BUILD_FLAG

if ((WAIT_SECONDS <= 0)); then
  echo "Container started. Health wait skipped."
  exit 0
fi

echo "Waiting for rag-container health check..."
deadline=$((SECONDS + WAIT_SECONDS))
while ((SECONDS < deadline)); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' rag-container 2>/dev/null || true)"
  if [[ "$health" == "healthy" ]]; then
    echo "Deployment complete: http://localhost:${APP_PORT}"
    exit 0
  fi
  if [[ "$health" == "unhealthy" ]]; then
    docker logs --tail 100 rag-container
    echo "Container became unhealthy." >&2
    exit 1
  fi
  sleep 3
done

docker logs --tail 100 rag-container
echo "Health check timed out after ${WAIT_SECONDS}s." >&2
exit 1
