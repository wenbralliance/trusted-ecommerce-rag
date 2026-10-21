#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

command -v docker >/dev/null || {
  echo "Docker CLI was not found." >&2
  exit 1
}

docker compose -f "$PROJECT_ROOT/docker-compose.yml" down --remove-orphans
