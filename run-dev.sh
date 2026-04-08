#!/usr/bin/env bash
set -euo pipefail

RUNTIME="${CONTAINER_RUNTIME:-docker}"

exec "$RUNTIME" run --rm -it \
  --network host \
  -v "$PWD":/workspace \
  -v "$HOME/.claw":/root/.claw \
  -w /workspace/rust \
  claw-code-dev \
  "$@"
