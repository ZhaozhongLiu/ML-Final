#!/usr/bin/env bash
set -euo pipefail

# Starts pico-llm part2 in a detachable session (screen or tmux), so SSH disconnects won't kill the run.
# It also pins RUN_TAG so you can find outputs/logs deterministically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART2_DIR="${ROOT_DIR}/pico-llm/part2"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d-%H%M%S")}"
SESSION_NAME="${SESSION_NAME:-pico-part2-${RUN_TAG}}"

echo "[part2] starting detached run"
echo "[part2] RUN_TAG=${RUN_TAG}"
echo "[part2] SESSION_NAME=${SESSION_NAME}"
echo "[part2] run_all: ${PART2_DIR}/run_all.sh"

cmd="cd \"${ROOT_DIR}\" && RUN_TAG=\"${RUN_TAG}\" bash \"${PART2_DIR}/run_all.sh\""

if command -v screen >/dev/null 2>&1; then
  screen -dmS "${SESSION_NAME}" bash -lc "${cmd}"
  echo "[part2] started screen session: ${SESSION_NAME}"
  echo "[part2] attach with: screen -r \"${SESSION_NAME}\""
  exit 0
fi

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${SESSION_NAME}" bash -lc "${cmd}"
  echo "[part2] started tmux session: ${SESSION_NAME}"
  echo "[part2] attach with: tmux attach -t \"${SESSION_NAME}\""
  exit 0
fi

echo "[part2] WARN: neither screen nor tmux is installed; falling back to nohup."
nohup bash -lc "${cmd}" >/dev/null 2>&1 &
echo "[part2] started with nohup; find outputs under: ${PART2_DIR}/runs/${RUN_TAG}/"

