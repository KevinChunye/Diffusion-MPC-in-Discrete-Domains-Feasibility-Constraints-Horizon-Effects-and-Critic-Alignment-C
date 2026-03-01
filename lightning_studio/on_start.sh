#!/usr/bin/env bash
set -euo pipefail

# This script runs every time a Lightning Studio starts.
# Logs: ~/.lightning_studio/logs/

REPO_SSH="${DIFFUSION_TETRIS_REPO_SSH:-git@github.com:KevinChunye/Diffusion-Tetris.git}"
REPO_DIR="${DIFFUSION_TETRIS_REPO_DIR:-$HOME/Diffusion-Tetris}"
ART_ROOT="${TETRIS_ARTIFACT_ROOT:-/teamspace/studios/tetris_artifacts}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keyscan -H github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 600 "$HOME/.ssh/known_hosts" 2>/dev/null || true

if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --all --prune || true
else
  git clone "$REPO_SSH" "$REPO_DIR" || true
fi

mkdir -p "$ART_ROOT"/{runs,datasets,checkpoints}
echo "[on_start] repo=$REPO_DIR"
echo "[on_start] artifact_root=$ART_ROOT"
