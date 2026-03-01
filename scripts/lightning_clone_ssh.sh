#!/usr/bin/env bash
set -euo pipefail

REPO_SSH="${DIFFUSION_TETRIS_REPO_SSH:-git@github.com:KevinChunye/Diffusion-Tetris.git}"
REPO_DIR="${DIFFUSION_TETRIS_REPO_DIR:-$HOME/Diffusion-Tetris}"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# Ensure GitHub host key exists so SSH clone does not prompt interactively.
ssh-keyscan -H github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 600 "$HOME/.ssh/known_hosts" 2>/dev/null || true

if [[ ! -f "$HOME/.ssh/id_ed25519" && ! -f "$HOME/.ssh/id_rsa" ]]; then
  echo "[warn] No SSH private key found in ~/.ssh. Add your key before cloning."
fi

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[info] Repo exists at $REPO_DIR; fetching latest."
  git -C "$REPO_DIR" fetch --all --prune
else
  echo "[info] Cloning $REPO_SSH -> $REPO_DIR"
  git clone "$REPO_SSH" "$REPO_DIR"
fi

echo "[ok] Ready: $REPO_DIR"

