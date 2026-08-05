#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${CURRENT_DIR}/Assets"

HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${CURRENT_DIR}/.cache/huggingface}"

REQUIRED_ASSET_SUBDIRS=(Robots Object Material Eval_Layout)

info()  { echo -e "\033[1;32m>>> $*\033[0m"; }
warn()  { echo -e "\033[1;33m>>> $*\033[0m" >&2; }
error() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; exit 1; }

usage() {
  cat <<EOF
Download RoboDojo Assets with Hugging Face CLI.

Usage:
  bash scripts/init_assets_hf.sh

Environment overrides:
  HF_REPO_ID    default: ${HF_REPO_ID}
  HF_REVISION   default: ${HF_REVISION}
  HF_CACHE_DIR  default: ${HF_CACHE_DIR}

The files are downloaded to:
  ${TARGET_DIR}
EOF
}

check_tools() {
  command -v hf >/dev/null 2>&1 || error "hf command not found. Install with: pip install -U huggingface_hub"
  command -v git >/dev/null 2>&1 || warn "git not found; continuing because hf download does not require git."
}

assets_ready() {
  [[ -d "${TARGET_DIR}" ]] || return 1
  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    [[ -d "${TARGET_DIR}/${subdir}" ]] || return 1
  done
}

download_assets() {
  mkdir -p "${CURRENT_DIR}/.cache" "${TARGET_DIR}"

  if assets_ready; then
    warn "${TARGET_DIR} already contains all required asset directories."
    warn "hf download will still verify/download missing or stale files only."
  fi

  info "Repository: ${HF_REPO_ID}"
  info "Revision: ${HF_REVISION}"
  info "Target: ${TARGET_DIR}"
  info "Cache: ${HF_CACHE_DIR}"
  info "Downloading only Assets/**; existing local files are reused when verified by hf."

  hf download "${HF_REPO_ID}" \
    --repo-type dataset \
    --revision "${HF_REVISION}" \
    --include "Assets/**" \
    --local-dir "${CURRENT_DIR}" 
}

verify_assets() {
  [[ -d "${TARGET_DIR}" ]] || error "${TARGET_DIR} was not created."

  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    [[ -d "${TARGET_DIR}/${subdir}" ]] || \
      error "Missing asset directory: ${TARGET_DIR}/${subdir}"
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

check_tools
download_assets
verify_assets
info "Assets directory is ready: ${TARGET_DIR}"
