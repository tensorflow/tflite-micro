#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Syncs the minimal pure-C TensorFlow Lite Opaque API headers from upstream
# TensorFlow into third_party/tflite_c_headers.
#
# Usage:
#   ./third_party/tflite_c_headers/sync.sh [--commit <git-ref>] [--tf-dir <path>]

set -euo pipefail

# Default pinned TensorFlow release (v2.20.0).
DEFAULT_TF_COMMIT="72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa"
TF_COMMIT="${DEFAULT_TF_COMMIT}"
TF_COMMIT_EXPLICIT=0
TF_DIR=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--commit <git-ref>] [--tf-dir <path>]

Options:
  --commit <git-ref>  TensorFlow commit SHA, tag, or branch to sync from
                      (default: ${DEFAULT_TF_COMMIT} / v2.20.0).
  --tf-dir <path>     Local TensorFlow git checkout to read files from instead
                      of downloading from GitHub.
  -h, --help          Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      TF_COMMIT="$2"
      TF_COMMIT_EXPLICIT=1
      shift 2
      ;;
    --tf-dir)
      TF_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All files keep their exact upstream paths and are copied verbatim.
HEADERS=(
  "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
  "tensorflow/lite/builtin_ops.h"
  "tensorflow/lite/c/c_api.h"
  "tensorflow/lite/c/c_api_opaque.h"
  "tensorflow/lite/c/c_api_types.h"
  "tensorflow/lite/c/common.h"
  "tensorflow/lite/core/async/c/types.h"
  "tensorflow/lite/core/c/c_api.h"
  "tensorflow/lite/core/c/c_api_opaque.h"
  "tensorflow/lite/core/c/c_api_types.h"
  "tensorflow/lite/core/c/common.h"
  "tensorflow/lite/core/c/operator.h"
)

rm -rf "${DEST_DIR}/tensorflow"

fetch_file() {
  local src="$1"
  local dst="${DEST_DIR}/${src}"
  mkdir -p "$(dirname "${dst}")"

  if [[ -n "${TF_DIR}" ]]; then
    if [[ "${TF_COMMIT_EXPLICIT}" -eq 1 ]]; then
      git -C "${TF_DIR}" show "${TF_COMMIT}:${src}" > "${dst}"
    else
      cp "${TF_DIR}/${src}" "${dst}"
    fi
  else
    local url="https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_COMMIT}/${src}"
    curl -fsSL "${url}" -o "${dst}"
  fi
}

for hdr in "${HEADERS[@]}"; do
  fetch_file "${hdr}"
done

if [[ -n "${TF_DIR}" && "${TF_COMMIT_EXPLICIT}" -eq 0 ]]; then
  RESOLVED_REF="$(git -C "${TF_DIR}" rev-parse HEAD 2>/dev/null || echo "working-tree")"
  echo "Synced ${#HEADERS[@]} TFLite C headers from ${TF_DIR} (${RESOLVED_REF})."
else
  echo "Synced ${#HEADERS[@]} TFLite C headers at ${TF_COMMIT}."
fi
