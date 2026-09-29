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

# Runs a command inside the TFLM CI Docker container with the repository mounted.
#
# Usage:
#   run_in_docker.sh [--image <image>] [command [args...]]
#
# Examples:
#   # Run code style checks inside container
#   run_in_docker.sh tensorflow/lite/micro/tools/ci_build/test_code_style.sh
#
#   # Fix formatting inside container
#   run_in_docker.sh tensorflow/lite/micro/tools/ci_build/test_code_style.sh --fix_formatting
#
#   # Start an interactive shell
#   run_in_docker.sh
#
# Environment Variables:
#   TFLM_DOCKER_IMAGE: Override the default container image.
#   DOCKER_RUN_OPTS:   Additional flags to pass to `docker run`.

set -e
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

DEFAULT_IMAGE="ghcr.io/tflm-bot/tflm-ci:0.6.10"
IMAGE="${TFLM_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"

if ! command -v docker > /dev/null 2>&1; then
  echo "Error: docker command not found. Please install Docker to use this script." >&2
  exit 1
fi

# Parse optional --image flag
if [[ $# -ge 2 && "$1" == "--image" ]]; then
  IMAGE="$2"
  shift 2
fi

# Determine working directory relative to ROOT_DIR
CURRENT_DIR="$(pwd)"
CONTAINER_WORKDIR="/workspace"
if command -v python3 > /dev/null 2>&1; then
  REL_DIR="$(python3 -c "import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "${CURRENT_DIR}" "${ROOT_DIR}" 2>/dev/null || true)"
  if [[ -n "${REL_DIR}" && "${REL_DIR}" != "." && ! "${REL_DIR}" =~ ^\.\. ]]; then
    CONTAINER_WORKDIR="/workspace/${REL_DIR}"
  fi
fi

# TTY allocation
DOCKER_TTY_FLAGS=""
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_FLAGS="-it"
elif [[ -t 1 ]]; then
  DOCKER_TTY_FLAGS="-t"
fi

# Default to bash if no command specified
CMD=("$@")
if [[ ${#CMD[@]} -eq 0 ]]; then
  CMD=("bash")
fi

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"

exec docker run ${DOCKER_TTY_FLAGS} --rm --init \
  ${DOCKER_RUN_OPTS:-} \
  --user "${USER_ID}:${GROUP_ID}" \
  -e USER="${USER:-$(id -un)}" \
  -e HOME=/tmp \
  -e RUFF_CACHE_DIR=/tmp/ruff_cache \
  -e GIT_CONFIG_COUNT=1 \
  -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=* \
  -v "${ROOT_DIR}":/workspace \
  -w "${CONTAINER_WORKDIR}" \
  "${IMAGE}" \
  "${CMD[@]}"
