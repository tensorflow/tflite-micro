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

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${ROOT_DIR}"

USE_DOCKER=0
BASE_REF=""
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --docker)
      USE_DOCKER=1
      ;;
    *)
      if [ -z "${BASE_REF}" ]; then
        BASE_REF="${arg}"
      else
        PASSTHROUGH_ARGS+=("${arg}")
      fi
      ;;
  esac
done

# If --docker was requested and we are not already inside a container,
# delegate execution to run_in_docker.sh
if [[ ${USE_DOCKER} -eq 1 ]]; then
  if [[ -f /.dockerenv ]]; then
    # Already running inside Docker; do not recurse.
    :
  else
    CACHE_DIR="${HOME}/.cache/tflm_docker_bazel"
    mkdir -p "${CACHE_DIR}"
    export DOCKER_RUN_OPTS="${DOCKER_RUN_OPTS:-} -v ${CACHE_DIR}:/tmp/.cache"
    exec "${SCRIPT_DIR}/run_in_docker.sh" "${BASH_SOURCE[0]}" ${BASE_REF:+"${BASE_REF}"} "${PASSTHROUGH_ARGS[@]}"
  fi
fi

if ! command -v clang-tidy > /dev/null 2>&1; then
  echo "Error: clang-tidy command not found." >&2
  if [[ ! -f /.dockerenv && -z "${GITHUB_ACTIONS:-}" ]]; then
    echo "Tip: Run with --docker to use the CI container:" >&2
    echo "  ${BASH_SOURCE[0]} --docker" >&2
  fi
  exit 1
fi

if [ -z "${BASE_REF}" ]; then
  if git rev-parse --verify upstream/main >/dev/null 2>&1; then
    BASE_REF="upstream/main"
  elif git rev-parse --verify origin/main >/dev/null 2>&1; then
    BASE_REF="origin/main"
  elif [ -n "${GITHUB_ACTIONS:-}" ]; then
    echo "Fetching origin main for differential check..."
    git fetch origin main --depth=100 2>/dev/null || true
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
      BASE_REF="origin/main"
    elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
      BASE_REF="HEAD~1"
    else
      BASE_REF="HEAD"
    fi
  elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    BASE_REF="HEAD~1"
  else
    BASE_REF="HEAD"
  fi
fi

MERGE_BASE=$(git merge-base "${BASE_REF}" HEAD 2>/dev/null || echo "${BASE_REF}")

if [ "${MERGE_BASE}" = "$(git rev-parse HEAD 2>/dev/null)" ] && [ -z "$(git status --porcelain)" ]; then
  echo "HEAD is at ${BASE_REF} with clean working directory. No changes to check."
  exit 0
fi

# Generate compilation database if not present, invalid, or external symlink broken
REGEN_DB=0
if [ ! -f "compile_commands.json" ]; then
  REGEN_DB=1
elif ! grep -q "\"directory\": \"${ROOT_DIR}\"" compile_commands.json 2>/dev/null; then
  echo "compile_commands.json was generated for a different directory. Regenerating..."
  REGEN_DB=1
elif [ ! -e "external" ]; then
  echo "external symlink is missing or broken. Regenerating..."
  REGEN_DB=1
fi

if [ "${REGEN_DB}" -eq 1 ]; then
  echo "Generating compilation database..."
  python3 "${SCRIPT_DIR}/generate_compile_commands.py"
fi

echo "Running clang-tidy diff against ${BASE_REF} (merge-base: ${MERGE_BASE})..."
git diff -U0 --diff-filter=d "${MERGE_BASE}" -- \
    tensorflow/lite/micro signal \
    ':(exclude)*.inc' \
    ':(exclude)*schema_generated.h' \
    ':(exclude)*metadata_saved.h' \
    ':(exclude)*/downloads/*' \
    ':(exclude)*_model_data.h' \
    ':(exclude)*_test_data.h' \
    ':(exclude)tensorflow/lite/micro/examples/*' \
    ':(exclude)tensorflow/lite/micro/integration_tests/*' \
    ':(exclude)tensorflow/lite/micro/kernels/arc_mli/*' \
    ':(exclude)tensorflow/lite/micro/kernels/ceva/*' \
    ':(exclude)tensorflow/lite/micro/kernels/cmsis_nn/*' \
    ':(exclude)tensorflow/lite/micro/kernels/ethos_u/*' \
    ':(exclude)tensorflow/lite/micro/kernels/ethosu.*' \
    ':(exclude)tensorflow/lite/micro/kernels/xtensa/*' | \
  clang-tidy-diff.py -p1 -path . -iregex '.*\.(cpp|cc|c\+\+|cxx|c|h|hpp)$'
