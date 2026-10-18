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
# Verifies that checked-in FlatBuffers generated files (used by the Make build)
# are in sync with their .fbs schemas. Pass --fix to update them in place.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${ROOT_DIR}"

USE_DOCKER=0
FIX_MODE=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --docker)
      USE_DOCKER=1
      ;;
    --fix|-f|--update)
      FIX_MODE=1
      PASSTHROUGH_ARGS+=("${arg}")
      ;;
    *)
      PASSTHROUGH_ARGS+=("${arg}")
      ;;
  esac
done

if [[ ${USE_DOCKER} -eq 1 && ! -f /.dockerenv ]]; then
  CACHE_DIR="${HOME}/.cache/tflm_docker_bazel"
  mkdir -p "${CACHE_DIR}"
  export DOCKER_RUN_OPTS="${DOCKER_RUN_OPTS:-} -v ${CACHE_DIR}:/tmp/.cache"
  exec "${SCRIPT_DIR}/run_in_docker.sh" "${BASH_SOURCE[0]}" "${PASSTHROUGH_ARGS[@]}"
fi

TARGETS=(
  "//tensorflow/lite/micro/compression:metadata_cc_srcs"
  "//tensorflow/lite/micro/schema:schema_fbs_srcs"
  "//tensorflow/lite/micro/tools:layer_by_layer_schema_srcs"
  "//tensorflow/lite/micro/python:schema_py_srcs"
)

FILES=(
  "tensorflow/lite/micro/compression/metadata_generated.h"
  "tensorflow/lite/micro/schema/schema_generated.h"
  "tensorflow/lite/micro/tools/layer_by_layer_schema_generated.h"
  "tensorflow/lite/micro/python/schema_py_generated.py"
)

BAZEL_CONFIG="${BAZEL_CONFIG:-ci}"
bazel build --config="${BAZEL_CONFIG}" "${TARGETS[@]}"
BAZEL_BIN="$(bazel info --config="${BAZEL_CONFIG}" bazel-bin)"

FAILED=0
for f in "${FILES[@]}"; do
  if [[ ${FIX_MODE} -eq 1 ]]; then
    cp "${BAZEL_BIN}/${f}" "${f}"
    chmod 644 "${f}"
    echo "Updated ${f}"
  else
    if ! diff -u "${f}" "${BAZEL_BIN}/${f}"; then
      echo "ERROR: ${f} is out of sync with its FlatBuffers schema."
      FAILED=1
    fi
  fi
done

if [[ ${FAILED} -ne 0 ]]; then
  echo ""
  echo "Checked-in FlatBuffers files can be updated automatically with:"
  echo "  tensorflow/lite/micro/tools/ci_build/test_generated_files.sh --fix"
  exit 1
fi

if [[ ${FIX_MODE} -eq 0 ]]; then
  echo "PASSED: All checked-in FlatBuffers generated files are in sync."
fi
