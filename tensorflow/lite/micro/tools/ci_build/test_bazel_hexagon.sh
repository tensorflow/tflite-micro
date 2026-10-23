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

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

# Ensure bazelisk is available in the container
if ! command -v bazel &> /dev/null && ! command -v bazelisk &> /dev/null; then
  curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /tmp/bazelisk
  chmod +x /tmp/bazelisk
  BAZEL_BIN=/tmp/bazelisk
else
  BAZEL_BIN=${BAZEL:-$(which bazelisk 2>/dev/null || which bazel 2>/dev/null || echo "bazel")}
fi

BAZEL_FLAGS="--config=_ci_base"

# Ensure Bazel server shuts down cleanly when run inside ephemeral containers
trap '${BAZEL_BIN} shutdown >/dev/null 2>&1 || true' EXIT

# Run Hexagon DSP (v66) smoke test, core smoke tests, and operator kernel suite in a unified invocation
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=hexagon-env \
  //tensorflow/lite/micro/tools/toolchains/hexagon:smoke_test_hexagon \
  //tensorflow/lite/micro:core_hexagon_smoke_tests \
  //tensorflow/lite/micro/kernels:kernel_hexagon_tests
