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

# Ensure FVP binary is on PATH if running in Corstone-300 docker container
if [ -d "/opt/arm/fvp/models/Linux64_GCC-9.3" ]; then
  export PATH="/opt/arm/fvp/models/Linux64_GCC-9.3:${PATH}"
fi

# Pre-baked in corstone300:0.2+; fallback kept only until workflow bump merges on main (pull_request_target).
if ! command -v bazel &> /dev/null; then
  curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /tmp/bazelisk
  chmod +x /tmp/bazelisk
  BAZEL_BIN=/tmp/bazelisk
else
  BAZEL_BIN=${BAZEL:-$(command -v bazel 2>/dev/null || echo "bazel")}
fi

BAZEL_FLAGS="--config=_ci_base"

# Ensure Bazel server shuts down cleanly when run inside ephemeral containers
trap '${BAZEL_BIN} shutdown >/dev/null 2>&1 || true' EXIT

# Run Corstone-300 + Ethos-U55 smoke test, core tests, operator kernels, and end-to-end vision model in a single unified invocation
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=corstone-env \
  //tensorflow/lite/micro/tools/toolchains/arm/corstonefvp:smoke_test_corstone300 \
  //tensorflow/lite/micro/tools/toolchains/arm/corstonefvp:corstone_exit_code_test \
  //tensorflow/lite/micro:core_corstone300_smoke_tests \
  //tensorflow/lite/micro/kernels:kernel_corstone300_tests \
  //tensorflow/lite/micro/examples/person_detection:person_detection_test_corstone300

