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

BAZEL_BIN=${BAZEL:-$(which bazelisk 2>/dev/null || which bazel 2>/dev/null || echo "bazel")}
BAZEL_FLAGS="--config=_ci_base"

# Ensure host libstdc++ takes precedence over older sysroot libraries in the Xtensa container
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Ensure Xtensa tools are on PATH if installed in the standard container path
if [ -d "/opt/xtensa/XtDevTools/install/tools/RI-2022.9-linux/XtensaTools/bin" ]; then
  export PATH="/opt/xtensa/XtDevTools/install/tools/RI-2022.9-linux/XtensaTools/bin:${PATH}"
fi

# Ensure Bazel server shuts down cleanly when run inside ephemeral containers
trap '${BAZEL_BIN} shutdown >/dev/null 2>&1 || true' EXIT

# 1. Run smoke tests across all 4 Cadence Xtensa DSP cores
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-fusion-f1 //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-hifi3z //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-hifi5 //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-vision-p6 //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test

# 2. Run full framework, allocator, memory planner, signal kernels, and example tests on HiFi 3z
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-hifi3z //tensorflow/lite/micro:core_framework_tests

# 3. Run full operator kernel test suite (84 tests) on HiFi 3z
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-hifi3z //tensorflow/lite/micro/kernels:all

# 4. Run representative operator kernel tests on HiFi 5
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-hifi5 \
  //tensorflow/lite/micro:micro_utils_test \
  //tensorflow/lite/micro/kernels:add_test \
  //tensorflow/lite/micro/kernels:conv_test \
  //tensorflow/lite/micro/kernels:depthwise_conv_test \
  //tensorflow/lite/micro/kernels:mul_test \
  //tensorflow/lite/micro/kernels:pooling_test \
  //tensorflow/lite/micro/kernels:softmax_test
