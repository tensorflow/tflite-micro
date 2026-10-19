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

# Run smoke tests across Cortex-M profiles (M0, M4, M55 Helium)
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=cortex-m0 //tensorflow/lite/micro/tools/toolchains/arm:smoke_test
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=cortex-m4 //tensorflow/lite/micro/tools/toolchains/arm:smoke_test
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=cortex-m55 //tensorflow/lite/micro/tools/toolchains/arm:smoke_test

# Run unit tests on Cortex-M55 with CMSIS-NN optimization under QEMU
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=cortex-m55-cmsis-nn \
  //tensorflow/lite/micro/testing:util_v2_test \
  //tensorflow/lite/micro/kernels:add_test \
  //tensorflow/lite/micro/kernels:conv_test
