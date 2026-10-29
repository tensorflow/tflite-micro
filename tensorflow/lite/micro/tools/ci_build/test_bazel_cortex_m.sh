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

# Run Cortex-M smoke tests (M0, M4, M55), reference suites, and CMSIS-NN Helium MVE suites in a unified invocation
${BAZEL_BIN} test ${BAZEL_FLAGS} \
  //tensorflow/lite/micro/tools/toolchains/arm:smoke_test_cortex_m0 \
  //tensorflow/lite/micro/tools/toolchains/arm:smoke_test_cortex_m4 \
  //tensorflow/lite/micro/tools/toolchains/arm:smoke_test_cortex_m55 \
  //tensorflow/lite/micro/tools/toolchains/arm:qemu_exit_code_test \
  //tensorflow/lite/micro:core_cortex_m55_ref_tests \
  //tensorflow/lite/micro/kernels:kernel_cortex_m55_ref_tests \
  //tensorflow/lite/micro:core_cortex_m55_tests \
  //tensorflow/lite/micro/kernels:kernel_cortex_m55_tests
