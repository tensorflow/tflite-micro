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

# Ensure Bazel server shuts down cleanly when run inside ephemeral containers
trap '${BAZEL_BIN} shutdown >/dev/null 2>&1 || true' EXIT

# 1. Run smoke test on RISC-V 32-bit (RV32IMC)
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=riscv32 //tensorflow/lite/micro/tools/toolchains/riscv:smoke_test

# 2. Run full framework, allocator, memory planner, signal kernels, and example tests on RISC-V 32-bit
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=riscv32 //tensorflow/lite/micro:core_framework_tests

# 3. Run full operator kernel test suite on RISC-V 32-bit
${BAZEL_BIN} test ${BAZEL_FLAGS} --config=riscv32 //tensorflow/lite/micro/kernels:all
