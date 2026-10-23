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

TARGET_CORE="${1:-hifi5}"

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

TARGETS=(
  //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test_xtensa_fusion_f1
  //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test_xtensa_hifi3z
  //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test_xtensa_hifi5
  //tensorflow/lite/micro/tools/toolchains/xtensa:smoke_test_xtensa_vision_p6
)

if [[ "${TARGET_CORE}" == "hifi5" || "${TARGET_CORE}" == "all" ]]; then
  TARGETS+=(
    //tensorflow/lite/micro:core_xtensa_hifi5_smoke_tests
    //tensorflow/lite/micro/kernels:kernel_xtensa_hifi5_smoke_tests
  )
fi

if [[ "${TARGET_CORE}" == "hifi3z" || "${TARGET_CORE}" == "all" ]]; then
  TARGETS+=(
    //tensorflow/lite/micro:core_xtensa_hifi3z_tests
    //tensorflow/lite/micro/kernels:kernel_xtensa_hifi3z_tests
  )
fi

if [[ "${TARGET_CORE}" == "fusion_f1" || "${TARGET_CORE}" == "all" ]]; then
  TARGETS+=(
    //tensorflow/lite/micro:core_xtensa_fusion_f1_smoke_tests
    //tensorflow/lite/micro/kernels:kernel_xtensa_fusion_f1_smoke_tests
  )
fi

if [[ "${TARGET_CORE}" == "vision_p6" || "${TARGET_CORE}" == "all" ]]; then
  TARGETS+=(
    //tensorflow/lite/micro:core_xtensa_vision_p6_smoke_tests
    //tensorflow/lite/micro/kernels:kernel_xtensa_vision_p6_smoke_tests
  )
fi

${BAZEL_BIN} test ${BAZEL_FLAGS} --config=xtensa-env "${TARGETS[@]}"
