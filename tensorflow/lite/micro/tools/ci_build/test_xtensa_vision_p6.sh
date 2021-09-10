#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

# TODO(b/143904317): downloading first to allow for parallel builds.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=xtensa \
  TARGET_ARCH=vision_p6 \
  OPTIMIZED_KERNEL_DIR=xtensa \
  XTENSA_CORE=P6_200528 \
  build -j$(nproc)


# Since we currently do not have optimized kernel implementations for vision_p6,
# running the tests (in particular person_detection_int8) takes a very long
# time. So, we have changed the default for this script to only perform a build
# and added an option to run all the tests when that is feasible.
if [[ ${1} == "RUN_TESTS" ]]; then
  readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=vision_p6 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=P6_200528 \
    test -j$(nproc)
fi
