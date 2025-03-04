#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# Tests Arm Cortex-M55 microprocessor code with CMSIS-NN optimizied kernels using FVP based on Arm Corstone-300 software.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

if [[ $1 = "armclang" ]]; then
    TOOLCHAIN=armclang
else
    TOOLCHAIN=gcc
fi

TARGET=cortex_m_corstone_300
TARGET_ARCH=cortex-m55
OPTIMIZED_KERNEL_DIR=cmsis_nn
TOOLCHAINS=(gcc armclang)

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} TOOLCHAIN=${TOOLCHAIN} third_party_downloads

# Avoid running tests in parallel.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} TOOLCHAIN=${TOOLCHAIN} build
readable_run make -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} TOOLCHAIN=${TOOLCHAIN} test

# Run generic benchmark. Not supported for armclang - see comment in target makefile.
if [[ $1 != "armclang" ]]; then
  readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=ethos_u \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect_vela.tflite \
    GENERIC_BENCHMARK_ARENA_SIZE=`expr 150 \* 1024` \
    run_tflm_benchmark
fi
