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
# Called with following arguments:
# 1 - EXTERNAL or INTERNAL to signal how to run the script
# 2 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 3 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code

# CI test with compression enabled for hifi3z

set -e
set -x
pwd

TENSORFLOW_ROOT=${2}
EXTERNAL_DIR=${3}
MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile 

TARGET=xtensa 
TARGET_ARCH=hifi3 
OPTIMIZED_KERNEL_DIR=xtensa
XTENSA_CORE=HIFI_190304_swupgrade

make -f ${MAKEFILE} \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    XTENSA_CORE=${XTENSA_CORE} \
    third_party_downloads  # TODO(b/143904317): first to allow parallel builds

make -f ${MAKEFILE} \
    -j$(nproc) \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    XTENSA_CORE=${XTENSA_CORE} \
    USE_TFLM_COMPRESSION=yes \
    build

make -f ${MAKEFILE} \
    -j$(nproc) \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    XTENSA_CORE=${XTENSA_CORE} \
    USE_TFLM_COMPRESSION=yes \
    test

make -f ${MAKEFILE} \
    -j$(nproc) \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    XTENSA_CORE=${XTENSA_CORE} \
    GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite \
    USE_TFLM_COMPRESSION=yes \
    run_tflm_benchmark
