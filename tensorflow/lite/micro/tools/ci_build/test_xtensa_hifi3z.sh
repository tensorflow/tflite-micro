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
# Called with following arguments:
# 1 - EXTERNAL or INTERNAL to signal how to run the script
# 2 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 3 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code

set -e
pwd

TENSORFLOW_ROOT=${2}
EXTERNAL_DIR=${3}

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# TODO(b/143904317): downloading first to allow for parallel builds.
readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile third_party_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# optional command line parameter "INTERNAL" uses internal test code
if [[ ${1} == "INTERNAL" ]]; then
  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    build -j$(nproc)

  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    test -j$(nproc)

  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    test_integration_tests_seanet_conv_test -j$(nproc)

  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    test_integration_tests_seanet_add_test -j$(nproc)

  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    test_integration_tests_seanet_leaky_relu_test -j$(nproc)
else
  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    build -j$(nproc)

  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    test -j$(nproc)

  # run generic benchmark
  readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
    TARGET=xtensa \
    TARGET_ARCH=hifi3 \
    OPTIMIZED_KERNEL_DIR=xtensa \
    XTENSA_CORE=HIFI_190304_swupgrade \
    TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
    EXTERNAL_DIR=${EXTERNAL_DIR} \
    GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite \
    run_tflm_benchmark -j$(nproc)
fi
