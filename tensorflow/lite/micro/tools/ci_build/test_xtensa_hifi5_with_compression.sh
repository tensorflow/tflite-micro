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
# 1 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) TEST_MODE: EXTERNAL (default) or INTERNAL to signal how to run the script.
# 3 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code
# Also supports legacy parameter order: [TEST_MODE] [TENSORFLOW_ROOT] [EXTERNAL_DIR].

# CI test with compression enabled for hifi5

set -ex
set -x
pwd

if [[ "${1:-}" == "EXTERNAL" || "${1:-}" == "INTERNAL" ]]; then
  TEST_MODE="${1}"
  TENSORFLOW_ROOT="${2:-}"
  EXTERNAL_DIR="${3:-}"
else
  TENSORFLOW_ROOT="${1:-}"
  TEST_MODE="${2:-EXTERNAL}"
  EXTERNAL_DIR="${3:-}"
fi
MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=xtensa
TARGET_ARCH=hifi5
OPTIMIZED_KERNEL_DIR=xtensa
XTENSA_CORE=PRD_H5_RDO_07_01_2022

COMMON_ARGS="TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR} TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} XTENSA_CORE=${XTENSA_CORE}"

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} USE_TFLM_COMPRESSION=yes config_info

readable_run make -f ${MAKEFILE} $(get_parallel_jobs) ${COMMON_ARGS} USE_TFLM_COMPRESSION=yes microlite
readable_run make -f ${MAKEFILE} $(get_parallel_jobs) ${COMMON_ARGS} USE_TFLM_COMPRESSION=yes \
  test_kernel_conv_test \
  test_kernel_depthwise_conv_test \
  test_kernel_fully_connected_test \
  test_kernel_transpose_conv_test \
  test_kernel_concatenation_test \
  test_kernel_decompress_test \
  test_micro_interpreter_test \
  test_recording_micro_allocator_test \
  test_micro_resource_variable_test

readable_run make -f ${MAKEFILE} \
  $(get_parallel_jobs) \
  ${COMMON_ARGS} \
  GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite \
  USE_TFLM_COMPRESSION=yes \
  run_tflm_benchmark
