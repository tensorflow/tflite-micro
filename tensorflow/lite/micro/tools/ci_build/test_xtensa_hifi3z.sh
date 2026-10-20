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
# 1 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) TEST_MODE: EXTERNAL (default) or INTERNAL to signal how to run the script.
# 3 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code
# Also supports legacy parameter order: [TEST_MODE] [TENSORFLOW_ROOT] [EXTERNAL_DIR].

set -ex
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

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile
COMMON_ARGS="TARGET=xtensa TARGET_ARCH=hifi3 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=P29A_HiFi3z TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}"

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} config_info

# optional command line parameter "INTERNAL" uses internal test code
if [[ ${TEST_MODE} == "INTERNAL" ]]; then
  readable_run make -f ${MAKEFILE} ${COMMON_ARGS} $(get_parallel_jobs) build
  readable_run make -f ${MAKEFILE} ${COMMON_ARGS} $(get_parallel_jobs) test

  readable_run make -f ${MAKEFILE} \
    ${COMMON_ARGS} \
    $(get_parallel_jobs) test_integration_tests_seanet_conv_test

  readable_run make -f ${MAKEFILE} \
    ${COMMON_ARGS} \
    $(get_parallel_jobs) test_integration_tests_seanet_add_test

  readable_run make -f ${MAKEFILE} \
    ${COMMON_ARGS} \
    $(get_parallel_jobs) test_integration_tests_seanet_leaky_relu_test
else
  BENCHMARK_ARGS="${COMMON_ARGS} GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite"
  readable_run make -f ${MAKEFILE} ${BENCHMARK_ARGS} $(get_parallel_jobs) build
  readable_run make -f ${MAKEFILE} ${BENCHMARK_ARGS} $(get_parallel_jobs) test

  # run generic benchmark
  readable_run make -f ${MAKEFILE} ${BENCHMARK_ARGS} $(get_parallel_jobs) run_tflm_benchmark
fi
