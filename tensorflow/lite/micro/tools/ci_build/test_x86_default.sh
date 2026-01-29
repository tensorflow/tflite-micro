#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# Called with following arguments:
# 1 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code
# Tests the microcontroller code using native x86 execution.
#
# This file is a subset of the tests in test_x86.sh. It is for parallelizing the test
# suite on github actions.

set -ex

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile
COMMON_ARGS="TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}"

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} config_info

readable_run make -f ${MAKEFILE} clean ${COMMON_ARGS}

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f ${MAKEFILE} third_party_downloads ${COMMON_ARGS}

# Build w/o release so that we can run the tests and get additional
# debugging info on failures.
readable_run make -f ${MAKEFILE} clean ${COMMON_ARGS}
readable_run make -s $(get_parallel_jobs) -f ${MAKEFILE} build ${COMMON_ARGS}
readable_run make -s $(get_parallel_jobs) -f ${MAKEFILE} test ${COMMON_ARGS}
readable_run make -s $(get_parallel_jobs) -f ${MAKEFILE} integration_tests ${COMMON_ARGS}

# run generic benchmark
readable_run make $(get_parallel_jobs) -f ${MAKEFILE} \
  ${COMMON_ARGS} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite \
  run_tflm_benchmark
