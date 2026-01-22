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
# 2 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code

set -ex
pwd

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile

COMMON_ARGS="TARGET=xtensa TARGET_ARCH=hifi5 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=PRD_H5_RDO_07_01_2022 TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}"

readable_run make -f ${MAKEFILE} clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# TODO(b/143904317): downloading first to allow for parallel builds.
readable_run make -f ${MAKEFILE} third_party_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} $(get_parallel_jobs) build
readable_run make -f ${MAKEFILE} ${COMMON_ARGS} $(get_parallel_jobs) test

# run generic benchmark
readable_run make -f ${MAKEFILE} \
  ${COMMON_ARGS} \
  GENERIC_BENCHMARK_MODEL_PATH=${TENSORFLOW_ROOT}tensorflow/lite/micro/models/person_detect.tflite \
  $(get_parallel_jobs) run_tflm_benchmark
