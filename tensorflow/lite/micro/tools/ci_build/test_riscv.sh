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
#
# Tests the RISC-V MCU platform for the SiFive FE310.

set -ex

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}
source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=riscv32_generic

MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile
COMMON_ARGS="TARGET=${TARGET} TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}"

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} config_info

readable_run make -f ${MAKEFILE} ${COMMON_ARGS} third_party_downloads

# check that the release build is ok.
readable_run make -f ${MAKEFILE} clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}
readable_run make $(get_parallel_jobs) -f ${MAKEFILE} ${COMMON_ARGS} BUILD_TYPE=release build

# Next, build w/o release so that we can run the tests and get additional
# debugging info on failures.
readable_run make -f ${MAKEFILE} clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}
readable_run make $(get_parallel_jobs) -f ${MAKEFILE} ${COMMON_ARGS} BUILD_TYPE=debug build
readable_run make $(get_parallel_jobs) -f ${MAKEFILE} ${COMMON_ARGS} BUILD_TYPE=debug test
