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

set -e

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}
source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=riscv32_generic

readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} third_party_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# check that the release build is ok.
readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}
readable_run make -j8 -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} BUILD_TYPE=release build TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}

# Next, build w/o release so that we can run the tests and get additional
# debugging info on failures.
readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}
readable_run make -j8 -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} BUILD_TYPE=debug test TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}
