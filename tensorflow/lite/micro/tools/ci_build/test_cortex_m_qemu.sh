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
# Called with following arguments:
# 1 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code
# Tests the microcontroller code with QEMU emulator

set -e
pwd

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}
TARGET=cortex_m_qemu
TARGET_ARCH=${3:-cortex-m3}
OPTIMIZED_KERNEL_DIR=cmsis_nn

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
  TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  clean

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  third_party_downloads

readable_run make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
  EXTERNAL_DIR=${EXTERNAL_DIR} \
  test -j$(nproc)

