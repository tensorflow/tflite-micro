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
# Tests the microcontroller code using ARC platform.
# These tests require a MetaWare C/C++ Compiler.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

TARGET_ARCH=arc
TARGET=arc_custom
OPTIMIZED_KERNEL_DIR=arc_mli

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  build -j$(nproc)

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=${TARGET} \
  TARGET_ARCH=${TARGET_ARCH} \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  test -j$(nproc)
