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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET=mcu_riscv

readable_run make -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} third_party_downloads

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

# Validate build for libtensorflow-microlite.a
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET}

# Make sure an example will build.
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} hello_world

# There are currently no runtime tests, we only check that compilation is working.
