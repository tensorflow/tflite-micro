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
# This script builds a TFLite micro test binary and compare the size difference
# between this binary and that same binary from the main repo.
# If the optional argument string "error_on_memory_increase" is provided as the
# script input, the script will error exit on any memory increase.
# If no argument is provided, the script produce a size comparison report.
set -e

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# Utility function to build a target and return its path back to caller through
# a global variable __BINARY_TARGET_PATH.
# The caller is expected to store this  __BINARY_TARGET_PATH back to its local
# variable if it needs to use the generated binary target with path later on.
__BINARY_TARGET_PATH=
function build_target() {
  local binary_target=$1
  local build_type=$2
  local target=$3
  local target_arch=$4
  readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
  readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile build build_type=${build_type} TARGET=${target} TARGET_ARCH=${target_arch} ${binary_target}

  # Return the relative binary with path and name.
  __BINARY_TARGET_PATH="gen/${target}_${target_arch}_${build_type}/bin/${binary_target}"
}

FLAG_ERROR_ON_MEM_INCREASE=$1
# TODO(b/196637015): change this to a real benchmark binary after the experiment
# is complete.
BENCHMARK_TARGET=binary_size_test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..

# Build a binary for the current repo
cd "${ROOT_DIR}"
# Clean once.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

build_target ${BENCHMARK_TARGET} default linux x86_64
CURRENT_BINARY=${__BINARY_TARGET_PATH}
size ${CURRENT_BINARY} > ${ROOT_DIR}/ci/size_log.txt

# Get a clone of the main repo as the reference.
REF_ROOT_DIR="$(mktemp -d ${ROOT_DIR}/../main_ref.XXXXXX)"
git clone https://github.com/tensorflow/tflite-micro.git  ${REF_ROOT_DIR}

# Build a binary for the main repo.
cd ${REF_ROOT_DIR}
build_target ${BENCHMARK_TARGET} default linux x86_64
REF_BINARY=${__BINARY_TARGET_PATH}
size ${REF_BINARY} > ${REF_ROOT_DIR}/ci/size_log.txt

# Compare the two files at th root of current repo.
cd ${ROOT_DIR}
if [ "${FLAG_ERROR_ON_MEM_INCREASE}" = "error_on_mem_increase" ]
then
  tensorflow/lite/micro/tools/ci_build/size_comp.py -a ${REF_ROOT_DIR}/ci/size_log.txt ${ROOT_DIR}/ci/size_log.txt --error_on_mem_increase
else
  tensorflow/lite/micro/tools/ci_build/size_comp.py -a ${REF_ROOT_DIR}/ci/size_log.txt ${ROOT_DIR}/ci/size_log.txt
fi