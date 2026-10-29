#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}

ROOT_DIR="$(pwd)/${TENSORFLOW_ROOT}"

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

EXAMPLES="-e hello_world -e micro_speech -e person_detection"

MAKEFILE=${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile
make -f ${MAKEFILE} config_info TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
make -f ${MAKEFILE} clean clean_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT}

TEST_OUTPUT_DIR_CMSIS="$(mktemp -d)"

readable_run \
  python3 ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=project_generation TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}" \
  "${TEST_OUTPUT_DIR_CMSIS}" \
  ${EXAMPLES}

readable_run \
  cp ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR_CMSIS}"

pushd "${TEST_OUTPUT_DIR_CMSIS}" >/dev/null

PATH="${PATH}:${ROOT_DIR}tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin" \
  readable_run \
  make $(get_parallel_jobs) BUILD_TYPE=cmsis_nn TENSORFLOW_ROOT=${TENSORFLOW_ROOT}

popd >/dev/null

rm -rf "${TEST_OUTPUT_DIR_CMSIS}"

# Test that tflm tree creation works even from inside TENSORFLOW_ROOT directory.
pushd "${ROOT_DIR}" >/dev/null
TEST_OUTPUT_DIR="$(mktemp -d)"
readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=cortex-m4" \
  "${TEST_OUTPUT_DIR}"
rm -rf "${TEST_OUTPUT_DIR}"
popd >/dev/null
