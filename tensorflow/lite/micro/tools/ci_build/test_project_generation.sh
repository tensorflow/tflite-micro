#!/usr/bin/env bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../../../.."
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# First, we test that create_tflm_tree without any examples can be used to build a
# static library.
TEST_OUTPUT_DIR="$(mktemp -d)"

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  "${TEST_OUTPUT_DIR}"

readable_run cp tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR}"
pushd "${TEST_OUTPUT_DIR}" > /dev/null
readable_run make -j8 libtflm
popd > /dev/null

rm -rf "${TEST_OUTPUT_DIR}"

# Next, we test that create_tflm_tree can be used to build example binaries.
EXAMPLES="-e hello_world -e magic_wand -e micro_speech -e person_detection"

TEST_OUTPUT_DIR="$(mktemp -d)"

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  "${TEST_OUTPUT_DIR}" \
  ${EXAMPLES}

# Confirm that print_src_files and print_dest_files output valid paths (and
# nothing else).
set +x
FILES="$(python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
           ${TEST_OUTPUT_DIR} \
           --print_src_files --print_dest_files --no_copy)"

readable_run ls ${FILES} > /dev/null

# Next, make sure that the output tree has all the files needed buld the
# examples.
readable_run cp tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR}"
pushd "${TEST_OUTPUT_DIR}" > /dev/null
readable_run make -j8 examples
popd > /dev/null

rm -rf "${TEST_OUTPUT_DIR}"

# Remove existing state prior to testing project generation for cortex-m target.
make -f tensorflow/lite/micro/tools/make/Makefile clean clean_downloads

ARM_CPU=55

TEST_OUTPUT_DIR_CMSIS="$(mktemp -d)"

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=cortex-m${ARM_CPU}" \
  "${TEST_OUTPUT_DIR_CMSIS}" \
  ${EXAMPLES}

readable_run \
  cp tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR_CMSIS}"

readable_run \
  cp tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc "${TEST_OUTPUT_DIR_CMSIS}"

readable_run \
  mkdir -p "${TEST_OUTPUT_DIR_CMSIS}/third_party/cmsis/Device/ARM/ARMCM${ARM_CPU}"

readable_run \
  cp -r tensorflow/lite/micro/tools/make/downloads/cmsis/Device/ARM/ARMCM${ARM_CPU}/Include \
    "${TEST_OUTPUT_DIR_CMSIS}/third_party/cmsis/Device/ARM/ARMCM${ARM_CPU}/"

pushd "${TEST_OUTPUT_DIR_CMSIS}" > /dev/null

PATH="${PATH}:${ROOT_DIR}/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin" \
  readable_run \
  make -j8 BUILD_TYPE=cmsis_nn TARGET_ARCH=cortex-m${ARM_CPU}

popd > /dev/null

rm -rf "${TEST_OUTPUT_DIR_CMSIS}"

# Test that C++ files are renamed to .cpp

TEST_OUTPUT_DIR_RENAME_CC="$(mktemp -d)"

readable_run \
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --rename_cc_to_cpp \
  "${TEST_OUTPUT_DIR_RENAME_CC}"

CC_FILES="$(find ${TEST_OUTPUT_DIR_RENAME_CC} -name "*.cc" | head)"
CPP_FILES="$(find ${TEST_OUTPUT_DIR_RENAME_CC} -name "*.cpp" | head)"

if test -n "${CC_FILES}"; then
  echo "Expected no .cc file to exist"
  echo "${CC_FILES}"
  exit 1;
fi

if test -z "${CPP_FILES}"; then
  echo "Expected a .cpp file to exist"
  echo "${CPP_FILES}}}"
  exit 1;
fi
