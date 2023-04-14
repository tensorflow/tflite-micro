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
# Called with following arguments:
# 1 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
# 2 - (optional) EXTERNAL_DIR: Path to the external directory that contains external code
set -e

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}

ROOT_DIR="$(pwd)/${TENSORFLOW_ROOT}"

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# TODO(b/261685878): re-enable once all the issues with the bazel project
# generation CI are sorted out.
#
# # First, we test that create_tflm_tree without any examples can be used to build a
# # static library with bazel. Bazel can help catch errors that are not caught by
# # a simple makefile (e.g. http://b/261106859).
# TEST_OUTPUT_DIR="$(mktemp -d)"
#
# # We currently run the bazel build from TENSORFLOW_ROOT.
# pushd "${ROOT_DIR}" > /dev/null
# readable_run \
#   python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
#   "${TEST_OUTPUT_DIR}"
#
# readable_run cp tensorflow/lite/micro/tools/project_generation/BUILD.testing "${TEST_OUTPUT_DIR}/BUILD"
# popd > /dev/null
#
# pushd "${TEST_OUTPUT_DIR}" > /dev/null
# readable_run touch WORKSPACE
# readable_run bazel build :libtflm
# popd > /dev/null
#
# rm -rf "${TEST_OUTPUT_DIR}"

# Next, we test that create_tflm_tree can be used to build example binaries. We
# perform this test with a Makefile (instead of bazel) because make is more
# commonly understood and because we use make for cross-compilation.
EXAMPLES="-e hello_world -e micro_speech -e person_detection"

TEST_OUTPUT_DIR="$(mktemp -d)"

readable_run \
  python3 ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}" \
  "${TEST_OUTPUT_DIR}" \
  ${EXAMPLES}

# Confirm that print_src_files and print_dest_files output valid paths (and
# nothing else).
set +x
FILES="$(python3 ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
           --makefile_options="TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}" \
           ${TEST_OUTPUT_DIR} \
           --print_src_files --print_dest_files --no_copy)"

readable_run ls ${FILES} > /dev/null

# Next, make sure that the output tree has all the files needed buld the
# examples.
readable_run cp ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR}"
pushd "${TEST_OUTPUT_DIR}" > /dev/null
readable_run make -j8 examples TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
popd > /dev/null

rm -rf "${TEST_OUTPUT_DIR}"

# Remove existing state prior to testing project generation for cortex-m target.
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile clean clean_downloads TENSORFLOW_ROOT=${TENSORFLOW_ROOT}

TEST_OUTPUT_DIR_CMSIS="$(mktemp -d)"

readable_run \
  python3 ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=project_generation TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}" \
  "${TEST_OUTPUT_DIR_CMSIS}" \
  ${EXAMPLES}

readable_run \
  cp ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR_CMSIS}"

pushd "${TEST_OUTPUT_DIR_CMSIS}" > /dev/null

PATH="${PATH}:${ROOT_DIR}tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin" \
  readable_run \
  make -j8 BUILD_TYPE=cmsis_nn TENSORFLOW_ROOT=${TENSORFLOW_ROOT}

popd > /dev/null

rm -rf "${TEST_OUTPUT_DIR_CMSIS}"

# Test that C++ files are renamed to .cpp
TEST_OUTPUT_DIR_RENAME_CC="$(mktemp -d)"

readable_run \
  python3 ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --rename_cc_to_cpp \
  --makefile_options="TENSORFLOW_ROOT=${TENSORFLOW_ROOT} EXTERNAL_DIR=${EXTERNAL_DIR}" \
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

# Test the tflm tree creation works even inside from TENSORFLOW_ROOT directory.
pushd "${TENSORFLOW_ROOT}" > /dev/null
TEST_OUTPUT_DIR="$(mktemp -d)"
readable_run \
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
--makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=cortex-m4" \
"${TEST_OUTPUT_DIR}"
rm -rf "${TEST_OUTPUT_DIR}"
popd > /dev/null

