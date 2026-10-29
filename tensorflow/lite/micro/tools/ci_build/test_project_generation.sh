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
set -ex

TENSORFLOW_ROOT=${1}
EXTERNAL_DIR=${2}

ROOT_DIR="$(pwd)/${TENSORFLOW_ROOT}"

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# Test that create_tflm_tree can be used to build example binaries. We
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

readable_run ls ${FILES} >/dev/null

# Next, make sure that the output tree has all the files needed build the
# examples.
readable_run cp ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/project_generation/Makefile "${TEST_OUTPUT_DIR}"
pushd "${TEST_OUTPUT_DIR}" >/dev/null
readable_run make $(get_parallel_jobs) examples TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
readable_run make clean
readable_run make $(get_parallel_jobs) examples CXX=g++ CC=gcc DISABLE_EXCEPTIONS=false TENSORFLOW_ROOT=${TENSORFLOW_ROOT}
popd >/dev/null

rm -rf "${TEST_OUTPUT_DIR}"

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
  exit 1
fi

if test -z "${CPP_FILES}"; then
  echo "Expected a .cpp file to exist"
  echo "${CPP_FILES}"
  exit 1
fi

rm -rf "${TEST_OUTPUT_DIR_RENAME_CC}"
