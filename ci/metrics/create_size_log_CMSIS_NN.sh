#!/bin/bash -e
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
#
# Measures the size of specified binaries and append the report to a log.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]
then
  SAVE_TO_DIR=$(pwd)/report
else
  SAVE_TO_DIR=${1}
fi

ROOT_DIR=${SCRIPT_DIR}/../../
GIT_DIR=${ROOT_DIR}/../cmsis
cd "${ROOT_DIR}"
source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET="cortex_m_corstone_300"
TARGET_ARCH="cortex-m4"
BUILD_TYPE="release"

BINARY_LIST="hello_world,person_detection,keyword_benchmark,baseline_memory_footprint,interpreter_memory_footprint"
OPTIMIZED_KERNEL_DIRS=("cmsis_nn")
TOOLCHAINS=("armclang") # "gcc")

for OKD in ${OPTIMIZED_KERNEL_DIRS[@]}
do
  for TC in ${TOOLCHAINS[@]}
  do
    # Clean the own build and download third party
    readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean clean_downloads
    readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
    python3 ci/metrics/create_size_log_CMSIS_NN.py \
        --make_flags=-j4 \
        --target=$TARGET \
        --target_arch=$TARGET_ARCH \
        --build_type=$BUILD_TYPE \
        --toolchain=$TC \
        --optimized_kernel_dir=$OKD \
        --cmsis_path=../cmsis \
        --binary_list=$BINARY_LIST \
        --relative_root_dir=../../ \
        --save_top_path=$SAVE_TO_DIR \
        --git_dir=${GIT_DIR}
    LOG_GENERATION_STATUS=$?

    if [[ ${LOG_GENERATION_STATUS} != 0 ]]
    then
      echo "Failure in profiling."
      exit -1
    fi

    echo "Success in size log generation for $OKD - $TC"
  done
done
