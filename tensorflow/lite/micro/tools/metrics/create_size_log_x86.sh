#!/bin/bash -e
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
# Measures the size of specified binaries and append the report to a log.
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..

cd "${ROOT_DIR}"
source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET="linux"
TARGET_ARCH="x86_64"
BUILD_TYPE="release"

# Clean the own build and download third party
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean clean_downloads
readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

BINARY_LIST="keyword_benchmark,baseline_memory_footprint,interpreter_memory_footprint"
python3 tensorflow/lite/micro/tools/metrics/create_size_log.py --build_type=${BUILD_TYPE} --target=${TARGET} --target_arch=${TARGET_ARCH} --binary_list=${BINARY_LIST}
LOG_GENERATION_STATUS=$?

if [[ ${LOG_GENERATION_STATUS} != 0 ]]
then
  echo "Failure in profiling."
  exit -1
fi

echo "Success in size log generation"

LOG_DIR="${ROOT_DIR}/data/continuous_builds/size_profiling/${TARGET}_${TARGET_ARCH}_${BUILD_TYPE}"
python3 tensorflow/lite/micro/tools/metrics/detect_size_increase_and_plot_history.py --input_dir=${LOG_DIR} --output_dir=${LOG_DIR} --binary_list=${BINARY_LIST}
SIZE_ALERT_STATUS=$?

if [[ ${SIZE_ALERT_STATUS} != 0 ]]
then
  echo "Size increase may exceed threshold"
  exit -1
fi

echo "Size does not increase or size increase does not exceed threshold"