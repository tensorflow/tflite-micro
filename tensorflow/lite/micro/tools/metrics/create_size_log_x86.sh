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
 
# Utility function to build a target.
# Parameters:
# ${1}: binary target name such as keyworkd_benchmark
# ${2}: build type such as RELEASE, DEFAULT
# ${3}: target such as linux
# ${4}: target architecture such as x86_64
function build_target() {
  local binary_target=$1
  local build_type=$2
  local target=$3
  local target_arch=$4
  readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile build BUILD_TYPE=${build_type} TARGET=${target} TARGET_ARCH=${target_arch} ${binary_target}
}

# Utility function to profile a binary and report its size
#Parameters:
# ${1}: binary target path
# ${2}: size log file name
function profile_a_binary() {
  local binary=${1}
  local log=${2}

  raw_size=$(size ${binary})
  # Skip the title row
  sizes=$(echo "${raw_size}" | sed -n '2 p')
  text_size=$(echo "$sizes" | awk '{print $1}')
  data_size=$(echo "$sizes" | awk '{print $2}')
  bss_size=$(echo "$sizes" | awk '{print $3}')
  total_size=$(echo "$sizes" | awk '{print $4}')

  echo "${BUILD_TIME}, ${HEAD_SHA}, ${text_size}, ${data_size}, ${bss_size}, ${total_size}" >> ${log}
}

# Parameters:
# ${1} - size log file name
function start_size_report() {
  local log=${1}  

  if [[ ! -f ${log} ]]
  then
    echo "${CSV_HEADER}" >> ${log}
  fi
}

# Parameters:
# ${1}: binary target name such as keyworkd_benchmark
# ${2}: build type such as RELEASE, DEFAULT
# ${3}: target such as linux
# ${4}: target architecture such as x86_64
function report_size() {
  local binary=$1
  local build_type=$2
  local target=$3
  local target_arch=$4
  local log="${LOG_ROOT_DIR}/${binary}.csv"

  start_size_report ${log}

  build_target ${binary} ${build_type} ${target} ${target_arch}
  local build_result=$?

  if [[ ${build_result} != 0 ]]
  then
    # Here release build failed so mark failures and return appropriate error
    # code.
    echo "${binary} fail to build,">> ${log}
    return ${build_result}
  fi

  # If build is successful, profile the size.
  local binary_path="${GEN_FILES_DIR}/${target}_${target_arch}_${build_type}/bin/${binary}"
  profile_a_binary ${binary_path} ${log}
}

###################################################
### Start of main
###################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
GEN_FILES_DIR=${ROOT_DIR}/tensorflow/lite/micro/tools/make/gen/

cd "${ROOT_DIR}"
source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

CSV_HEADER="date, sha, text, data, bss, totoal"
HEAD_SHA=`git rev-parse HEAD`
BUILD_TIME=`date`
TARGET="linux"
TARGET_ARCH="x86_64"
BUILD_TYPE="release"
LOG_ROOT_DIR=${ROOT_DIR}/data/continuous_builds/size_profiling/${TARGET}_${TARGET_ARCH}_${BUILD_TYPE}

# Clean the own build and download third party
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean clean_downloads
readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

report_size keyword_benchmark ${BUILD_TYPE} ${TARGET} ${TARGET_ARCH}
KEYWORD_BENCHMARK_STATUS=$?

report_size baseline_memory_footprint ${BUILD_TYPE} ${TARGET} ${TARGET_ARCH}
BASELINE_MEMORY_FOOTPRINT_STATUS=$?

report_size interpreter_memory_footprint ${BUILD_TYPE} ${TARGET} ${TARGET_ARCH}
INTERPRETER_MEMORY_FOOTPRINT_STATUS=$?

if [[ ${KEYWORD_BENCHMARK_STATUS} != 0 || ${BASELINE_MEMORY_FOOTPRINT_STATUS} != 0 || ${INTERPRETER_MEMORY_FOOTPRINT_STATUS} != 0 ]]
then
  echo "Failure in profiling."
  exit -1
fi

## TODO(b/213646558): run difference detection and also return error code if detecting large increase
echo "Profiling succeed"