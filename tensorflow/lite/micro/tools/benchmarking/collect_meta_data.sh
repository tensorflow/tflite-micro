#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# Collect generic benchmark meta data and insert resulting strings into
# the file designated by TEMPLATE_FILE.
#
# Takes no arguments.
#
# Uses the following environment variables:
# TEMPLATE_FILE - path to the template source file
# GENERATED_FILE - path to the generated source file with substituted strings
# TENSORFLOW_ROOT - path to the root of the source tree
# MODEL_FILE - path to the .tflite model file
# CC - path to C compiler
# CXX - path to C++ compiler
# CC_FLAGS - C compiler flags
# CXX_FLAGS - C++ compiler flags
# KERNEL_OPTIMIZATION - kernel optimization flags
# CORE_OPTIMIZATION - core optimization flags
# THIRD_PARTY_KERNEL_OPTIMIZATION - third pary kernel optimization flags
# TARGET - target platform (xtensa, cortex_m_corstone_300, etc.)
# TARGET_ARCH - target architecture (hifi5, cortex-m0, etc.)
# OPTIMIZED_KERNEL - optimized kernel (xtensa, cmsis_nn, etc.)
# BUILD_TYPE - type of build (default, release, etc.)
# XTENSA_CORE - Xtensa core specification
# XTENSA_BASE - Xtensa base install directory
# XTENSA_TOOLS_VERSION - Xtensa tooling version


set -e

source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/ci_build/helper_functions.sh

function substitute_strings() {
  search="// %%%_$1_%%%"
  lines=$(fold -w 90 -s <<< "$2")
  SAVED_IFS=${IFS}
  IFS=$'\n' lines_array=( ${lines} )
  IFS=${SAVED_IFS}
  replacement=()
  for line in "${lines_array[@]}"; do
    line=$(sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' <<< "${line}")
    line=$(printf '"%s",\n    ' "${line}")
    replacement+=( "${line}" )
  done

  tempfile=$(mktemp)

  SEARCH_PATTERN="$search" REPLACEMENT_PATTERN="${replacement[@]}" awk '
    BEGIN {
        search = ENVIRON["SEARCH_PATTERN"]
        replacement = ENVIRON["REPLACEMENT_PATTERN"]
    }
    s = index($0,search) {
        $0 = substr($0,1,s-1) replacement substr($0,s+length(search))
    }
    { print }
  ' "${GENERATED_FILE}" > ${tempfile}
  mv ${tempfile} "${GENERATED_FILE}"
}

mkdir -p $(dirname ${GENERATED_FILE})
cp -p ${TEMPLATE_FILE} ${GENERATED_FILE}

# model analysis and SHA1
if [[ ${MODEL_FILE} ]]; then
  if result=$(python3 \
    "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/benchmarking/analyze_model.py" \
    --model_file="${MODEL_FILE}" 2>/dev/null); then
    substitute_strings model_analysis_strings "${result}"
  else
    substitute_strings model_analysis_strings "Model analysis not available"
  fi

  if result=$(shasum -b "${MODEL_FILE}" 2>/dev/null | cut -f 1 -d ' '); then
    substitute_strings model_sha1_strings "${result}"
  elif result=$(md5sum -b "${MODEL_FILE}" 2>/dev/null | cut -f 1 -d ' '); then
    substitute_strings model_sha1_strings "${result}"
  else
    substitute_strings model_sha1_strings "SHA1 not available"
  fi
fi

# compile date
result=$(date)
substitute_strings compilation_date_strings "${result}"

GIT_TENSORFLOW_ROOT="${TENSORFLOW_ROOT:-./}"
set +e
# Git repo commit information
result=$(cd ${GIT_TENSORFLOW_ROOT} && git rev-parse --verify HEAD)
if [[ $? != 0 ]]; then
  result="<git commit information not available>"
fi
substitute_strings git_commit_strings "${result}"

# Git repo status information
result=$(cd ${GIT_TENSORFLOW_ROOT} && git status)
if [[ $? != 0 ]]; then
  result="<git status information not available>"
fi
substitute_strings git_status_strings "${result}"
set -e

# Compiler information
result="${CC}"
substitute_strings cc_name_strings "${result}"
result=$("${CC}" --version)
substitute_strings cc_version_strings "${result}"
result="${CC_FLAGS}"
substitute_strings cc_flags_strings "${result}"

result="${CXX}"
substitute_strings cxx_name_strings "${result}"
result=$("${CXX}" --version)
substitute_strings cxx_version_strings "${result}"
result="${CXX_FLAGS}"
substitute_strings cxx_flags_strings "${result}"

result="kernel= ${KERNEL_OPTIMIZATION}"
result+="  core= ${CORE_OPTIMIZATION}"
result+="  third-party-kernel= ${THIRD_PARTY_KERNEL_OPTIMIZATION}"
substitute_strings optimization_flag_strings "${result}"

# Target information
TARGET="${TARGET:-linux}"
TARGET_ARCH="${TARGET_ARCH:-x86}"
OPTIMIZED_KERNEL="${OPTIMIZED_KERNEL:-none}"
BUILD_TYPE="${BUILD_TYPE:-default}"
result=$(printf 'TARGET=%s\nTARGET_ARCH=%s\nOPTIMIZATION=%s\nBUILD_TYPE=%s\n' \
  "${TARGET}" \
  "${TARGET_ARCH}" \
  "${OPTIMIZED_KERNEL}" \
  "${BUILD_TYPE}" \
)
if [[ ${XTENSA_CORE} ]]; then
  result+=$(printf '\nXTENSA_CORE=%s' "${XTENSA_CORE}")
  result+=$(printf '\nXTENSA_BASE=%s' "${XTENSA_BASE}")
  result+=$(printf '\nXTENSA_TOOLS_VERSION=%s' "${XTENSA_TOOLS_VERSION}")
fi
substitute_strings target_info_strings "${result}"

dep_names=()
if [[ ${OPTIMIZED_KERNEL} == "cmsis_nn" ]]; then
  dep_names+=( "cmsis_nn" )
elif [[ ${OPTIMIZED_KERNEL} == "xtensa" ]]; then
  if [[ ${TARGET_ARCH} == "vision_p6" ]]; then
    dep_names+=( "xi_tflmlib_vision_p6" )
  elif [[ ${TARGET_ARCH} == "hifi3" ]]; then
    dep_names+=( "xa_nnlib_hifi4" "ndsplib-hifi3" )
  elif [[ ${TARGET_ARCH} == "hifi4" ]]; then
    dep_names+=( "xa_nnlib_hifi4" "ndsplib-hifi4" )
  elif [[ ${TARGET_ARCH} == "hifi5" ]]; then
    dep_names+=( "xa_nnlib_hifi5" "ndsplib-hifi5" )
  fi
fi

if [[ ${#dep_names[@]} -gt 0 ]]; then
  results=$(python3 \
    "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/download_dependencies.py" \
    --print_metadata "${dep_names[@]}")
  results_url=$(sed -En 's/^LIBRARY_URL=(.*)$/\1/p' <<< "${results}")
  results_sha256=$(sed -En 's/^LIBRARY_SHA256=(.*)$/\1/p' <<< "${results}")
  substitute_strings nn_library_url_strings "${results_url}"
  substitute_strings nn_library_sha256_strings "${results_sha256}"
fi
