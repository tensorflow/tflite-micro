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
    line=$(sed -e 's/"/\\"/g' <<< "${line}")
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
  result=$(python3 \
    "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/benchmarking/analyze_model.py" \
    --model_file="${MODEL_FILE}" \
    )
  substitute_strings model_analysis_strings "${result}"

  result=$(shasum -b "${MODEL_FILE}" | cut -f 1 -d ' ')
  substitute_strings model_sha1_strings "${result}"
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
if [[ "${XTENSA_CORE}" ]]; then
  result+=$(printf '\nXTENSA_CORE=%s\n' "${XTENSA_CORE}")
fi
substitute_strings target_info_strings "${result}"

if [[ ${OPTIMIZED_KERNEL} == "cmsis_nn" ]]; then
  search_file="${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/ext_libs/cmsis_nn_download.sh" 
  zip_prefix_nn=$(sed -rn \
    's/^[[:space:]]+ZIP_PREFIX_NN="([a-f0-9]+)"$/\1/p' \
    "${search_file}")
  cmsis_nn_url=$(grep -E 'CMSIS_NN_URL=".+"' "${search_file}")
  cmsis_nn_url="${cmsis_nn_url/'${ZIP_PREFIX_NN}'/${zip_prefix_nn}}"
  # trim leading whitespace
  cmsis_nn_url=$(sed -e 's/^[[:space:]]*//' <<< "${cmsis_nn_url}")
  cmsis_nn_md5=$(grep -E 'CMSIS_NN_MD5=".+"' "${search_file}")
  # trim leading whitespace
  cmsis_nn_md5=$(sed -e 's/^[[:space:]]*//' <<< "${cmsis_nn_md5}")
  result=$(printf '%s\n%s' "${cmsis_nn_url}" "${cmsis_nn_md5}")
  substitute_strings cmsis_nn_info_strings "${result}"
fi
