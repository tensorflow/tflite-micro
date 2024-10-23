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
  python3 -m pip install absl-py tensorflow
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
if [[ ${XTENSA_CORE} ]]; then
  result+=$(printf '\nXTENSA_CORE=%s' "${XTENSA_CORE}")
  result+=$(printf '\nXTENSA_BASE=%s' "${XTENSA_BASE}")
  result+=$(printf '\nXTENSA_TOOLS_VERSION=%s' "${XTENSA_TOOLS_VERSION}")
fi
substitute_strings target_info_strings "${result}"

download_scripts=()
download_script_args=( "--no-downloads" )
if [[ ${OPTIMIZED_KERNEL} == "cmsis_nn" ]]; then
  download_scripts+=( "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/ext_libs/cmsis_nn_download.sh" )
  download_script_args+=( "${TENSORFLOW_ROOT}" )
elif [[ ${OPTIMIZED_KERNEL} == "xtensa" ]]; then
  download_script_args+=( "${TARGET_ARCH}" "${TENSORFLOW_ROOT}" )
  if [[ ${TARGET_ARCH} =~ ^(vision_p6)$ ]]; then
    download_scripts+=( "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/ext_libs/xtensa_download.sh" )
  elif [[ ${TARGET_ARCH} =~ ^(hifi3|hifi4|hifi5)$ ]]; then
    download_scripts+=( "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/ext_libs/xtensa_download.sh" )
    download_scripts+=( "${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/ext_libs/xtensa_ndsp_download.sh" )
  fi
fi

if [[ ${#download_scripts[@]} -gt 0 ]]; then
  results_url=
  results_md5=
  for script in "${download_scripts[@]}"; do
    results=$("${script}" "${download_script_args[@]}" 2>&1)
    url=$(sed -rn 's/^LIBRARY_URL=(.*)$/\1/p' <<< "${results}")
    results_url+=$(printf '\n%s' "${url}")
    md5=$(sed -rn 's/^LIBRARY_MD5=(.*)$/\1/p' <<< "${results}")
    results_md5+=$(printf '\n%s' "${md5}")
  done
  substitute_strings nn_library_url_strings "${results_url}"
  substitute_strings nn_library_md5_strings "${results_md5}"
fi
