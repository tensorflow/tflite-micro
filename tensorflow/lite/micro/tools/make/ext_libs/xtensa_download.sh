#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# Downloads necessary to build with OPTIMIZED_KERNEL_DIR=xtensa.
#
# Called with four arguments:
# 1 - Path to the downloads folder which is typically
#     tensorflow/lite/micro/tools/make/downloads
# 2 - Xtensa variant to download for (e.g. hifi4)
#
# This script is called from the Makefile and uses the following convention to
# enable determination of sucess/failure:
#
#   - If the script is successful, the only output on stdout should be SUCCESS.
#     The makefile checks for this particular string.
#
#   - Any string on stdout that is not SUCCESS will be shown in the makefile as
#     the cause for the script to have failed.
#
#   - Any other informational prints should be on stderr.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

if [[ ${2} == "hifi4" ]]; then
  LIBRARY_URL="http://github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/xa_nnlib_hifi4_11_09_2021.zip"
  LIBRARY_DIRNAME="xa_nnlib_hifi4"
  LIBRARY_MD5="fd6445b3d281220e2f584e2adc10165d"
elif [[ ${2} == "hifi5" ]]; then
  LIBRARY_URL="http://github.com/foss-xtensa/nnlib-hifi5/raw/master/archive/xa_nnlib_hifi5_06_30.zip"
  LIBRARY_DIRNAME="xa_nnlib_hifi5"
  LIBRARY_MD5="0c832b15d27ac557fa5453c902c5662a"
elif [[ ${2} == "vision_p6" ]]; then
  LIBRARY_URL="https://github.com/foss-xtensa/tflmlib_vision/raw/main/archive/xi_tflmlib_vision_p6_22_06_29.zip"
  LIBRARY_DIRNAME="xi_tflmlib_vision_p6"
  LIBRARY_MD5="fea3720d76fdb3a5a337ace7b6081b56"
else
  echo "Attempting to download an unsupported xtensa variant: ${2}"
  exit 1
fi

LIBRARY_INSTALL_PATH=${DOWNLOADS_DIR}/${LIBRARY_DIRNAME}

if [ -d ${LIBRARY_INSTALL_PATH} ]; then
  echo >&2 "${LIBRARY_INSTALL_PATH} already exists, skipping the download."
else
  TEMPDIR="$(mktemp -d)"
  TEMPFILE="${TEMPDIR}/${LIBRARY_DIRNAME}.zip"
  wget ${LIBRARY_URL} -O "$TEMPFILE" >&2
  MD5=`md5sum "$TEMPFILE" | awk '{print $1}'`

  if [[ ${MD5} != ${LIBRARY_MD5} ]]
  then
    echo "Bad checksum. Expected: ${LIBRARY_MD5}, Got: ${MD5}"
    exit 1
  fi

  # Check if another make process has already extracted the downloaded files.
  # If so, skip extracting and patching.
  if [ -d ${LIBRARY_INSTALL_PATH} ]; then
    echo >&2 "${LIBRARY_INSTALL_PATH} already exists, skipping the extraction."
  else
    unzip -qo "$TEMPFILE" -d ${DOWNLOADS_DIR} >&2

    rm -rf "${TEMPDIR}"

    pushd "${LIBRARY_INSTALL_PATH}" > /dev/null
    chmod -R +w ./
    if [[ -f "../../ext_libs/xa_nnlib_${2}.patch" ]]; then
      create_git_repo ./
      apply_patch_to_folder ./ "../../ext_libs/xa_nnlib_${2}.patch" "TFLM patch"
    fi
  fi
fi

echo "SUCCESS"
