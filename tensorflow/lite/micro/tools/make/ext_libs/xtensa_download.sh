#!/bin/bash
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
# Downloads necessary to build with OPTIMIZED_KERNEL_DIR=xtensa.
#
# Called with four arguments:
# 1 - Path to the downloads folder which is typically
#     ${TENSORFLOW_ROOT}/tensorflow/lite/micro/tools/make/downloads
# 2 - Xtensa variant to download for (e.g. hifi4)
# 3 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
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

source ${3}tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
PATCH=""

if [[ ${2} == "hifi3" ]]; then
  LIBRARY_URL="http://github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/xa_nnlib_hifi4_09_05_2023.zip"
  LIBRARY_DIRNAME="xa_nnlib_hifi4"
  LIBRARY_MD5="2a54e056aef73a4fcffde4643998501a"
elif [[ ${2} == "hifi4" ]]; then
  LIBRARY_URL="http://github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/xa_nnlib_hifi4_09_05_2023.zip"
  LIBRARY_DIRNAME="xa_nnlib_hifi4"
  LIBRARY_MD5="2a54e056aef73a4fcffde4643998501a"
  PATCH="../../ext_libs/xa_nnlib_hifi4.patch"
elif [[ ${2} == "hifi5" ]]; then
  LIBRARY_URL="http://github.com/foss-xtensa/nnlib-hifi5/raw/master/archive/xa_nnlib_hifi5_09_05_2023.zip"
  LIBRARY_DIRNAME="xa_nnlib_hifi5"
  LIBRARY_MD5="1deb55ef200bf5dbedc70b99b02140c0"
elif [[ ${2} == "vision_p6" ]]; then
  LIBRARY_URL="https://github.com/foss-xtensa/tflmlib_vision/raw/main/archive/xi_tflmlib_vision_p6_22_06_29.zip"
  LIBRARY_DIRNAME="xi_tflmlib_vision_p6"
  LIBRARY_MD5="fea3720d76fdb3a5a337ace7b6081b56"
  PATCH="../../ext_libs/xi_tflmlib_vision_p6.patch"
else
  echo "Attempting to download an unsupported xtensa variant: ${2}"
  exit 1
fi

LIBRARY_INSTALL_PATH=${DOWNLOADS_DIR}/${LIBRARY_DIRNAME}

should_download=$(check_should_download ${DOWNLOADS_DIR})

if [[ ${should_download} == "no" ]]; then
  show_download_url_md5 ${LIBRARY_URL} ${LIBRARY_MD5}
elif [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
elif [ -d ${LIBRARY_INSTALL_PATH} ]; then
  echo >&2 "${LIBRARY_INSTALL_PATH} already exists, skipping the download."
else
  TEMPDIR="$(mktemp -d)"
  TEMPFILE="${TEMPDIR}/${LIBRARY_DIRNAME}.zip"
  wget ${LIBRARY_URL} -O "$TEMPFILE" >&2
  check_md5 "${TEMPFILE}" ${LIBRARY_MD5}

  unzip -qo "$TEMPFILE" -d ${DOWNLOADS_DIR} >&2

  rm -rf "${TEMPDIR}"

  pushd "${LIBRARY_INSTALL_PATH}" > /dev/null
  chmod -R +w ./
  if [ "${PATCH}" ]; then
    create_git_repo ./
    apply_patch_to_folder ./ ${PATCH} "TFLM patch"
  fi
fi

echo "SUCCESS"
