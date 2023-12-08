#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# Called with following arguments:
# 1 - Path to the downloads folder which is typically
#     ${TENSORFLOW_ROOT}/tensorflow/lite/micro/tools/make/downloads
# 2 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
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

TENSORFLOW_ROOT=${2}
source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

DOWNLOADED_GCC_PATH=${DOWNLOADS_DIR}/gcc_embedded

if [ -d ${DOWNLOADED_GCC_PATH} ]; then
  echo >&2 "${DOWNLOADED_GCC_PATH} already exists, skipping the download."
else

  HOST_OS=
  if [ "${OS}" == "Windows_NT" ]; then
    HOST_OS=windows
  else
    UNAME_S=`uname -s`
    if [ "${UNAME_S}" == "Linux" ]; then
      HOST_OS=linux
    elif [ "${UNAME_S}" == "Darwin" ]; then
      HOST_OS=osx
    fi
  fi

  if [ "${HOST_OS}" == "linux" ]; then
    # host architechture
    UNAME_M=`uname -m`
    if [ "${UNAME_M}" == "x86_64" ]; then
      GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz"
      EXPECTED_MD5="791754852f8c18ea04da7139f153a5b7"
    elif [ "${UNAME_M}" == "aarch64" ]; then
      GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-aarch64-arm-none-eabi.tar.xz"
      EXPECTED_MD5="5a08122e6d4caf97c6ccd1d29e62599c"
    fi

  elif [ "${HOST_OS}" == "osx" ]; then
    # host architechture
    UNAME_M=`uname -m`
    if [ "${UNAME_M}" == "arm64" ]; then
      GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-darwin-arm64-arm-none-eabi.tar.xz"
      EXPECTED_MD5="2c43e9d72206c1f81227b0a685df5ea6"
    else
      GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-darwin-x86_64-arm-none-eabi.tar.xz"
      EXPECTED_MD5="41d49840b0fc676d2ae35aab21a58693"
    fi
  elif [ "${HOST_OS}" == "windows" ]; then
    GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-mingw-w64-i686-arm-none-eabi.zip"
    EXPECTED_MD5="7fd677088038cdf82f33f149e2e943ee"
  else
    echo "OS type ${HOST_OS} not supported."
    exit 1
  fi

  TEMPDIR=$(mktemp -d)
  TEMPFILE=${TEMPDIR}/temp_file
  wget ${GCC_URL} -O ${TEMPFILE} >&2
  check_md5 ${TEMPFILE} ${EXPECTED_MD5}

  mkdir ${DOWNLOADED_GCC_PATH}

  if [ "${HOST_OS}" == "windows" ]; then
    unzip -q ${TEMPFILE} -d ${TEMPDIR} >&2
    mv ${TEMPDIR}/*/* ${DOWNLOADED_GCC_PATH}
  else
    tar -C ${DOWNLOADED_GCC_PATH} --strip-components=1 -xJf ${TEMPFILE} >&2
  fi
  echo >&2 "Unpacked to directory: ${DOWNLOADED_GCC_PATH}"
fi

echo "SUCCESS"
