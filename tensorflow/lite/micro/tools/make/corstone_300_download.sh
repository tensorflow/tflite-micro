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
# Called with following arguments:
# 1 - Path to the downloads folder which is typically
#     tensorflow/lite/micro/tools/make/downloads
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

DOWNLOADED_CORSTONE_PATH=${DOWNLOADS_DIR}/corstone300

if [ -d ${DOWNLOADED_CORSTONE_PATH} ]; then
  echo >&2 "${DOWNLOADED_CORSTONE_PATH} already exists, skipping the download."
else
  UNAME_S=$(uname -s)
  UNAME_M=$(uname -m)
  if [ ${UNAME_S} == Linux ]; then
    if [ ${UNAME_M} == x86_64 ]; then
      CORSTONE_URL=https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz
      EXPECTED_MD5=42500e49a4b9e8e0f633d1bad9b7c052
    elif [ ${UNAME_M} == aarch64 ]; then
      CORSTONE_URL=https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz
      EXPECTED_MD5=89904e875c863235635e1570c4f6459e
    else
          echo "Cpu type ${UNAME_M} with OS type ${UNAME_S} not supported."
          exit 1
    fi
  else
    echo "OS type ${UNAME_S} not supported."
    exit 1
  fi

  TEMPFILE=$(mktemp -d)/temp_file
  wget ${CORSTONE_URL} -O ${TEMPFILE} >&2
  check_md5 ${TEMPFILE} ${EXPECTED_MD5}

  TEMPDIR=$(mktemp -d)
  tar -C ${TEMPDIR} -xvzf ${TEMPFILE} >&2
  mkdir ${DOWNLOADED_CORSTONE_PATH}
  ${TEMPDIR}/FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --no-interactive -d ${DOWNLOADED_CORSTONE_PATH} >&2
fi

echo "SUCCESS"
