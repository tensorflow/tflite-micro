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

DOWNLOADED_CMSIS_DSP_PATH=${DOWNLOADS_DIR}/cmsis_dsp

if [ -d ${DOWNLOADED_CMSIS_DSP_PATH} ]; then
  echo >&2 "${DOWNLOADED_CMSIS_DSP_PATH} already exists, skipping the download."
else

  ZIP_PREFIX_DSP="540161c11527b8dc91754b5ed41cad43367f8529"
  CMSIS_DSP_URL="http://github.com/ARM-software/CMSIS-DSP/archive/${ZIP_PREFIX_DSP}.zip"
  CMSIS_DSP_MD5="c2d9339b377d6199ed09684149efe151"

  # wget is much faster than git clone of the entire repo. So we wget a specific
  # version and can then apply a patch, as needed.
  wget ${CMSIS_DSP_URL} -O /tmp/${ZIP_PREFIX_DSP}.zip >&2
  check_md5 /tmp/${ZIP_PREFIX_DSP}.zip ${CMSIS_DSP_MD5}

  unzip -qo /tmp/${ZIP_PREFIX_DSP}.zip -d /tmp >&2
  mv /tmp/CMSIS-DSP-${ZIP_PREFIX_DSP} ${DOWNLOADED_CMSIS_DSP_PATH}
fi

echo "SUCCESS"
