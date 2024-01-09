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

DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH=${DOWNLOADS_DIR}/ethos_u_core_platform

if [ -d ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} ]; then
  echo >&2 "${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} already exists, skipping the download."
else
  UNAME_S=`uname -s`
  if [ ${UNAME_S} != Linux ]; then
    echo "OS type ${UNAME_S} not supported."
    exit 1
  fi

  git clone "https://review.mlplatform.org/ml/ethos-u/ethos-u-core-platform" ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} >&2
  pushd ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} > /dev/null
  git checkout e25a89dec1cf990f3168dbd6c565e3b0d51cb151 >&2
  rm -rf .git
  create_git_repo ./
  apply_patch_to_folder ./ ../../ethos_u_core_platform.patch "TFLM patch"
  popd > /dev/null

  LINKER_PATH=${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}/targets/corstone-300

  # Run C preprocessor on linker file to get rid of ifdefs and make sure compiler is downloaded first.
  COMPILER=${DOWNLOADS_DIR}/gcc_embedded/bin/arm-none-eabi-gcc
  if [ ! -f ${COMPILER} ]; then
    RETURN_VALUE=`${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/arm_gcc_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT}`
    if [ "SUCCESS" != "${RETURN_VALUE}" ]; then
      echo "The script ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/arm_gcc_download.sh failed."
      exit 1
    fi
  fi
  ${COMPILER} -E -x c -P -o ${LINKER_PATH}/platform_parsed.ld ${LINKER_PATH}/platform.ld

fi

echo "SUCCESS"
