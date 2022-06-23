#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/make/bash_helpers.sh

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

  git clone https://git.mlplatform.org/ml/ethos-u/ethos-u-core-platform.git ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} >&2
  cd ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}
  git checkout e25a89dec1cf990f3168dbd6c565e3b0d51cb151 >&2
  rm -rf .git
  create_git_repo ./

  apply_patch_to_folder ./ ../../increase-stack-size-and-switch-DTCM-SRAM.patch "TFLM patch"

  cd "${ROOT_DIR}"

  LINKER_PATH=${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}/targets/corstone-300

  # Prepend #!cpp to scatter file.
  SCATTER=${LINKER_PATH}/platform.scatter
  echo -e "#!cpp\n$(cat ${SCATTER})" > ${SCATTER}

  # Run C preprocessor on linker file to get rid of ifdefs and make sure compiler is downloaded first.
  COMPILER=${DOWNLOADS_DIR}/gcc_embedded/bin/arm-none-eabi-gcc
  if [ ! -f ${COMPILER} ]; then
    RETURN_VALUE=`./tensorflow/lite/micro/tools/make/arm_gcc_download.sh ${DOWNLOADS_DIR}`
    if [ "SUCCESS" != "${RETURN_VALUE}" ]; then
      echo "The script ./tensorflow/lite/micro/tools/make/arm_gcc_download.sh failed."
      exit 1
    fi
  fi

  ${COMPILER} -E -x c -P -o ${LINKER_PATH}/platform_parsed.ld ${LINKER_PATH}/platform.ld

  # Patch retarget.c so that g++ can find exit symbol.
  cat <<EOT >> ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH}/targets/corstone-300/retarget.c

#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6100100)
#else
void RETARGET(exit)(int return_code) {
  RETARGET(_exit)(return_code);
  while (1) {}
}
#endif

EOT

fi

echo "SUCCESS"
