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

DOWNLOADED_ETHOS_U_CORE_DRIVER_PATH=${DOWNLOADS_DIR}/ethos_u_core_driver

if [ -d ${DOWNLOADED_ETHOS_U_CORE_DRIVER_PATH} ]; then
  echo >&2 "${DOWNLOADED_ETHOS_U_CORE_DRIVER_PATH} already exists, skipping the download."
else
  UNAME_S=`uname -s`
  if [ ${UNAME_S} != Linux ]; then
    echo "OS type ${UNAME_S} not supported."
    exit 1
  fi

  git clone "https://review.mlplatform.org/ml/ethos-u/ethos-u-core-driver" \
      ${DOWNLOADED_ETHOS_U_CORE_DRIVER_PATH} >&2
  pushd ${DOWNLOADED_ETHOS_U_CORE_DRIVER_PATH} > /dev/null
  git -c advice.detachedHead=false checkout 9622608a5cc318c0933bcce720b59737d03bfb6f
  rm -rf .git
  create_git_repo ./
  popd > /dev/null

fi

echo "SUCCESS"
