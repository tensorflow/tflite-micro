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

# If the TENSORFLOW_ROOT command line flag is mentioned during the build,
# like $TENSORFLOW_ROOT=tflite-micro/, then we can simply jump to that
# particular directory and run the process. If that is not mentioned, then
# we following the old path of moving up from the SCRIPT_DIR.
if [ ! -z ${2} ]
then
  ROOT_DIR=${2}
else
  ROOT_DIR=${SCRIPT_DIR}/../../../../..
fi

cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

DOWNLOADED_PIGWEED_PATH=${DOWNLOADS_DIR}/pigweed

if [ -d ${DOWNLOADED_PIGWEED_PATH} ]; then
  echo >&2 "${DOWNLOADED_PIGWEED_PATH} already exists, skipping the download."
else
  git clone https://pigweed.googlesource.com/pigweed/pigweed ${DOWNLOADED_PIGWEED_PATH} >&2
  pushd ${DOWNLOADED_PIGWEED_PATH} > /dev/null

  git checkout 47268dff45019863e20438ca3746c6c62df6ef09 >&2
  rm -rf ${DOWNLOADED_PIGWEED_PATH}/.git
  rm -f `find . -name BUILD`

  create_git_repo ./
  apply_patch_to_folder ./ ../../pigweed.patch "TFLM patch"

  popd > /dev/null
fi

echo "SUCCESS"
