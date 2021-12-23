#!/bin/bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

DOWNLOADED_SIFIVE_SDK_PATH=${DOWNLOADS_DIR}/sifive_fe310_lib

if [ -d ${DOWNLOADED_SIFIVE_SDK_PATH} ]; then
  echo >&2 "${DOWNLOADED_SIFIVE_SDK_PATH} already exists, skipping the download."
else
  ZIP_PREFIX="baeeb8fd497a99b3c141d7494309ec2e64f19bdf"
  SIFIVE_SDK_URL="http://mirror.tensorflow.org/github.com/sifive/freedom-e-sdk/archive/${ZIP_PREFIX}.zip"
  SIFIVE_SDK_MD5="06ee24c4956f8e21670ab3395861fe64"

  TEMPDIR="$(mktemp -d)"
  TEMPFILE="${TEMPDIR}/${ZIP_PREFIX}.zip"
  wget ${SIFIVE_SDK_URL} -O "$TEMPFILE" >&2
  check_md5 "${TEMPFILE}" ${SIFIVE_SDK_MD5}

  unzip -qo "$TEMPFILE" -d "${TEMPDIR}" >&2
  mv "${TEMPDIR}/freedom-e-sdk-${ZIP_PREFIX}" ${DOWNLOADED_SIFIVE_SDK_PATH}
  rm -rf "${TEMPDIR}"

  pushd ${DOWNLOADED_SIFIVE_SDK_PATH} > /dev/null
  create_git_repo ./
  apply_patch_to_folder ./ ../../sifive_sdk.patch "TFLM patch"
  popd > /dev/null
fi

echo "SUCCESS"
