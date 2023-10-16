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

source ${2}tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
if [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

# The BUILD files in the downloaded folder result in an error with:
#  bazel build tensorflow/lite/micro/...
#
# Parameters:
#   $1 - path to the downloaded flatbuffers code.
function delete_build_files() {
  rm -f `find . -name BUILD -o -name BUILD.bazel`
}

DOWNLOADED_FLATBUFFERS_PATH=${DOWNLOADS_DIR}/flatbuffers

if [ -d ${DOWNLOADED_FLATBUFFERS_PATH} ]; then
  echo >&2 "${DOWNLOADED_FLATBUFFERS_PATH} already exists, skipping the download."
else
  ZIP_PREFIX="v23.5.26"
  FLATBUFFERS_URL="https://github.com/google/flatbuffers/archive/${ZIP_PREFIX}.zip"
  FLATBUFFERS_MD5="e87e8acd8e2d53653387ad78720316e2"

  TEMPDIR="$(mktemp -d)"
  TEMPFILE="${TEMPDIR}/${ZIP_PREFIX}.zip"
  wget ${FLATBUFFERS_URL} -O "$TEMPFILE" >&2
  check_md5 "${TEMPFILE}" ${FLATBUFFERS_MD5}

  unzip -qo "$TEMPFILE" -d "${TEMPDIR}" >&2
  mv "${TEMPDIR}/flatbuffers-${ZIP_PREFIX#v}" ${DOWNLOADED_FLATBUFFERS_PATH}
  rm -rf "${TEMPDIR}"

  pushd ${DOWNLOADED_FLATBUFFERS_PATH} > /dev/null
  delete_build_files ${DOWNLOADED_FLATBUFFERS_PATH}
  create_git_repo ./
  apply_patch_to_folder ./ ../../flatbuffers.patch "TFLM patch"

  popd > /dev/null
fi

echo "SUCCESS"
