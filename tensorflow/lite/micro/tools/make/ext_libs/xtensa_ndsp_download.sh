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

if [[ ${2} == "hifi3" ]]; then
  COMMIT="d17bf205dc530a9e1a1d979249520f4401529db1"
  LIBRARY_DIRNAME="ndsplib-hifi3"
  LIBRARY_URL="https://github.com/foss-xtensa/${LIBRARY_DIRNAME}/archive/${COMMIT}.zip"
  LIBRARY_MD5="5572b27361736c1f773474ebaf42c5d4"
  CORE_NAME="HiFi3"
elif [[ ${2} == "hifi4" ]]; then
  COMMIT="aba2485ba12d9851fa398bcb5c18c05cc3731a17"
  LIBRARY_DIRNAME="ndsplib-hifi4"
  LIBRARY_URL="https://github.com/foss-xtensa/${LIBRARY_DIRNAME}/archive/${COMMIT}.zip"
  LIBRARY_MD5="062b8f957c662b6ab834bbe284237b6c"
  CORE_NAME="HiFi4"
elif [[ ${2} == "hifi5" ]]; then
  COMMIT="01c92ceb26cc0a598c6d83d17c3d88363bd8f7fc"
  LIBRARY_DIRNAME="ndsplib-hifi5"
  LIBRARY_URL="https://github.com/foss-xtensa/${LIBRARY_DIRNAME}/archive/${COMMIT}.zip"
  LIBRARY_MD5="94b372d608781c13be2fb2d1a8fd3b58"
  CORE_NAME="HiFi5"
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

  unzip -qo "$TEMPFILE" -d ${TEMPDIR} >&2
  unzip -qo ${TEMPDIR}/${LIBRARY_DIRNAME}-${COMMIT}/NDSP_${CORE_NAME}/NDSP_${CORE_NAME}*.zip -d ${TEMPDIR}/${LIBRARY_DIRNAME}-${COMMIT}/NDSP_${CORE_NAME}/ >&2
  find ${TEMPDIR}/${LIBRARY_DIRNAME}-${COMMIT}/NDSP_${CORE_NAME}/* -maxdepth 0 -type d -exec mv {} ${LIBRARY_INSTALL_PATH} \;
  rm -rf "${TEMPDIR}"
  # NDSP sources in GitHub currently uses DOS style newlines, which causes compiler errors.
  find ${LIBRARY_INSTALL_PATH} -type f -exec sed -i.bak 's/\r$//g' {} \;

  pushd "${LIBRARY_INSTALL_PATH}" > /dev/null
  chmod -R +w ./
  if [[ -f "../../ext_libs/ndsplib-${2}.patch" ]]; then
    create_git_repo ./
    apply_patch_to_folder ./ "../../ext_libs/ndsplib-${2}.patch" "TFLM patch"
  fi
  # Rename the strings in __renaming__.h to names that are traceable to TFLM.
  # Note that renaming is disabled by default and must be enabled with -D__RENAMING__
  sed -i 's/NatureDSP_/NatureDSP_TFLM_/' library/include_private/__renaming__.h
fi

echo "SUCCESS"
