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

DOWNLOADED_QEMU_PATH=${DOWNLOADS_DIR}/qemu

if [ -d ${DOWNLOADED_QEMU_PATH} ]; then
  echo >&2 "${DOWNLOADED_QEMU_PATH} already exists, skipping the download."
else
  # TODO remove this once ci.yaml is updated.
  sudo apt-get install -y ninja-build
  mkdir ${DOWNLOADED_QEMU_PATH}
  LINUX_PORTABLE_URL="https://download.qemu.org/qemu-6.2.0.tar.xz"
  TEMP_ARCHIVE="/tmp/qemu.tar.xz"

  echo >&2 "Downloading from url: ${LINUX_PORTABLE_URL}"
  wget ${LINUX_PORTABLE_URL} -O ${TEMP_ARCHIVE} >&2
  tar xJf ${TEMP_ARCHIVE} --strip-components=1 --directory ${DOWNLOADED_QEMU_PATH} >&2
  cd ${DOWNLOADED_QEMU_PATH}
  ./configure
  make -j8
fi

echo "SUCCESS"
