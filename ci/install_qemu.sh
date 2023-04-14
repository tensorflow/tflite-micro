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

sudo apt-get install -y ninja-build
LINUX_PORTABLE_URL="https://download.qemu.org/qemu-6.2.0.tar.xz"
TEMP_ARCHIVE="/tmp/qemu.tar.xz"

echo >&2 "Downloading from url: ${LINUX_PORTABLE_URL}"
wget ${LINUX_PORTABLE_URL} -O ${TEMP_ARCHIVE} >&2

QEMU_HOME="/usr/local/qemu"
mkdir ${QEMU_HOME}
tar xJf ${TEMP_ARCHIVE} --strip-components=1 --directory ${QEMU_HOME} >&2
cd ${QEMU_HOME}
./configure
make -j8
make install
