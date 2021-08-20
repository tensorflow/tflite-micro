#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Tests the microcontroller code for esp32 platform

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

pip3 install Pillow
pip3 install Wave

TARGET=esp
TARGET_ARCH=xtensa-esp32

readable_run make -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} third_party_downloads

# clean all
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

# validate esp32 build for libtensorflow-microlite.a
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} TARGET_TOOLCHAIN_PREFIX=xtensa-esp32-elf-

# generate examples
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} generate_hello_world_esp_project

# readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} generate_person_detection_int8_esp_project

readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} generate_micro_speech_esp_project

# build examples
cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32_default/prj/hello_world/esp-idf
readable_run idf.py build

#cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32_default/prj/person_detection_int8/esp-idf
#readable_run git clone https://github.com/espressif/esp32-camera.git components/esp32-camera
#cd components/esp32-camera/
#readable_run git checkout eacd640b8d379883bff1251a1005ebf3cf1ed95c
#cd ../../
#readable_run idf.py build

cd "${ROOT_DIR}"/tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32_default/prj/micro_speech/esp-idf
readable_run idf.py build

