#!/usr/bin/env bash
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
# Build a TFLite micro test binary and capture its size.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"


SIZEFILE_DIR=${ROOT_DIR}/ci
MAKEFILE_DIR=${ROOT_DIR}/tensorflow/lite/micro/tools/make
BENCHMARK_TARGET=binary_size_test
BUILD_TYPE=default

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

# TODO(b/143715361): downloading first to allow for parallel builds.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Next, make sure that the release build succeeds.
# Build for x86.
TARGET=linux
TARGET_ARCH=x86_64
readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean
readable_run make -j8 -f tensorflow/lite/micro/tools/make/Makefile build BUILD_TYPE=${BUILD_TYPE} TARGET=${TARGET} TARGET_ARCH=${TARGET_ARCH} ${BENCHMARK_TARGET}

# Capture size of the test binary.
GENDIR=${MAKEFILE_DIR}/gen/${TARGET}_${TARGET_ARCH}_${BUILD_TYPE}/
BINDIR=${GENDIR}/bin/
size ${BINDIR}/${BENCHMARK_TARGET} > ${SIZEFILE_DIR}/size_log.txt

