#!/usr/bin/env bash
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
# Sync's the shared third_party/hexagon code from the codelinario repo.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

TEMP_DIR=/tmp/codelinario_tflite_micro
TARGET_DIR=third_party/hexagon
rm -rf ${TEMP_DIR}

git clone --branch release_tflm_ci https://git.codelinaro.org/clo/embedded-ai/tflite-micro.git ${TEMP_DIR}

#
pushd ${TEMP_DIR}
hexagon_hash=$(git log -1 --format=format:"%H")
echo "Importing third_party/hexagon from codelinario at commit: ${hexagon_hash}"
popd

# As part of the import from upstream TF, we generate the Python bindings for
# the TfLite flatbuffer schema.
cp -r ${TEMP_DIR}/${TARGET_DIR}/. ${TARGET_DIR}/
