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
# Sync's the shared TfLite / TFLM code from the upstream Tensorflow repo.
#
# While the standalone TFLM repo is under development, we are also sync'ing all
# of the TFLM code via this script.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

rm -rf /tmp/tensorflow

git clone https://github.com/tensorflow/tensorflow.git --depth=1 /tmp/tensorflow

SHARED_TFL_CODE=$(<ci/upstream_list.txt)

for filepath in ${SHARED_TFL_CODE}
do
  mkdir -p $(dirname ${filepath})
  /bin/cp /tmp/tensorflow/${filepath} ${filepath}
done

# The microfrontend is sync'd from upstream but not as part of the explicitly
# specified SHARED_TFL_CODE since this is only needed for the examples.
rm -rf tensorflow/lite/experimental/microfrontend/lib
cp -r /tmp/tensorflow/tensorflow/lite/experimental/microfrontend/lib tensorflow/lite/experimental/microfrontend/lib
