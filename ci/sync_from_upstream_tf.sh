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

# As part of the import from upstream TF, we generate the Python bindings for
# the TfLite flatbuffer schema.
cd /tmp/tensorflow
bazel build tensorflow/lite/python:schema_py
/bin/cp bazel-bin/tensorflow/lite/python/schema_py_generated.py tensorflow/lite/python
cd -

SHARED_TFL_CODE=$(<ci/tflite_files.txt)

for filepath in ${SHARED_TFL_CODE}
do
  mkdir -p $(dirname ${filepath})
  /bin/cp /tmp/tensorflow/${filepath} ${filepath}
done

# The shared TFL/TFLM python code uses a different bazel workspace in the two
# repositories (TF and tflite-micro) which needs the import statements to be
# modified.
PY_FILES=$(find tensorflow/lite/tools tensorflow/lite/python -name "*.py")
sed -i 's/from tensorflow\.lite/from tflite_micro\.tensorflow\.lite/' ${PY_FILES}

# Since the TFLM code was deleted from the tensorflow repository, the
# microfrontend is no longer sync'd from upstream and instead maintaned as a
# fork.
git checkout tensorflow/lite/experimental/microfrontend/lib/
