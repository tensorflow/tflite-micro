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

rm -rf /tmp/litert

git clone https://github.com/google-ai-edge/LiteRT.git --depth=1 /tmp/litert

SHARED_TFL_CODE=$(grep -v tflite/converter ci/tflite_files.txt)
SHARED_CONVERTER_CODE=$(grep tflite/converter ci/tflite_files.txt)

# Delete all the shared TFL/TFLM code prior to copying from upstream to ensure
# no stale files are left in the tree.
rm -f $(find tensorflow/lite/ -type d \( -path tensorflow/lite/experimental -o -path tensorflow/lite/micro \) -prune -false -o -name "*.cc" -o -name "*.c" -o -name "*.h" -o -name "*.py" -o -name "*.fbs")

# Copy all the right files from the TFLite repo to all the right places in this repo.
for filepath in ${SHARED_TFL_CODE}
do
  local_filepath=${filepath//tflite\//tensorflow\/lite\/}
  mkdir -p $(dirname ${local_filepath})
  /bin/cp /tmp/litert/${filepath} ${local_filepath}
  sed -i 's/tflite\/converter\//tensorflow\/compiler\/mlir\/lite\//' ${local_filepath}
  sed -i 's/tflite\//tensorflow\/lite\//' ${local_filepath}
done
for filepath in ${SHARED_CONVERTER_CODE}
do
  local_filepath=${filepath//tflite\/converter\//tensorflow\/compiler\/mlir\/lite\/}
  mkdir -p $(dirname ${local_filepath})
  /bin/cp /tmp/litert/${filepath} ${local_filepath}
  sed -i 's/tflite\/converter\//tensorflow\/compiler\/mlir\/lite\//' ${local_filepath}
  sed -i 's/tflite\//tensorflow\/lite\//' ${local_filepath}
done

# https://github.com/tensorflow/tflite-micro/pull/8
git checkout tensorflow/lite/kernels/internal/optimized/neon_check.h
# http://b/149862813
git checkout tensorflow/lite/kernels/internal/runtime_shape.h
git checkout tensorflow/lite/kernels/internal/runtime_shape.cc
# http://b/187728891
git checkout tensorflow/lite/kernels/op_macros.h
# http://b/242077843
git checkout tensorflow/lite/kernels/internal/tensor_utils.cc

# We are still generating and checking in the C++ and Python bindings for the TfLite
# flatbuffer schema in the nightly sync to keep it working with the Makefiles.
bazel build tensorflow/lite/python:schema_py
/bin/cp bazel-bin/tensorflow/lite/python/schema_py_generated.py tensorflow/lite/python/schema_py_generated.py

bazel build tensorflow/compiler/mlir/lite/schema:schema_fbs_srcs
/bin/cp ./bazel-bin/tensorflow/compiler/mlir/lite/schema/schema_generated.h tensorflow/lite/schema/schema_generated.h

# Must clean the bazel directories out after building as we don't check these in.
bazel clean

# The shared TFL/TFLM python code uses a different bazel workspace in the two
# repositories (TF and tflite-micro) which needs the import statements to be
# modified.
PY_FILES=$(find tensorflow/lite/tools tensorflow/lite/python -name "*.py")
sed -i 's/from tflite/from tflite_micro\.tensorflow\.lite/' ${PY_FILES}

# Since the TFLM code was deleted from the tensorflow repository, the
# microfrontend is no longer sync'd from upstream and instead maintaned as a
# fork.
git checkout tensorflow/lite/experimental/
