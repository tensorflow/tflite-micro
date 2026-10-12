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

# On-demand schema sync script for standalone TFLM.
# Downloads the canonical schema.fbs and regenerates schema_generated.h and schema_py_generated.py.
# See header comments in builtin_op_data.h and flatbuffer_conversions.h for syncing C++ files.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../../.."
cd "${ROOT_DIR}"

SCHEMA_URL="https://raw.githubusercontent.com/google-ai-edge/LiteRT/main/tflite/converter/schema/schema.fbs"
TARGET_FBS="tensorflow/lite/schema/schema.fbs"

echo "Downloading schema.fbs from ${SCHEMA_URL}..."
curl -fsSL "${SCHEMA_URL}" -o "${TARGET_FBS}"

echo "Generating C++ and Python FlatBuffers bindings..."
bazel build //tensorflow/lite/schema:schema_fbs_srcs //tensorflow/lite/micro/python:schema_py

if [ -f "bazel-bin/tensorflow/lite/schema/schema_generated.h" ]; then
  cp -f bazel-bin/tensorflow/lite/schema/schema_generated.h tensorflow/lite/schema/schema_generated.h
  echo "Updated tensorflow/lite/schema/schema_generated.h"
fi

if [ -f "bazel-bin/tensorflow/lite/micro/python/schema_py_generated.py" ]; then
  cp -f bazel-bin/tensorflow/lite/micro/python/schema_py_generated.py tensorflow/lite/micro/python/schema_py_generated.py
  echo "Updated tensorflow/lite/micro/python/schema_py_generated.py"
fi

echo "Schema sync completed successfully."
