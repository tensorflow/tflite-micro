#!/usr/bin/env bash
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

#
# Updates the checked-generated source for the preprocessor schema
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../..
cd "${ROOT_DIR}"

# We generate and check in the C++ and Python bindings for the schema to keep it
# working with the Makefiles. The makefiles do not support running flatc.
bazel build //codegen/preprocessor:preprocessor_schema_fbs_srcs
/bin/cp ./bazel-bin/codegen/preprocessor/preprocessor_schema_generated.h \
  codegen/preprocessor/preprocessor_schema_generated.h

bazel build //codegen/preprocessor:preprocessor_schema_py
/bin/cp ./bazel-bin/codegen/preprocessor/preprocessor_schema_py_generated.py \
  codegen/preprocessor/preprocessor_schema_py_generated.py

