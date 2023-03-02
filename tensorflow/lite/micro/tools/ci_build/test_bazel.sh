#!/usr/bin/env bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# We are using a bazel build followed by bazel test to make sure that the CI
# covers non-test binary targets as well. These were previousbly covered by
# having build_test but that was removed with #194.

CC=clang readable_run bazel build ... \
  --build_tag_filters=-no_oss
CC=clang readable_run bazel test ... \
  --test_tag_filters=-no_oss --build_tag_filters=-no_oss \
  --test_output=errors

# TODO(b/178621680): enable ubsan once bazel + clang + ubsan errors are fixed.
#CC=clang readable_run bazel test tensorflow/lite/micro/... --config=ubsan --test_tag_filters=-no_oss,-noubsan --build_tag_filters=-no_oss,-noubsan

