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

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# Explicitly disable exit on error so that we can report all the style errors in
# one pass.
set +e

# --fix_formatting to let the script fix both code and build file format errors.
FIX_FORMAT_FLAG=${1}

############################################################
# License Check
############################################################

python3 tensorflow/lite/micro/tools/ci_build/check_license.py
LICENSE_CHECK_RESULT=$?

############################################################
# File Exclusions for Formatting
############################################################

EXCLUDES_REGEX="(\.github|third_party/hexagon|third_party/xtensa|ci/|c/common\.c|core/api/error_reporter\.cc|kernels/internal/reference/integer_ops/|kernels/internal/reference/reference_ops\.h|kernels/internal/types\.h|lite/python|lite/tools|experimental|schema/schema_generated\.h|schema/schema_utils\.h|tensorflow/lite/micro/compression/metadata_saved\.h|tensorflow/lite/micro/tools/layer_by_layer_schema_generated\.h|\.inc$|\.md$)"

CPP_FILES=$(git ls-files "*.cc" "*.h" "*.c" | grep -v -E "${EXCLUDES_REGEX}" | grep -v -F -f ci/tflite_files.txt)
PY_FILES=$(git ls-files "*.py" | grep -v -E "${EXCLUDES_REGEX}" | grep -v -F -f ci/tflite_files.txt)

############################################################
# C/C++ Formatting Check (clang-format)
############################################################

if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" ]]; then
  echo "${CPP_FILES}" | xargs -r clang-format -i
  CPP_FORMAT_RESULT=$?
else
  echo "${CPP_FILES}" | xargs -r clang-format --dry-run --Werror
  CPP_FORMAT_RESULT=$?
fi

############################################################
# Python Formatting Check (yapf)
############################################################

if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" ]]; then
  echo "${PY_FILES}" | xargs -r python3 -m yapf --parallel -i
  PY_FORMAT_RESULT=$?
else
  echo "${PY_FILES}" | xargs -r python3 -m yapf --parallel --diff
  PY_FORMAT_RESULT=$?
fi

############################################################
# Build Formatting Check (buildifier)
############################################################

BUILDIFIER_MODE="diff"
if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" ]]
then
  BUILDIFIER_MODE="fix"
fi

BUILD_FILES=$(find . -name BUILD -o -name "*.bzl" -not -path "./tensorflow/lite/micro/tools/make/downloads/*")
buildifier --mode=${BUILDIFIER_MODE} --diff_command="diff -u" ${BUILD_FILES}
BUILD_FORMAT_RESULT=$?

#############################################################################
# Avoided specific-code snippets for TFLM
#############################################################################
pushd tensorflow/lite/

CHECK_CONTENTS_PATHSPEC=\
"micro"\
" :(exclude)micro/tools/ci_build/test_code_style.sh"\
" :(exclude)*\.md"

# See https://github.com/tensorflow/tensorflow/issues/46297 for more context.
check_contents "gtest|gmock" "${CHECK_CONTENTS_PATHSPEC}" \
  "These matches can likely be deleted."
GTEST_RESULT=$?

# See http://b/175657165 for more context.
ERROR_REPORTER_MESSAGE=\
"TF_LITE_REPORT_ERROR should be used instead, so that log strings can be "\
"removed to save space, if needed."

check_contents "error_reporter.*Report\(|context->ReportError\(" \
  "${CHECK_CONTENTS_PATHSPEC}" "${ERROR_REPORTER_MESSAGE}"
ERROR_REPORTER_RESULT=$?

# See http://b/175657165 for more context.
ASSERT_PATHSPEC=\
"${CHECK_CONTENTS_PATHSPEC}"\
" :(exclude)micro/examples/micro_speech/esp/ringbuf.c"\
" :(exclude)*\.ipynb"\
" :(exclude)*\.py"\

check_contents "\<assert\>" "${ASSERT_PATHSPEC}" \
  "assert should not be used in TFLM code.."
ASSERT_RESULT=$?

popd

###########################################################################
# All checks are complete, report errors and exit.
###########################################################################

set -ex

if [[ ${CPP_FORMAT_RESULT}   != 0 || \
      ${PY_FORMAT_RESULT}    != 0 || \
      ${BUILD_FORMAT_RESULT} != 0 ]]
then
  echo "The formatting errors can be fixed with tensorflow/lite/micro/tools/ci_build/test_code_style.sh --fix_formatting"
fi

if [[ ${LICENSE_CHECK_RESULT}  != 0 || \
      ${CPP_FORMAT_RESULT}     != 0 || \
      ${PY_FORMAT_RESULT}      != 0 || \
      ${BUILD_FORMAT_RESULT}   != 0 || \
      ${GTEST_RESULT}          != 0 || \
      ${ERROR_REPORTER_RESULT} != 0 || \
      ${ASSERT_RESULT}         != 0    \
   ]]
then
  exit 1
fi
