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

# explicitly call third_party_downloads since we need pigweed for the license
# and clang-format checks.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Explicitly disable exit on error so that we can report all the style errors in
# one pass and clean up the temporary git repository even when one of the
# scripts fail with an error code.
set +e

# --fix_formatting to let the script fix both code and build file format error.
FIX_FORMAT_FLAG=${1}

############################################################
# License Check
############################################################
tensorflow/lite/micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/pigweed_presubmit.py \
  tensorflow/lite/kernels/internal/reference/ \
  tensorflow/lite/micro/ \
  third_party/ \
  -p copyright_notice \
  -e kernels/internal/reference/integer_ops/ \
  -e kernels/internal/reference/reference_ops.h \
  -e python/schema_py_generated.py \
  -e python_requirements.in \
  -e tensorflow/lite/micro/compression/metadata_saved.h \
  -e tools/make/downloads \
  -e tools/make/targets/ecm3531 \
  -e BUILD\
  -e leon_commands \
  -e "\.bmp" \
  -e "\.bzl" \
  -e "\.csv" \
  -e "\.h5" \
  -e "\.inc" \
  -e "\.ipynb" \
  -e "\.patch" \
  -e "\.properties" \
  -e "\.tflite" \
  -e "\.tpl" \
  -e "\.txt" \
  -e "\.wav" \
  --output-directory /tmp

LICENSE_CHECK_RESULT=$?

############################################################
# Code Formatting Check
############################################################

if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" ]]
then
  FIX_FORMAT_OPTIONS="--fix"
else
  FIX_FORMAT_OPTIONS=""
fi

EXCLUDE_SHARED_TFL_CODE=$(sed 's/^/-e /' ci/tflite_files.txt)

tensorflow/lite/micro/tools/make/downloads/pigweed/pw_presubmit/py/pw_presubmit/format_code.py \
  ${FIX_FORMAT_OPTIONS} \
  -e "\.github" \
  -e third_party/hexagon \
  -e third_party/xtensa \
  -e ci \
  -e c/common.c \
  -e codegen/preprocessor/preprocessor_schema_generated.h \
  -e codegen/preprocessor/preprocessor_schema_py_generated.py \
  -e core/api/error_reporter.cc \
  -e kernels/internal/reference/integer_ops/ \
  -e kernels/internal/reference/reference_ops.h \
  -e kernels/internal/types.h \
  -e lite/python \
  -e lite/tools \
  -e experimental \
  -e schema/schema_generated.h \
  -e schema/schema_utils.h \
  -e tensorflow/lite/micro/compression/metadata_saved.h \
  -e tensorflow/lite/micro/tools/layer_by_layer_schema_generated.h \
  -e "\.inc" \
  -e "\.md" \
  ${EXCLUDE_SHARED_TFL_CODE}

CODE_FORMAT_RESULT=$?

############################################################
# Build Formatting Check
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
# All checks are complete, clean up.
###########################################################################


# Re-enable exit on error now that we are done with the temporary git repo.
set -e

if [[ ${CODE_FORMAT_RESULT}  != 0 || ${BUILD_FORMAT_RESULT} != 0 ]]
then
  echo "The formatting errors can be fixed with tensorflow/lite/micro/tools/ci_build/test_code_style.sh --fix_formatting"
fi
if [[ ${LICENSE_CHECK_RESULT}  != 0 || \
      ${CODE_FORMAT_RESULT}    != 0 || \
      ${BUILD_FORMAT_RESULT}   != 0 || \
      ${GTEST_RESULT}          != 0 || \
      ${ERROR_REPORTER_RESULT} != 0 || \
      ${ASSERT_RESULT}         != 0    \
   ]]
then
  exit 1
fi
