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

# Explicitly disable exit on error so that we can report all the style errors in
# one pass.
set +e

# Parse arguments
USE_DOCKER=0
FIX_FORMAT_FLAG=""
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --docker)
      USE_DOCKER=1
      ;;
    --fix_formatting|-f|--fix)
      FIX_FORMAT_FLAG="--fix_formatting"
      PASSTHROUGH_ARGS+=("${arg}")
      ;;
    *)
      PASSTHROUGH_ARGS+=("${arg}")
      ;;
  esac
done

# If --docker was requested and we are not already inside a container,
# delegate execution to run_in_docker.sh
if [[ ${USE_DOCKER} -eq 1 ]]; then
  if [[ -f /.dockerenv ]]; then
    # Already running inside Docker; do not recurse.
    :
  else
    exec "${SCRIPT_DIR}/run_in_docker.sh" "${BASH_SOURCE[0]}" "${PASSTHROUGH_ARGS[@]}"
  fi
fi

function start_group() {
  local title="$1"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::${title}"
  else
    echo ""
    echo "============================================================"
    echo "=== ${title}"
    echo "============================================================"
  fi
}

function end_group() {
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::endgroup::"
  fi
}

############################################################
# License Check
############################################################

start_group "License Check"
python3 tensorflow/lite/micro/tools/ci_build/check_license.py
LICENSE_CHECK_RESULT=$?
end_group

############################################################
# File Exclusions for Formatting
############################################################

EXCLUDES_REGEX="(\.github|third_party/hexagon|third_party/xtensa|ci/|c/common\.c|core/api/error_reporter\.cc|kernels/internal/reference/integer_ops/|kernels/internal/reference/reference_ops\.h|kernels/internal/types\.h|lite/python|lite/tools|experimental|schema/schema_generated\.h|schema/schema_utils\.h|tensorflow/lite/micro/compression/metadata_saved\.h|tensorflow/lite/micro/tools/layer_by_layer_schema_generated\.h|\.inc$|\.md$)"

CPP_FILES=$(git ls-files "*.cc" "*.h" "*.c" | grep -v -E "${EXCLUDES_REGEX}" | grep -v -F -f ci/tflite_files.txt)
PY_FILES=$(git ls-files "*.py" | grep -v -E "${EXCLUDES_REGEX}" | grep -v -F -f ci/tflite_files.txt)
BUILD_FILES=$(git ls-files "*BUILD" "*BUILD.bazel" "*.bzl" | grep -v -E "${EXCLUDES_REGEX}" | grep -v -F -f ci/tflite_files.txt)

############################################################
# C/C++ Formatting Check (clang-format)
############################################################

start_group "C/C++ Formatting (clang-format)"
if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" || ${FIX_FORMAT_FLAG} == "-f" || ${FIX_FORMAT_FLAG} == "--fix" ]]; then
  echo "${CPP_FILES}" | xargs -r clang-format -i
  CPP_FORMAT_RESULT=$?
  if [[ ${CPP_FORMAT_RESULT} -eq 0 ]]; then
    echo "Formatted C/C++ files."
  fi
else
  CLANG_FORMAT_LOG=$(mktemp)
  echo "${CPP_FILES}" | xargs -r clang-format --dry-run --Werror > "${CLANG_FORMAT_LOG}" 2>&1
  CPP_FORMAT_RESULT=$?
  if [[ ${CPP_FORMAT_RESULT} -ne 0 ]]; then
    cat "${CLANG_FORMAT_LOG}"
    echo ""
    echo "The following C/C++ file(s) require formatting:"
    grep -E ': error: code should be clang-formatted' "${CLANG_FORMAT_LOG}" | cut -d: -f1 | sort -u | sed 's/^/  /'
  else
    echo "PASSED: All C/C++ files are properly formatted."
  fi
  rm -f "${CLANG_FORMAT_LOG}"
fi
end_group

############################################################
# Python Formatting & Lint Check (ruff)
############################################################

start_group "Python Formatting & Lint (ruff)"
if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" || ${FIX_FORMAT_FLAG} == "-f" || ${FIX_FORMAT_FLAG} == "--fix" ]]; then
  echo "${PY_FILES}" | xargs -r ruff format
  PY_FORMAT_RESULT=$?
  echo "${PY_FILES}" | xargs -r ruff check --fix
  PY_LINT_RESULT=$?
  if [[ ${PY_FORMAT_RESULT} -ne 0 || ${PY_LINT_RESULT} -ne 0 ]]; then
    PY_CHECK_RESULT=1
  else
    PY_CHECK_RESULT=0
  fi
else
  echo "${PY_FILES}" | xargs -r ruff format --diff
  PY_FORMAT_RESULT=$?
  echo "${PY_FILES}" | xargs -r ruff check
  PY_LINT_RESULT=$?
  if [[ ${PY_FORMAT_RESULT} -eq 0 && ${PY_LINT_RESULT} -eq 0 ]]; then
    echo "PASSED: All Python files are properly formatted and pass linting."
    PY_CHECK_RESULT=0
  else
    PY_CHECK_RESULT=1
  fi
fi
end_group

############################################################
# Build Formatting Check (buildifier)
############################################################

start_group "Build File Formatting (buildifier)"
BUILDIFIER_MODE="diff"
BUILDIFIER_LINT="warn"
if [[ ${FIX_FORMAT_FLAG} == "--fix_formatting" || ${FIX_FORMAT_FLAG} == "-f" || ${FIX_FORMAT_FLAG} == "--fix" ]]; then
  BUILDIFIER_MODE="fix"
  BUILDIFIER_LINT="fix"
fi

echo "${BUILD_FILES}" | xargs -r buildifier --mode=${BUILDIFIER_MODE} --lint=${BUILDIFIER_LINT} --diff_command="diff -u"
BUILD_FORMAT_RESULT=$?
if [[ ${BUILD_FORMAT_RESULT} -eq 0 ]]; then
  echo "PASSED: All build files are properly formatted and pass linting."
fi
end_group

#############################################################################
# Avoided specific-code snippets for TFLM
#############################################################################

pushd tensorflow/lite/ > /dev/null

CHECK_CONTENTS_PATHSPEC=\
"micro"\
" :(exclude)micro/tools/ci_build/test_code_style.sh"\
" :(exclude)*\.md"

# See https://github.com/tensorflow/tensorflow/issues/46297 for more context.
start_group "Disallowed Patterns (gtest / gmock)"
check_contents "gtest|gmock" "${CHECK_CONTENTS_PATHSPEC}" \
  "These matches can likely be deleted."
GTEST_RESULT=$?
if [[ ${GTEST_RESULT} -eq 0 ]]; then
  echo "PASSED: No disallowed gtest/gmock patterns found."
fi
end_group

# See http://b/175657165 for more context.
start_group "Disallowed Patterns (ReportError)"
ERROR_REPORTER_MESSAGE=\
"TF_LITE_REPORT_ERROR should be used instead, so that log strings can be "\
"removed to save space, if needed."

check_contents "error_reporter.*Report\(|context->ReportError\(" \
  "${CHECK_CONTENTS_PATHSPEC}" "${ERROR_REPORTER_MESSAGE}"
ERROR_REPORTER_RESULT=$?
if [[ ${ERROR_REPORTER_RESULT} -eq 0 ]]; then
  echo "PASSED: No disallowed ReportError patterns found."
fi
end_group

# See http://b/175657165 for more context.
start_group "Disallowed Patterns (<assert>)"
ASSERT_PATHSPEC=\
"${CHECK_CONTENTS_PATHSPEC}"\
" :(exclude)micro/examples/micro_speech/esp/ringbuf.c"\
" :(exclude)*\.ipynb"\
" :(exclude)*\.py"\

check_contents "\<assert\>" "${ASSERT_PATHSPEC}" \
  "assert should not be used in TFLM code."
ASSERT_RESULT=$?
if [[ ${ASSERT_RESULT} -eq 0 ]]; then
  echo "PASSED: No disallowed assert patterns found."
fi
end_group

popd > /dev/null

###########################################################################
# All checks are complete, report summary and exit.
###########################################################################

TOTAL_FAILURES=0

if [[ -t 1 || -n "${GITHUB_ACTIONS:-}" ]]; then
  COLOR_PASS="\033[32mPASS\033[0m"
  COLOR_FAIL="\033[31mFAIL\033[0m"
else
  COLOR_PASS="PASS"
  COLOR_FAIL="FAIL"
fi

function print_status() {
  local status=$1
  local name=$2
  if [[ ${status} -eq 0 ]]; then
    printf "  [ %b ] %s\n" "${COLOR_PASS}" "${name}"
  else
    printf "  [ %b ] %s\n" "${COLOR_FAIL}" "${name}"
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      echo "::error::Code style check failed: ${name}"
    fi
  fi
}

echo ""
echo "============================================================"
echo "                 Code Style Check Summary"
echo "============================================================"
print_status "${LICENSE_CHECK_RESULT}"  "License Check"
print_status "${CPP_FORMAT_RESULT}"     "C/C++ Formatting (clang-format)"
print_status "${PY_CHECK_RESULT}"      "Python Formatting & Lint (ruff)"
print_status "${BUILD_FORMAT_RESULT}"   "Build File Formatting (buildifier)"
print_status "${GTEST_RESULT}"          "Disallowed Code: gtest / gmock"
print_status "${ERROR_REPORTER_RESULT}" "Disallowed Code: ReportError"
print_status "${ASSERT_RESULT}"         "Disallowed Code: <assert>"
echo "============================================================"

if [[ ${TOTAL_FAILURES} -gt 0 ]]; then
  echo "FAILED: ${TOTAL_FAILURES} check(s) failed."
  echo ""
  if [[ ${CPP_FORMAT_RESULT}   != 0 || \
        ${PY_FORMAT_RESULT}    != 0 || \
        ${BUILD_FORMAT_RESULT} != 0 ]]; then
    echo "Formatting errors can be fixed automatically with:"
    echo "  tensorflow/lite/micro/tools/ci_build/test_code_style.sh --fix_formatting"
    echo ""
  fi
  if [[ ${LICENSE_CHECK_RESULT}  != 0 || \
        ${PY_LINT_RESULT}        != 0 || \
        ${GTEST_RESULT}          != 0 || \
        ${ERROR_REPORTER_RESULT} != 0 || \
        ${ASSERT_RESULT}         != 0 ]]; then
    echo "Non-formatting / lint errors require manual code fixes (see log above for details)."
    echo ""
  fi
  if [[ ! -f /.dockerenv && -z "${GITHUB_ACTIONS:-}" ]]; then
    echo "Tip: To run formatting or checks using the exact CI tool versions, run with --docker:"
    echo "  tensorflow/lite/micro/tools/ci_build/test_code_style.sh --docker"
    echo "  tensorflow/lite/micro/tools/ci_build/test_code_style.sh --docker --fix_formatting"
    echo ""
  fi
  echo "============================================================"
  exit 1
else
  echo "All code style checks passed successfully!"
  echo "============================================================"
  exit 0
fi
