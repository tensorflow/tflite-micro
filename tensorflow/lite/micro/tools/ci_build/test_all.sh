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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"
pwd

echo "Starting to run micro tests at `date`"

make -f tensorflow/lite/micro/tools/make/Makefile clean_downloads DISABLE_DOWNLOADS=true
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn clean DISABLE_DOWNLOADS=true
if [ -d tensorflow/lite/micro/tools/make/downloads ]; then
  echo "ERROR: Downloads directory should not exist, but it does."
  exit 1
fi

echo "Running code style checks at `date`"
tensorflow/lite/micro/tools/ci_build/test_code_style.sh PRESUBMIT

# Add all the test scripts for the various supported platforms here. This
# enables running all the tests together has part of the continuous integration
# pipeline and reduces duplication associated with setting up the docker
# environment.

if [[ ${1} == "GITHUB_PRESUBMIT" ]]; then
  # We enable bazel as part of the github CI only. This is because the same
  # checks are already part of the internal CI and there isn't a good reason to
  # duplicate them.
  #
  # Another reason is that the bazel checks involve some patching of TF
  # workspace and BUILD files and this is an experiment to see what the
  # trade-off should be between the maintenance overhead, increased CI time from
  # the unnecessary TF downloads.
  #
  # See https://github.com/tensorflow/tensorflow/issues/46465 and
  # http://b/177672856 for more context.
  echo "Running bazel tests at `date`"
  tensorflow/lite/micro/tools/ci_build/test_bazel.sh
fi

echo "Running x86 tests at `date`"
tensorflow/lite/micro/tools/ci_build/test_x86.sh

echo "Finished all micro tests at `date`"
