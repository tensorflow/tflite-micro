#!/bin/bash -e
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
# Tests a binary with QEMU by parsing the log output.
# Parameters:
#  ${1} suffix for qemu binary (e.g. to use qemu-arm ${1} should be arm
#  ${2} architecture to pass to qemu (e.g. cortex-m3)
#  ${3} cross-compiled binary to be emulated
#  ${4} - String that is checked for pass/fail.
#  ${5} - target (cortex_m_qemu etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_ROOT_DIR=${SCRIPT_DIR}/../../../../

TEST_TMPDIR=/tmp/test_${5}
MICRO_LOG_PATH=${TEST_TMPDIR}/${3}
MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt

mkdir -p ${MICRO_LOG_PATH}
qemu-${1} -cpu ${2} ${3} 2>&1 | tee ${MICRO_LOG_FILENAME}
if [[ ${4} != "non_test_binary" ]]
then
  if grep -q "${4}" ${MICRO_LOG_FILENAME}
  then
    echo "Pass"
    exit 0
  else
    echo "Fail"
    exit 1
  fi
fi
