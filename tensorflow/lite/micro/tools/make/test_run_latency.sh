#!/bin/bash
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
# Called with following arguments:
# 1 - Name of the test file

set -e

TEST_FILE_NAME=${1}
TEST_LATENCY_FILE=/tmp/${TEST_FILE_NAME}

if [ ! -e ${TEST_LATENCY_FILE} ]; then
  exit 0
fi

REQUIRED_LOG=$(head -n -3 ${TEST_LATENCY_FILE})
echo "${REQUIRED_LOG}"
END_TIME=$(tail -n 3 ${TEST_LATENCY_FILE} | head -n 1 | cut -c 5- | xargs echo -n)
echo "Run and Build of ${TEST_FILE_NAME} took ${END_TIME}"
rm ${TEST_LATENCY_FILE}
