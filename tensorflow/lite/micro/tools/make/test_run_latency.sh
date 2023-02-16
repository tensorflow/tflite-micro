#!/bin/bash
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
# Called with following arguments:
# 1 - Name of the test file
# 2 - Name of the test script
# 3 - Name of the binary
# 4 - String output after all the tests are passed
# 5 - Name of the target
# The first parameter is used for logging purpose. The last four parameters are
# used to run the test.

set -e

TEST_FILE_NAME=${1}
TEST_SCRIPT=${2}
BINARY_NAME=${3}
TEST_PASS_STRING=${4}
TARGET_NAME=${5}

# Output to stdout and stderr go to their normal places:
exec 3>&1 4>&2
time_log=$( { TIMEFORMAT="%R"; time ${TEST_SCRIPT} ${BINARY_NAME} ${TEST_PASS_STRING} ${TARGET_NAME} 1>&3 2>&4; } 2>&1 ) # Captures time output only.
exec 3>&- 4>&-

echo "Running ${TEST_FILE_NAME} took ${time_log} seconds"
