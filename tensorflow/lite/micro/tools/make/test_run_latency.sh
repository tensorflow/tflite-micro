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

var=$({ time ${TEST_SCRIPT} ${BINARY_NAME} ${TEST_PASS_STRING} ${TARGET_NAME}; } 2>&1)

IFS=$'\n'
# Split the output of the command into sentences
sentences=$(echo "${var}" | sed 's/\n//g')

# Get the number of lines
line_count=0
for sentence in $sentences; do
  let "line_count += 1"
done

# Reduce the line_count by 3 lines as those are the time related data.
let "line_count -= 3"

pos=0
test_latency=''
for sentence in $sentences; do
  # Print all but time related logs
  if [ $pos -lt $line_count ]; then
    echo "$sentence"
  else
    # Just get the first time related log
    test_latency=$sentence
    break;
  fi
  let "pos += 1"
done

# Discard the 'real' part of the log message
latency=${test_latency:5:${#test_latency}}
echo "Running of ${TEST_FILE_NAME} took ${latency}"