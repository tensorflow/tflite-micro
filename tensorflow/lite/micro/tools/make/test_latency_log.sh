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
# This script is responsible for running the tests and also to log out the
# time (in seconds) it took to run the test file. It is using the linux time
# command to measure the latency. Setting the TIMEFORMAT to '%R' is providing
# us the real time latency.
#
# Called with following arguments:
# 1 - Name of the test file
# 2 - Name of the test script
# <variable list of parameters for the test script>

set -e

ARGS=("${@}")
TEST_FILE_NAME=$1
TEST_SCRIPT=$2
shift 2

if [[ ! -x "$TEST_SCRIPT" && ! -f "$TEST_SCRIPT" ]]; then
  echo "ERROR: Test script '$TEST_SCRIPT' not found or not executable."
  exit 1
fi

# FD 3 -> Original Stdout
# FD 4 -> Original Stderr
exec 3>&1 4>&2

# We turn off 'set -e' temporarily so we can capture the exit code manually.
set +e

time_log=$( { 
  TIMEFORMAT="%R"
  # Run the test. 
  # - Output goes to FD3/4 (screen).
  # - 'time' output goes to 2>&1 (captured into variable).
  time "${TEST_SCRIPT}" "$@" 1>&3 2>&4
} 2>&1 )

EXIT_CODE=$?

# Turn 'set -e' back on
set -e

# Close File Descriptors
exec 3>&- 4>&-

if [ $EXIT_CODE -ne 0 ]; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${TEST_FILE_NAME} failed with exit code ${EXIT_CODE}"
  echo "DEBUG INFO: The test script attempted was: ${TEST_SCRIPT}"
  echo "--------------------------------------------------------"
  exit $EXIT_CODE
fi

echo "Running ${TEST_FILE_NAME} took ${time_log} seconds"
