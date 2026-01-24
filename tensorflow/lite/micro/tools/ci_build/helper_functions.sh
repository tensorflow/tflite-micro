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

# Collection of helper functions that can be used in the different continuous
# integration scripts.

# Executes a command with a timeout and formats output for GitHub Actions.
function readable_run {
  # 1. specific GitHub Actions syntax to start a collapsible section
  echo "::group::Running: $*"

  # Print timestamp (optional, if you want it in the raw text log)
  echo "Starting at $(date)"

  # 2. Run with timeout, but allow stdout/stderr to stream LIVE.
  #    We don't need > log_file because GHA hides the noise inside the group.
  timeout 60m "$@"
  local status=$?

  # 3. Close the group.
  #    Everything printed between ::group:: and ::endgroup:: is folded.
  echo "::endgroup::"

  if [ $status -eq 0 ]; then
    return 0
  elif [ $status -eq 124 ]; then
    echo "::error::Command timed out after 60 minutes!"
    return 124
  else
    echo "::error::Command failed with exit code $status"
    return $status
  fi
}

# Check if the regex ${1} is to be found in the pathspec ${2}.
# An optional error messsage can be passed with ${3}
function check_contents() {
  GREP_OUTPUT=$(git grep -E -rn ${1} -- ${2})

  if [ "${GREP_OUTPUT}" ]; then
    echo "=============================================="
    echo "Found matches for ${1} that are not permitted."
    echo "${3}"
    echo "=============================================="
    echo "${GREP_OUTPUT}"
    return 1
  fi
}

# Determine the number of parallel jobs to use for make.
# Respects the MAKE_JOBS_NUM environment variable if set.
# Otherwise, defaults to the number of processing units available.
function get_parallel_jobs {
  if [[ -n "${MAKE_JOBS_NUM}" ]]; then
    echo "-j${MAKE_JOBS_NUM}"
  elif command -v nproc > /dev/null; then
    echo "-j$(nproc)"
  elif [[ "$(uname)" == "Darwin" ]]; then
    echo "-j$(sysctl -n hw.ncpu)"
  else
    echo "-j1"
  fi
}
