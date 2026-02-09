#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Checks if the PR modifies TFLite files by querying GitHub API and comparing
# against a list of tracked files.
#
# Inputs:
#   GITHUB_REPOSITORY
#   PR_NUMBER
#   TFLM_BOT_TOKEN

set -e
set -u

# Ensure we have the base branch to compare against.
git fetch origin main --depth=1

# Get the list of modified files using git.
# We compare the current HEAD against the merge-base with origin/main.
PR_FILES=$(git diff --name-only origin/main...HEAD)

# Create a temp file for PR files
TMP_PR_FILES=$(mktemp)
trap 'rm -f "${TMP_PR_FILES}"' EXIT

echo "${PR_FILES}" > "${TMP_PR_FILES}"

if [ ! -f ci/tflite_files.txt ]; then
  echo "Error: ci/tflite_files.txt not found!"
  exit 1
fi

# Check for intersection between PR files and TFLite files
CONFLICTS=$(grep -F -x -f ci/tflite_files.txt "${TMP_PR_FILES}" || true)

if [ -n "${CONFLICTS}" ]; then
  echo "The following files should be modified in the upstream Tensorflow repo:"
  echo "${CONFLICTS}"
  exit 1
else
  echo "No TfLite files are modified in the PR. We can proceed."
  exit 0
fi
