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
#   PR_SHA (optional, to download tracking list from PR commit)
#   TFLM_BOT_TOKEN

set -e
set -u

export GH_TOKEN="${TFLM_BOT_TOKEN}"

echo "Fetching files modified in PR #${PR_NUMBER}..."
# Use GitHub CLI auto-pagination to safely pull all file changes (up to 3,000)
PR_FILES=$(gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/files" --paginate --jq '.[].filename')

# Create temp files for PR files list and PR's tflite_files database
TMP_PR_FILES=$(mktemp)
TMP_TFLITE_FILES=$(mktemp)
trap 'rm -f "${TMP_PR_FILES}" "${TMP_TFLITE_FILES}"' EXIT

echo "${PR_FILES}" > "${TMP_PR_FILES}"

TFLITE_FILES_FILE="ci/tflite_files.txt"

if [ -n "${PR_SHA:-}" ]; then
  echo "Downloading ci/tflite_files.txt from PR commit ${PR_SHA}..."
  # Fetch via API (raw content accept header) to support secure reading of untrusted data
  URL_TXT="repos/${GITHUB_REPOSITORY}/contents/ci/tflite_files.txt?ref=${PR_SHA}"
  if gh api -H "Accept: application/vnd.github.v3.raw" "${URL_TXT}" > "${TMP_TFLITE_FILES}" 2>/dev/null; then
    TFLITE_FILES_FILE="${TMP_TFLITE_FILES}"
    echo "Successfully downloaded and using PR's version of tflite_files.txt."
  else
    echo "Warning: Could not download from PR commit. Falling back to base branch version."
  fi
fi

if [ ! -f "${TFLITE_FILES_FILE}" ]; then
  echo "Error: ${TFLITE_FILES_FILE} not found!"
  exit 1
fi

# Check for intersection between PR files and TFLite files
CONFLICTS=$(grep -F -x -f "${TFLITE_FILES_FILE}" "${TMP_PR_FILES}" || true)

if [ -n "${CONFLICTS}" ]; then
  echo "The following files should be modified in the upstream Tensorflow repo:"
  echo "${CONFLICTS}"
  exit 1
else
  echo "No TfLite files are modified in the PR. We can proceed."
  exit 0
fi
