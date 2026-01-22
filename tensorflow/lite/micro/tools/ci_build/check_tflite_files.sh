#!/bin/bash
# Checks if the PR modifies TFLite files.
# Inputs:
#   GITHUB_REPOSITORY
#   PR_NUMBER
#   TFLM_BOT_TOKEN

set -e

URL="https://api.github.com/repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/files"
PR_FILES=$(curl -s -X GET -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" $URL | jq -r '.[] | .filename')

# Create a temp file for PR files
TMP_PR_FILES=$(mktemp)
echo "${PR_FILES}" > "$TMP_PR_FILES"

if [ ! -f ci/tflite_files.txt ]; then
  echo "Error: ci/tflite_files.txt not found!"
  rm -f "$TMP_PR_FILES"
  exit 1
fi

# Check for intersection between PR files and TFLite files
# grep -F: Fixed strings
# -x: Match whole line
# -f: Read patterns from file
CONFLICTS=$(grep -F -x -f ci/tflite_files.txt "$TMP_PR_FILES" || true)

rm -f "$TMP_PR_FILES"

if [ -n "$CONFLICTS" ]; then
  echo "The following files should be modified in the upstream Tensorflow repo:"
  echo "$CONFLICTS"
  exit 1
else
  echo "No TfLite files are modified in the PR. We can proceed."
  exit 0
fi