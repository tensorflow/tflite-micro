#!/bin/bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

# Check the download path argument
#
# Parameter(s):
#   ${1} - path to the download directory or --no-downloads
#
# Outputs:
# "yes" or "no"
function check_should_download() {
  if [[ ${1} == "--no-downloads" ]]; then
    echo "no"
  else
    echo "yes"
  fi
}

# Show the download URL and MD5 checksum
#
# Parameter(s):
#   ${1} - download URL
#   ${2} - download MD5 checksum
#
# Download scripts require informational output should be on stderr.
function show_download_url_md5() {
  echo >&2 "LIBRARY_URL=${1}"
  echo >&2 "LIBRARY_MD5=${2}"
}

# Compute the MD5 sum.
#
# Parameter(s):
#   ${1} - path to the file
function compute_md5() {
  UNAME_S=`uname -s`
  if [ ${UNAME_S} == Linux ]; then
    tflm_md5sum=md5sum
  elif [ ${UNAME_S} == Darwin ]; then
    tflm_md5sum='md5 -r'
  else
    tflm_md5sum=md5sum
  fi
  ${tflm_md5sum} ${1} | awk '{print $1}'
}

# Check that MD5 sum matches expected value.
#
# Parameter(s):
#   ${1} - path to the file
#   ${2} - expected md5
function check_md5() {
  MD5=`compute_md5 ${1}`

  if [[ ${MD5} != ${2} ]]
  then
    echo "Bad checksum. Expected: ${2}, Got: ${MD5}"
    exit 1
  fi

}

# Create a git repo in a folder.
#
# Parameter(s):
#   $[1} - relative path to folder
create_git_repo() {
  pushd ${1} > /dev/null
  git init . > /dev/null
  git config user.email "tflm@google.com" --local
  git config user.name "TFLM" --local
  git add . >&2 2> /dev/null
  git commit -a -m "Commit for a temporary repository." > /dev/null
  git checkout -b tflm > /dev/null
  popd > /dev/null
}

# Create a new commit with a patch in a folder that has a git repo.
#
# Parameter(s):
#   $[1} - relative path to folder
#   ${2} - path to patch file (relative to ${1})
#   ${3} - commit nessage for the patch
function apply_patch_to_folder() {
  pushd ${1} > /dev/null
  echo >&2 "Applying ${PWD}/${1}/${2} to ${PWD}/${1}"
  git apply ${2}
  git commit -a -m "${3}" > /dev/null
  popd > /dev/null
}


