#!/usr/bin/env bash
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

# Utility script that handles downloading, extracting, and patching third-party
# library dependencies for TensorFlow Lite for Microcontrollers.
# Called with up to five arguments:
# 1 - URL to download from.
# 2 - MD5 checksum to verify the package's integrity. Use md5sum to create one.
# 3 - Path to new folder to unpack the library into.
# 4 - Optional post-download action name (e.g., build_embarc_mli).
# 5 - Optional action parameter.

set -e

build_embarc_mli() {
  if [[ ${ARC_TAGS} =~ "mli20_experimental" ]]; then
    make -C ${1}/lib/make build TCF_FILE=${2} BUILDLIB_DIR=${BUILD_LIB_DIR} GEN_EXAMPLES=0 JOBS=4
  else
    make -j 4 -C ${1}/lib/make TCF_FILE=${2}
  fi
}

# Main function handling the download, verify, extract, and patch process.
download_and_extract() {
  local usage="Usage: download_and_extract URL MD5 DIR [ACTION] [ACTION_PARAM]"
  local url="${1:?${usage}}"
  local expected_md5="${2:?${usage}}"
  local dir="${3:?${usage}}"
  local action=${4}
  local action_param1=${5}  # optional action parameter
  local curl_retries=5

  # Destination already downloaded and verified.
  if [ -d "${dir}" ] && [ -f "${dir}/.download_complete" ]; then
      exit 0
  fi

  command -v curl >/dev/null 2>&1 || {
    echo >&2 "The required 'curl' tool isn't installed. Try 'apt-get install curl'."; exit 1;
  }

  echo "downloading ${url}" >&2
  local parent_dir="$(dirname "${dir}")"
  mkdir -p "${parent_dir}"

  local staging_dir
  staging_dir="$(mktemp -d "${parent_dir}/.tmp_pkg.XXXXXX")"
  local tempdir2
  tempdir2="$(mktemp -d)"
  local tempfile="${staging_dir}/temp_file"

  cleanup() {
    rm -rf "${staging_dir}" "${tempdir2}"
  }
  trap cleanup EXIT INT TERM

  # We've been seeing occasional 56 errors from valid URLs, so set up a retry
  # loop to attempt to recover from them.
  for (( i=1; i<=$curl_retries; ++i )); do
    # We have to use this approach because we normally halt the script when
    # there's an error, and instead we want to catch errors so we can retry.
    set +e
    curl -LsS --fail --retry 5 "${url}" > ${tempfile}
    CURL_RESULT=$?
    set -e

    # Was the command successful? If so, continue.
    if [[ $CURL_RESULT -eq 0 ]]; then
      break
    fi

    # Keep trying if we see the '56' error code.
    if [[ ( $CURL_RESULT -ne 56 ) || ( $i -eq $curl_retries ) ]]; then
      echo "Error $CURL_RESULT downloading '${url}'"
      exit 1
    fi
    sleep 2
  done

  # Check that the file was downloaded correctly using a checksum. Put expected_md5 as "SKIP_MD5_CHECK" to skip this check.
  DOWNLOADED_MD5=$(openssl dgst -md5 ${tempfile} | sed 's/.* //g')
  if [ ${expected_md5} != ${DOWNLOADED_MD5} ] && [ ${expected_md5} != "SKIP_MD5_CHECK" ]; then
    echo "Checksum error for '${url}'. Expected ${expected_md5} but found ${DOWNLOADED_MD5}"
    exit 1
  fi

  # delete anything after the '?' in a url that may mask true file extension
  url=$(echo "${url}" | sed "s/\?.*//")

  if [[ "${url}" == *gz ]]; then
    tar -C "${tempdir2}" -xzf ${tempfile}
  elif [[ "${url}" == *tar.xz ]]; then
    tar -C "${tempdir2}" -xf ${tempfile}
  elif [[ "${url}" == *bz2 ]]; then
    tar -C "${tempdir2}" -xjf ${tempfile}
  elif [[ "${url}" == *zip ]]; then
    unzip -q ${tempfile} -d ${tempdir2}
  else
    echo "Error unsupported archive type. Failed to extract tool after download."
    exit 1
  fi

  rm -f "${tempfile}"

  # If the zip file contains nested directories, extract the files from the
  # inner directory.
  if [ $(find $tempdir2/* -maxdepth 0 | wc -l) = 1 ] && [ -d $tempdir2/* ]; then
    # Unzip to a temp dir, and move the files we want from the tempdir to destination.
    # We want this to be dependent on the folder structure of the zipped file, so --strip-components cannot be used.
    cp -R ${tempdir2}/*/* "${staging_dir}/"
  else
    cp -R ${tempdir2}/* "${staging_dir}/"
  fi

  rm -rf "${tempdir2}"

  # Delete any potential BUILD files, which would interfere with Bazel builds.
  find "${staging_dir}" -type f \( -name 'BUILD' -o -name 'BUILD.bazel' \) -delete

  if [[ ${action} == "build_embarc_mli" ]]; then
    if [[ "${action_param1}" == *.tcf ]]; then
      cp ${action_param1} "${staging_dir}/hw/arc.tcf"
      build_embarc_mli "${staging_dir}" ../../hw/arc.tcf
    else
      build_embarc_mli "${staging_dir}" ${action_param1}
    fi
  elif [[ ${action} ]]; then
    echo "Unknown action '${action}'"
    exit 1
  fi

  touch "${staging_dir}/.download_complete"

  # Check if another process completed the download in the meantime
  if [ -d "${dir}" ] && [ -f "${dir}/.download_complete" ]; then
    rm -rf "${staging_dir}"
    trap - EXIT INT TERM
    exit 0
  fi

  # Atomic promotion
  rm -rf "${dir}"
  mv "${staging_dir}" "${dir}"
  trap - EXIT INT TERM
}

download_and_extract "$1" "$2" "$3" "$4" "$5"
