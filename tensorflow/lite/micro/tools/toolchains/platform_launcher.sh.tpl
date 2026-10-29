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

set -euo pipefail

runfiles_dir=""
for base in \
  "${TEST_SRCDIR:-}" \
  "${RUNFILES_DIR:-}" \
  "$0.runfiles" \
  "${0%%.runfiles/*}.runfiles" \
  "${0%%/runfiles/*}/runfiles"; do
  if [[ -n "${base}" && -e "${base}/{{RUNNER_RLOCATION}}" && -e "${base}/{{GUEST_RLOCATION}}" ]]; then
    runfiles_dir="${base}"
    break
  fi
done

if [[ -z "${runfiles_dir}" ]]; then
  echo "ERROR: Could not locate runfiles for {{TARGET_NAME}}." >&2
  exit 2
fi

export RUNFILES_DIR="${runfiles_dir}"
export TEST_SRCDIR="${TEST_SRCDIR:-${runfiles_dir}}"
exec "${runfiles_dir}/{{RUNNER_RLOCATION}}" "${runfiles_dir}/{{GUEST_RLOCATION}}" "$@"
