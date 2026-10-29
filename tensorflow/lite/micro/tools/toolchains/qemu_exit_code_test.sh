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
#
# Verifies standalone launcher execution (without TEST_SRCDIR/RUNFILES_DIR),
# buffered stdout flushing without trailing newline, and exact non-zero exit
# status propagation under user-mode QEMU.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <pass_launcher_rloc> <fail37_launcher_rloc>" >&2
  exit 2
fi

resolve_rlocation() {
  local rel="$1"
  for base in \
    "${TEST_SRCDIR:-}" \
    "${RUNFILES_DIR:-}" \
    "$0.runfiles" \
    "${0%%.runfiles/*}.runfiles" \
    "${0%%/runfiles/*}/runfiles"; do
    if [[ -n "${base}" && -x "${base}/${rel}" ]]; then
      echo "${base}/${rel}"
      return 0
    fi
  done
  echo "ERROR: Could not resolve rlocation '${rel}'." >&2
  return 1
}

PASS_LAUNCHER="$(resolve_rlocation "$1")"
FAIL37_LAUNCHER="$(resolve_rlocation "$2")"

WORK_DIR="$(mktemp -d "${TEST_TMPDIR:-/tmp}/qemu_exit_test.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

PASS_OUT="${WORK_DIR}/pass.log"
(
  cd "${WORK_DIR}"
  env -u TEST_SRCDIR -u RUNFILES_DIR "${PASS_LAUNCHER}"
) >"${PASS_OUT}" 2>&1
if ! grep -q "QEMU_FLUSH_EXIT_0_OK" "${PASS_OUT}"; then
  echo "FAIL: Un-newlined stdout was not flushed on exit(0)." >&2
  cat "${PASS_OUT}" >&2
  exit 1
fi

FAIL_OUT="${WORK_DIR}/fail37.log"
set +e
(
  cd "${WORK_DIR}"
  env -u TEST_SRCDIR -u RUNFILES_DIR "${FAIL37_LAUNCHER}"
) >"${FAIL_OUT}" 2>&1
RC=$?
set -e
if [[ "${RC}" -ne 37 ]]; then
  echo "FAIL: Expected exit code 37, got ${RC}." >&2
  cat "${FAIL_OUT}" >&2
  exit 1
fi
if ! grep -q "QEMU_FLUSH_EXIT_37_OK" "${FAIL_OUT}"; then
  echo "FAIL: Un-newlined stdout was not flushed on exit(37)." >&2
  cat "${FAIL_OUT}" >&2
  exit 1
fi

echo "PASS: QEMU standalone launcher, stdout flush, and exit-code propagation checks succeeded."
