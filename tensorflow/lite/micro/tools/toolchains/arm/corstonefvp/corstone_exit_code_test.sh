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
# End-to-end verification of Corstone-300 FVP standalone launcher execution,
# .fini_array teardown, exact non-zero exit propagation, and lockup-free Cortex-M55
# fault handling (UDF and MSPLIM STKOF).

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <smoke_rloc> <exit42_rloc> <udf_rloc> <stkof_rloc>" >&2
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

SMOKE_LAUNCHER="$(resolve_rlocation "$1")"
EXIT42_LAUNCHER="$(resolve_rlocation "$2")"
UDF_LAUNCHER="$(resolve_rlocation "$3")"
STKOF_LAUNCHER="$(resolve_rlocation "$4")"

WORK_DIR="$(mktemp -d "${TEST_TMPDIR:-/tmp}/corstone_exit_test.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

# 1. Standalone execution from /tmp without TEST_SRCDIR/RUNFILES_DIR + .fini_array check.
SMOKE_OUT="${WORK_DIR}/smoke.log"
(
  cd "${WORK_DIR}"
  env -u TEST_SRCDIR -u RUNFILES_DIR FVP_TIMEOUT_SEC=60 "${SMOKE_LAUNCHER}"
) >"${SMOKE_OUT}" 2>&1
if ! grep -aq "~~~ALL TESTS PASSED~~~" "${SMOKE_OUT}"; then
  echo "FAIL: Standalone smoke test did not print pass banner." >&2
  cat "${SMOKE_OUT}" >&2
  exit 1
fi
if ! grep -aq "CORSTONE_FINI_ARRAY_OK" "${SMOKE_OUT}"; then
  echo "FAIL: .fini_array destructor output missing on normal exit." >&2
  cat "${SMOKE_OUT}" >&2
  exit 1
fi

# 2. Exact non-zero exit status propagation (exit 42 -> 42).
EXIT42_OUT="${WORK_DIR}/exit42.log"
set +e
FVP_TIMEOUT_SEC=30 "${EXIT42_LAUNCHER}" >"${EXIT42_OUT}" 2>&1
RC=$?
set -e
if [[ "${RC}" -ne 42 ]]; then
  echo "FAIL: Expected exit code 42 from exit_42 guest, got ${RC}." >&2
  cat "${EXIT42_OUT}" >&2
  exit 1
fi
if ! grep -aq "__TFLM_EXIT_CODE=42__" "${EXIT42_OUT}"; then
  echo "FAIL: Missing __TFLM_EXIT_CODE=42__ sentinel." >&2
  cat "${EXIT42_OUT}" >&2
  exit 1
fi

# 3. Undefined instruction UsageFault (Vector 6 -> exit 134, prints CFSR & stacked PC).
UDF_OUT="${WORK_DIR}/udf.log"
set +e
FVP_TIMEOUT_SEC=30 "${UDF_LAUNCHER}" >"${UDF_OUT}" 2>&1
RC=$?
set -e
if [[ "${RC}" -ne 134 ]]; then
  echo "FAIL: Expected exit code 134 (128 + Vector 6) from udf fault, got ${RC}." >&2
  cat "${UDF_OUT}" >&2
  exit 1
fi
if ! grep -aq "FATAL FAULT: Fault_Handler (Vector 6)!" "${UDF_OUT}" || \
   ! grep -aqE "PC=0x[0-9A-F]{8}" "${UDF_OUT}"; then
  echo "FAIL: Missing UsageFault diagnostics or stacked PC in udf output." >&2
  cat "${UDF_OUT}" >&2
  exit 1
fi

# 4. MSPLIM stack overflow UsageFault (CFSR bit 20 UFSR.STKOF = 0x00100000) without LOCKUP timeout.
STKOF_OUT="${WORK_DIR}/stkof.log"
set +e
FVP_TIMEOUT_SEC=30 "${STKOF_LAUNCHER}" >"${STKOF_OUT}" 2>&1
RC=$?
set -e
if [[ "${RC}" -eq 0 || "${RC}" -eq 124 ]]; then
  echo "FAIL: Expected immediate non-zero fault exit on STKOF, got ${RC}." >&2
  cat "${STKOF_OUT}" >&2
  exit 1
fi
CFSR_TOKEN="$(grep -aoE 'CFSR=0x[0-9A-Fa-f]{8}' "${STKOF_OUT}" | tail -n 1 || true)"
if [[ -z "${CFSR_TOKEN}" ]]; then
  echo "FAIL: Missing CFSR dump in STKOF fault output." >&2
  cat "${STKOF_OUT}" >&2
  exit 1
fi
CFSR_VAL="${CFSR_TOKEN#CFSR=}"
if (( (CFSR_VAL & 0x00100000) == 0 )); then
  echo "FAIL: Expected UFSR.STKOF (0x00100000) in ${CFSR_TOKEN}." >&2
  cat "${STKOF_OUT}" >&2
  exit 1
fi

echo "PASS: All Corstone-300 standalone, exit-code, and fault-handler checks succeeded."
