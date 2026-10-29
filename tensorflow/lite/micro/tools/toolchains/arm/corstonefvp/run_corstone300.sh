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
# Unified Arm Corstone-300 + Ethos-U55 Fixed Virtual Platform (FVP) test runner.

set -euo pipefail

# Execute non-ELF targets (such as py_test or sh_test) directly on the host.
if [[ -f "${1:-}" ]]; then
  magic=""
  LC_ALL=C IFS= read -r -N 4 magic < "$1" || true
  if [[ "${magic}" != $'\x7fELF' ]]; then
    exec "$@"
  fi
fi

if [[ -z "${FVP_BIN:-}" ]]; then
  for root in "${TEST_SRCDIR:-}" "${RUNFILES_DIR:-}" "$0.runfiles"; do
    if [[ -n "${root}" && -d "${root}" ]]; then
      for cand in \
        "${root}"/*/bin/FVP_Corstone_SSE-300_Ethos-U55 \
        "${root}"/*/*/bin/FVP_Corstone_SSE-300_Ethos-U55 \
        "${root}"/google3/*/corstonefvp/bin/FVP_Corstone_SSE-300_Ethos-U55; do
        if [[ -s "${cand}" && -x "${cand}" ]]; then
          FVP_BIN="$(readlink -f "${cand}")"
          break 2
        fi
      done
    fi
  done

  if [[ -z "${FVP_BIN:-}" ]]; then
    if [[ -x "/opt/arm/fvp/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55" ]]; then
      FVP_BIN="/opt/arm/fvp/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"
    elif command -v FVP_Corstone_SSE-300_Ethos-U55 >/dev/null 2>&1; then
      FVP_BIN="$(command -v FVP_Corstone_SSE-300_Ethos-U55)"
    else
      echo "ERROR: FVP_Corstone_SSE-300_Ethos-U55 not found in runfiles or PATH." >&2
      exit 2
    fi
  fi
fi

BIN_DIR="$(dirname "${FVP_BIN}")"
PARENT_DIR="$(dirname "${BIN_DIR}")"
FVP_LD_PATH="${BIN_DIR}"
for dir in "${PARENT_DIR}/lib" "${PARENT_DIR}/plugins"; do
  if [[ -d "${dir}" ]]; then
    FVP_LD_PATH="${FVP_LD_PATH}:${dir}"
  fi
done
export LD_LIBRARY_PATH="${FVP_LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CMD_PREFIX=()
if [[ -x "/usr/grte/v5/lib64/ld-linux-x86-64.so.2" ]]; then
  CMD_PREFIX=(
    "/usr/grte/v5/lib64/ld-linux-x86-64.so.2"
    "--library-path"
    "${LD_LIBRARY_PATH}:/usr/grte/v5/lib64:/lib64"
  )
fi

ETHOSU_NUM_MACS="${ETHOSU_NUM_MACS:-256}"
if [[ -z "${FVP_TIMEOUT_SEC:-}" ]]; then
  if [[ -n "${TEST_TIMEOUT:-}" && "${TEST_TIMEOUT}" -gt 15 ]]; then
    FVP_TIMEOUT_SEC=$((TEST_TIMEOUT - 10))
  else
    FVP_TIMEOUT_SEC=180
  fi
fi
HOST_KILL_SEC=$((FVP_TIMEOUT_SEC + 5))

LOG_FILE="$(mktemp "${TEST_TMPDIR:-/tmp}/fvp_corstone300.XXXXXX.log")"
trap 'rm -f "${LOG_FILE}"' EXIT

set +e
timeout -k 10 "${HOST_KILL_SEC}" "${CMD_PREFIX[@]}" "${FVP_BIN}" \
  -C cpu0.CFGDTCMSZ=13 \
  -C cpu0.CFGITCMSZ=13 \
  -C cpu0.INITSVTOR=0x0 \
  -C mps3_board.sse300.iotss3_systemcontrol.INITSVTOR_RST=0x0 \
  -C "ethosu.num_macs=${ETHOSU_NUM_MACS}" \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1 \
  --timelimit "${FVP_TIMEOUT_SEC}" \
  --stat \
  "$@" 2>&1 | tee "${LOG_FILE}"
FVP_STATUS="${PIPESTATUS[0]}"
set -e

SENTINEL="$(grep -aoE '__TFLM_EXIT_CODE=-?[0-9]+__' "${LOG_FILE}" | tail -n 1 || true)"
if [[ -z "${SENTINEL}" ]]; then
  if [[ "${FVP_STATUS}" -ne 0 ]]; then
    echo "ERROR: Corstone-300 FVP terminated with status ${FVP_STATUS} before guest exit." >&2
    exit "${FVP_STATUS}"
  fi
  echo "ERROR: Guest did not emit __TFLM_EXIT_CODE=<n>__ sentinel on Corstone-300 FVP." >&2
  exit 1
fi

GUEST_CODE="${SENTINEL#__TFLM_EXIT_CODE=}"
GUEST_CODE="${GUEST_CODE%__}"
if [[ "${GUEST_CODE}" -ne 0 ]]; then
  echo "ERROR: Guest exited with status ${GUEST_CODE} on Corstone-300 FVP." >&2
  if [[ "${GUEST_CODE}" -ge 1 && "${GUEST_CODE}" -le 255 ]]; then
    exit "${GUEST_CODE}"
  fi
  exit 1
fi

if [[ "${FVP_STATUS}" -ne 0 ]]; then
  exit "${FVP_STATUS}"
fi
