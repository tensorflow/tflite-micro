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
# Unified Qualcomm Hexagon simulator (hexagon-sim) test runner.

set -euo pipefail

# Execute non-ELF targets (such as py_test or sh_test) directly on the host.
if [[ -f "${1:-}" ]]; then
  magic=""
  LC_ALL=C IFS= read -r -N 4 magic < "$1" || true
  if [[ "${magic}" != $'\x7fELF' ]]; then
    exec "$@"
  fi
fi

if [[ -z "${HEXAGON_SIM:-}" ]]; then
  for root in "${TEST_SRCDIR:-}" "$0.runfiles"; do
    if [[ -n "${root}" && -d "${root}" ]]; then
      for cand in "${root}"/*hexagon_toolchain*/bin/hexagon-sim; do
        if [[ -s "${cand}" && -x "${cand}" ]]; then
          HEXAGON_SIM="$(readlink -f "${cand}")"
          break 2
        fi
      done
    fi
  done
  if [[ -z "${HEXAGON_SIM:-}" ]] && command -v hexagon-sim >/dev/null 2>&1; then
    HEXAGON_SIM="$(command -v hexagon-sim)"
  fi
  if [[ -z "${HEXAGON_SIM:-}" || ! -x "${HEXAGON_SIM}" ]]; then
    echo "ERROR: hexagon-sim not found in runfiles or PATH." >&2
    exit 2
  fi
fi

exec "${HEXAGON_SIM}" --simulated_returnval "$@"
