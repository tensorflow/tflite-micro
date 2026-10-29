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
# Unified Cadence Xtensa ISS (xt-run) test runner.

set -euo pipefail

# Execute non-ELF targets (such as py_test or sh_test) directly on the host.
if [[ -f "${1:-}" ]]; then
  magic=""
  LC_ALL=C IFS= read -r -N 4 magic < "$1" || true
  if [[ "${magic}" != $'\x7fELF' ]]; then
    exec "$@"
  fi
fi

SCRIPT_NAME="$(basename "$0")"
if [[ -z "${XTENSA_CORE:-}" || -z "${XTENSA_VERSION:-}" ]]; then
  case "${SCRIPT_NAME}" in
    *fusion_f1*)
      : "${XTENSA_VERSION:=RI_2022_9}"
      : "${XTENSA_CORE:=P29A_FusionF1}"
      ;;
    *hifi3z*)
      : "${XTENSA_VERSION:=RI_2022_9}"
      : "${XTENSA_CORE:=P29A_HiFi3z}"
      ;;
    *hifi5*)
      : "${XTENSA_VERSION:=RJ_2024_4}"
      : "${XTENSA_CORE:=PRD_H5_RDO_07_01_2022}"
      ;;
    *vision_p6*)
      : "${XTENSA_VERSION:=RI_2022_9}"
      : "${XTENSA_CORE:=P29A_VP6}"
      ;;
    *)
      echo "ERROR: Cannot infer Xtensa core from runner '${SCRIPT_NAME}'. Set XTENSA_CORE." >&2
      exit 2
      ;;
  esac
fi

GOOGLE3_DIR=""
for root in "${TEST_SRCDIR:-}" "${RUNFILES_DIR:-}" "$0.runfiles"; do
  if [[ -n "${root}" && -d "${root}/google3" ]]; then
    GOOGLE3_DIR="${root}/google3"
    break
  fi
done

TOOLS_PATH=""
if [[ -n "${GOOGLE3_DIR}" && -d "${GOOGLE3_DIR}/third_party/unsupported_toolchains/xtensa/${XTENSA_VERSION}/XtensaTools" ]]; then
  TOOLS_PATH="${GOOGLE3_DIR}/third_party/unsupported_toolchains/xtensa/${XTENSA_VERSION}/XtensaTools"
  CORE_PATH="${GOOGLE3_DIR}/third_party/xtensa/cores/${XTENSA_VERSION}/${XTENSA_CORE}"
  export XTENSA_SYSTEM="${CORE_PATH}/config"
  export XTENSA_CORE
  export XTENSAD_LICENSE_FILE="${CORE_PATH}/misc/license.dat"
  export PATH="${TOOLS_PATH}/bin:${PATH}"
  export LD_LIBRARY_PATH="${TOOLS_PATH}/llvm/lib:${TOOLS_PATH}/Tools/lib64:${TOOLS_PATH}/lib:${TOOLS_PATH}/lib/iss:${TOOLS_PATH}/lib/xcc/lib:${TOOLS_PATH}/lib/tools:${TOOLS_PATH}/tools/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  XT_RUN_BIN="${TOOLS_PATH}/bin/xt-run"
else
  XT_RUN_BIN="${XT_RUN_BIN:-}"
  if [[ -z "${XT_RUN_BIN}" ]]; then
    for root in "${TEST_SRCDIR:-}" "${RUNFILES_DIR:-}" "$0.runfiles"; do
      if [[ -n "${root}" && -d "${root}" ]]; then
        for cand in "${root}"/*xtensa_toolchain*/bin/xt-run; do
          if [[ -s "${cand}" && -x "${cand}" ]]; then
            XT_RUN_BIN="$(readlink -f "${cand}")"
            break 2
          fi
        done
      fi
    done
  fi
  if [[ -z "${XT_RUN_BIN}" ]] && command -v xt-run >/dev/null 2>&1; then
    XT_RUN_BIN="$(readlink -f "$(command -v xt-run)")"
  fi
  if [[ -z "${XT_RUN_BIN}" || ! -x "${XT_RUN_BIN}" ]]; then
    echo "ERROR: xt-run not found in runfiles or PATH." >&2
    exit 2
  fi
  TOOLS_PATH="$(dirname "$(dirname "${XT_RUN_BIN}")")"
  export PATH="${TOOLS_PATH}/bin:${PATH}"
fi

# Configure non-PIE host C compiler wrapper for xt-run --turbo JIT compilation
# so modern PIE-by-default Linux GCC toolchains do not fail Elf32_Rela linking.
TURBO_ARGS=(--turbo)
HOST_CC="$(command -v gcc || command -v cc || true)"
if [[ -n "${HOST_CC}" ]]; then
  TURBO_WORK_DIR="$(mktemp -d "${TEST_TMPDIR:-/tmp}/xtensa_turbo.XXXXXX")"
  trap 'rm -rf "${TURBO_WORK_DIR}"' EXIT
  TURBOSIM_WRAPPER="${TURBO_WORK_DIR}/turbosim_cc.sh"
  INCLUDE_ARG=""
  if [[ -d "${TOOLS_PATH}/include" ]]; then
    INCLUDE_ARG="-I${TOOLS_PATH}/include"
  fi
  cat <<EOF > "${TURBOSIM_WRAPPER}"
#!/usr/bin/env bash
set -e
export LD_LIBRARY_PATH=""
set -f
ARGS=()
for arg in "\$@"; do
  if [[ "\$arg" == -* && "\$arg" == *" "* ]]; then
    for word in \$arg; do
      ARGS+=("\$word")
    done
  else
    ARGS+=("\$arg")
  fi
done
set +f
exec "${HOST_CC}" -fno-pie -fno-pic ${INCLUDE_ARG:+"${INCLUDE_ARG}"} "\${ARGS[@]}"
EOF
  chmod +x "${TURBOSIM_WRAPPER}"
  export TURBO_TMPDIR="${TURBO_WORK_DIR}"
  export TURBOSIM_CC="${TURBOSIM_WRAPPER}"
  export TURBOSIM_CC_O0="${TURBOSIM_WRAPPER}"
  export XTENSA_TURBO_CC_OPT="-O2 -fno-pie -fno-pic"
else
  TURBO_ARGS=()
fi

"${XT_RUN_BIN}" ${TURBO_ARGS[@]+"${TURBO_ARGS[@]}"} --xtensa-core="${XTENSA_CORE}" --exit_with_target_code "$@"
