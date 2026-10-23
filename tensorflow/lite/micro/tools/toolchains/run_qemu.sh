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
# Unified user-mode QEMU test runner for Arm Cortex-M and RISC-V 32 targets.

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
if [[ -z "${QEMU_ARCH:-}" || -z "${QEMU_CPU:-}" ]]; then
  case "${SCRIPT_NAME}" in
    *m0*)
      : "${QEMU_ARCH:=arm}"
      : "${QEMU_CPU:=cortex-m0}"
      ;;
    *m4*)
      : "${QEMU_ARCH:=arm}"
      : "${QEMU_CPU:=cortex-m4}"
      ;;
    *m55*)
      : "${QEMU_ARCH:=arm}"
      : "${QEMU_CPU:=cortex-m55}"
      ;;
    *riscv32*)
      : "${QEMU_ARCH:=riscv32}"
      : "${QEMU_CPU:=rv32}"
      ;;
    *)
      echo "ERROR: Cannot infer QEMU architecture/CPU from runner '${SCRIPT_NAME}'. Set QEMU_ARCH and QEMU_CPU." >&2
      exit 2
      ;;
  esac
fi

if [[ -z "${QEMU_BIN:-}" ]]; then
  RUNFILES_ROOT=""
  if [[ -n "${TEST_SRCDIR:-}" && -d "${TEST_SRCDIR}/google3" ]]; then
    RUNFILES_ROOT="${TEST_SRCDIR}/google3"
  elif [[ -d "$0.runfiles/google3" ]]; then
    RUNFILES_ROOT="$0.runfiles/google3"
  fi

  if [[ -n "${RUNFILES_ROOT}" && -x "${RUNFILES_ROOT}/third_party/qemu/qemu-${QEMU_ARCH}-for-host" ]]; then
    QEMU_BIN="${RUNFILES_ROOT}/third_party/qemu/qemu-${QEMU_ARCH}-for-host"
  elif [[ -n "${RUNFILES_ROOT}" && -x "${RUNFILES_ROOT}/third_party/tflite_micro/tensorflow/lite/micro/tools/toolchains/qemu-${QEMU_ARCH}-for-host" ]]; then
    QEMU_BIN="${RUNFILES_ROOT}/third_party/tflite_micro/tensorflow/lite/micro/tools/toolchains/qemu-${QEMU_ARCH}-for-host"
  elif command -v "qemu-${QEMU_ARCH}" >/dev/null 2>&1; then
    QEMU_BIN="$(command -v "qemu-${QEMU_ARCH}")"
  else
    echo "ERROR: qemu-${QEMU_ARCH} not found in runfiles or PATH." >&2
    exit 2
  fi
fi

exec "${QEMU_BIN}" -cpu "${QEMU_CPU}" "$@"
