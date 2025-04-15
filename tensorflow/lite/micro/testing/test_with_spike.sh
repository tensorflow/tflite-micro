#!/bin/bash -e

# Parameters:
#  ${1} suffix for qemu binary (e.g. to use qemu-arm ${1} should be arm
#  ${2} architecture to pass to qemu (e.g. cortex-m3)
#  ${3} cross-compiled binary to be emulated
#  ${4} - String that is checked for pass/fail.
#  ${5} - target (cortex_m_qemu etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_ROOT_DIR=${SCRIPT_DIR}/../../../../

TEST_TMPDIR=/tmp/test_${5}
MICRO_LOG_PATH=${TEST_TMPDIR}/${3}
MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt

mkdir -p ${MICRO_LOG_PATH}
spike --isa=rv32gcv ~/rv32imc_zve32x_zvl128b/riscv32-unknown-elf/bin/pk ${1} 2>&1 | tee ${MICRO_LOG_FILENAME}
if [[ ${2} != "non_test_binary" ]]
then
  if grep -q "${2}" ${MICRO_LOG_FILENAME}
  then
    echo "Pass"
    exit 0
  else
    echo "Fail"
    exit 1
  fi
fi
