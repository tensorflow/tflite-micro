<!--ts-->
   * [Installation](#installlation)
   * [Software Emulation with QEMU](#software-emulation-with-qemu)
   * [Running Unit Tests](#running-unit-tests)
   * [Useful External Links for QEMU](#useful-external-links-for-qemu)

<!-- Added by: mikebernico, at: Mon April 10 2023 -->

<!--te-->

# Installlation

There are two primary ways to set up the testing environment.

## Native Installation

To run the test scripts, you must have a non-static, user-mode installation
of QEMU available in your system's PATH. For example, to test on ARM, the
`qemu-arm` binary must be installed and accessible from your shell.

## Docker Environment

As an alternative to installing dependencies on your host machine,
you can use the provided Docker environment.
For instructions, see the "Run locally for debugging" section in the
[Dockerfile](https://github.com/tensorflow/tflite-micro/blob/main/ci/Dockerfile.micro).

# Software Emulation with QEMU

TensorFlow Lite Micro uses [QEMU](https://qemu.org) to run tests that have
been cross-compiled for architectures other than x86_64. This setup enables
the rapid execution of unit tests for various hardware targets directly
on a standard development machine.

# Running Unit Tests

You can run all Cortex-M unit tests using a single build script:

```bash
tensorflow/lite/micro/tools/ci_build/test_cortex_m_qemu.sh
```

After this script completes the initial build, you can run an individual
test target as follows:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_qemu \
  TARGET_ARCH=cortex-m3 \
  OPTIMIZED_KERNEL_DIR=cmsis_nn
  <TEST_TARGET>
```

# Debugging

You can debug failures either interactively with GDB
or by analyzing the core file after a crash.

## Live GDB Debugging

### Terminal 1: Start QEMU, telling it to wait for a GDB connection on port 1234.

```bash
qemu-arm -cpu cortex-m3 -g 1234 <TEST_BINARY_PATH>
```

### Terminal 2: Launch GDB and connect to the waiting QEMU session.

```bash
$ arm-none-eabi-gdb <TEST_BINARY_PATH>
(gdb) target remote :1234
```

## Core Dump Analysis

Load the test executable and the generated core file into GDB
to inspect the state at the time of the crash.

```bash
$ arm-none-eabi-gdb <TEST_BINARY_PATH> <CORE_FILE_PATH>
(gdb) bt
```

# Useful External Links for QEMU

* The current testing framework uses QEMU's user mode.
  Read [doc](https://www.qemu.org/docs/master/user/index.html)

* QEMU uses ARM [semihosting](https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst)
  to replace newlib system calls for specific boards with the host OS.
  Further documentation on how this works is contained in `cortex_m_qemu_makefile.inc`.
