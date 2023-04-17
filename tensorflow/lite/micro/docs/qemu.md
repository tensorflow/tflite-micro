<!--ts-->
   * [Installation](#installlation)
   * [Software Emulation with QEMU](#software-emulation-with-qemu)
   * [Running Unit Tests](#running-unit-tests)
   * [Useful External Links for QEMU](#useful-external-links-for-qemu)

<!-- Added by: mikebernico, at: Mon April 10 2023 -->

<!--te-->

# Installlation
Our test scripts assume that the non static `user` mode installation of QEMU is
available in the PATH.  For example, if using QEMU for ARM testing, please make
sure `qemu-arm` is installed and available to the test scripts.

You can use `ci/install_qemu.sh` to download, build and install the version of
qemu that is used as part of the CI.

# Software Emulation with QEMU
TensorFlow Lite Micro makes use of [QEMU](https://qemu.org) to
for testing cross compiled tests.

QEMU can quickly test unit tests that are cross compiled for non x64\_86
hardware.

# Running Unit Tests
All unit tests can be ran using
`tensorflow/lite/micro/tools/ci_build/test_cortex_m_qemu.sh` for the cortex-m
processor.

# Useful External Links for QEMU
The current QEMU implementation uses `user` mode.  The documentation for [user
mode is here](https://www.qemu.org/docs/master/user/index.html).

QEMU uses ARM
[semihosting](https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst)
to replace newlib system calls for specific boards with the host OS.  Further
documentation on how this works is contained in `cortex_m_qemu_makefile.inc`.
