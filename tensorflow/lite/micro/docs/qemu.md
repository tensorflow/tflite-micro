<!--ts-->
   * [Software Emulation with QEMU](#software-emulation-with-qemu)
   * [Installation](#installation)
   * [Running Unit Tests](#running-unit-tests)
   * [Useful External Links for QEMU](#useful-external-links-for-qemu)

<!-- Added by: mikebernico, at: Mon April 10 2023 -->

<!--te-->

# Software Emulation with QEMU
TensorFlow Lite Micro makes use of [QEMU](https://qemu.org) to
for testing cross compiled tests.

QEMU can quickly test unit tests that are cross compiled for non x64_86 
hardware.  Currently QEMU is used in the project continuous integration
to test cross compiled unit tests against ARM Cortex M3. 


# Installation
*Requirements*
```bash
  sudo apt-get install git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev ninja-build  
```
  
QEMU can be installed with the script `tensorflow/lite/micro/tools/make/download_qemu.sh`.

This script will install and compile QEMU into the path `tensorflow/lite/micro/tools/make/downloads/qemu` and the binary will be located in the `build` folder within that path.

# Running Unit Tests
All unit tests can be ran using `tensorflow/lite/micro/tools/ci_build/test_qemu.sh` for the cortex-m3 processor.  

Alternatively they can be run using other cortex hardware  by using the Makefile from the root directory as

`make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m3 test`

Where cortex-m3 can be replaced with any other supported cortex processor (See `cortex_m_generic_makefile.inc` for a list of supported processors.)

# Useful External Links for QEMU
The current QEMU implementation uses `user` mode.  The documentation for [user mode is here](https://www.qemu.org/docs/master/user/index.html).

QEMU uses ARM [semihosting](https://github.com/ARM-software/abi-aa/blob/main/semihosting/semihosting.rst) to replace newlib system calls for specific boards with the host OS.  Further documentation on how this works is contained in `cortex_m_generic_makefile.inc` as well as `tensorflow/lite/micro/cortex_m_generic/`.
