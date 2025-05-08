# CMSIS-Core Unit Tests

This folder contains a unit test suite that validates CMSIS-Core implementation.
It uses test tools from LLVM, namely LIT and FileCheck, to validate consistency across compilers and devices.

Consult the manual of [FileCheck - Flexible pattern matching file verifier](https://llvm.org/docs/CommandGuide/FileCheck.html)
for details.

## Folder structure

```txt
    📂 Test
    ┣ 📂 src                       Test source files
    ┣ 📂 build.py                  Build wrapper
    ┣ 📂 lit.cfg.py                LIT test suite configuration
    ┣ 📂 requirements.txt          Python dependencies required for build.py script
    ┗ 📂 vcpkg-configuration.json  vcpkg configuration to create virtual environment required for running these tests
```

## Test matrix

Currently, the following build configurations are provided:

1. Compiler
   - Arm Compiler 6 (AC6)
   - GNU Compiler (GCC)
   - LLVM/Clang (Clang)
2. Devices
   - Cortex-M0
   - Cortex-M0+
   - Cortex-M3
   - Cortex-M4
     - w/o FPU
     - with FPU
   - Cortex-M7
     - w/o FPU
     - with SP FPU
     - with DP FPU
   - Cortex-M23
     - w/o security extensions (TrustZone)
     - in secure mode
     - in non-secure mode
   - Cortex-M33 (with FPU and DSP extensions)
     - w/o security extensions (TrustZone)
     - in secure mode
     - in non-secure mode
   - Cortex-M35P (with FPU and DSP extensions)
     - w/o security extensions (TrustZone)
     - in secure mode
     - in non-secure mode
   - Cortex-M55 (with FPU and DSP extensions)
     - in secure mode
     - in non-secure mode
   - Cortex-M85 (with FPU and DSP extensions)
     - in secure mode
     - in non-secure mode
   - Cortex-A5
     - w/o NEON extensions
     - w NEON extensions
   - Cortex-A7
     - w/o NEON extensions
     - w NEON extensions
   - Cortex-A9
     - w/o NEON extensions
     - w NEON extensions
3. Optimization Levels
   - none
   - balanced
   - size
   - speed

## Prerequisites

The following tools are required to build and run the CoreValidation tests:

- [CMSIS-Toolbox 2.1.0](https://artifacts.keil.arm.com/cmsis-toolbox/2.1.0/)*
- [CMake 3.25.2](https://cmake.org/download/)*
- [Ninja 1.10.2](https://github.com/ninja-build/ninja/releases)*
- [Arm Compiler 6.22](https://artifacts.tools.arm.com/arm-compiler/6.22/45/)*
- [GCC Compiler 13.2.1](https://artifacts.keil.arm.com/arm-none-eabi-gcc/13.2.1/)*
- [Clang Compiler 18.1.3](https://github.com/ARM-software/LLVM-embedded-toolchain-for-Arm/releases/tag/release-18.1.3)*
- [Arm Virtual Hardware for Cortex-M based on FastModels 11.22.39](https://artifacts.keil.arm.com/avh/11.22.39/)*
- [Python 3.9](https://www.python.org/downloads/)
- [LLVM FileCheck](https://github.com/llvm/llvm-project/releases/)
  - Ubuntu package `llvm-<version>-tools`
  - MacOS Homebrew formula `llvm`

The executables need to be present on the `PATH`.
For tools distributed via vcpkg (*) this can be achieved automatically:

```bash
 ./CMSIS/Core/Test $ vcpkg activate
```

Install the Python packages required by `build.py`:

```bash
 ./CMSIS/Core/Test $ pip install -r requirements.txt
```

## Execute LIT tests

To build and run the CoreValidation tests for one or more configurations use the following command line.
Select the `<compiler>`, `<device>`, and `<optimize>` level to execute `lit` for.

```bash
 ./CMSIS/Core/Test $ ./build.py -c <compiler> -d <device> -o <optimize> [lit]
```

For example, to execute the LIT tests using GCC for Cortex-M3 with no optimization, run:

```bash
 ./CMSIS/Core/Test $ ./build.py -c GCC -d CM3 -o none lit
[GCC][Cortex-M3][none](lit:run_lit) /opt/homebrew/bin/lit --xunit-xml-output lit.xml -D toolchain=GCC -D device=CM3 -D optimize=none src
[GCC][Cortex-M3][none](lit:run_lit) -- Testing: 49 tests, 10 workers --
 :
[GCC][Cortex-M3][none](lit:run_lit) /opt/homebrew/bin/lit succeeded with exit code 0

Matrix Summary
==============

compiler    device     optimize    lit
----------  ---------  ----------  -----
GCC         Cortex-M3  none        33/33
```

The summary lists the amount of test cases executed and passed.

## Analyse failing test cases

In case of failing test cases, one can run a single test case with verbose output like this:

```bash
 ./CMSIS/Core/Test $ lit -D toolchain=GCC -D device=CM3 -D optimize=none -a src/apsr.c
-- Testing: 1 tests, 1 workers --
PASS: CMSIS-Core :: src/apsr.c (1 of 1)
Script:
--
: 'RUN: at line 2';   arm-none-eabi-gcc -mcpu=cortex-m3 -mfloat-abi=soft -O1 -I ../Include -D CORE_HEADER="core_cm3.h" -c -D __CM3_REV=0x0000U -D __MPU_PRESENT=1U -D __VTOR_PRESENT=1U -D __NVIC_PRIO_BITS=3U -D __Vendor_SysTickConfig=0U -o ./src/Output/apsr.c.o ./src/apsr.c; llvm-objdump --mcpu=cortex-m3 -d ./src/Output/apsr.c.o | FileCheck --allow-unused-prefixes --check-prefixes CHECK,CHECK-THUMB ./src/apsr.c
--
Exit Code: 0
 :
********************

Testing Time: 0.10s
  Passed: 1
```

The output reveales wich commands are chained and their error output if any.

Failing FileCheck requires in detail analysis of the `// CHECK` annotations in the test source file
against the `llvm-objdump` output of the test compilation.

Investigating the disassembly can be done like

```bash
 ./CMSIS/Core/Test $ llvm-objdump --mcpu=cortex-m3 -d ./src/Output/apsr.c.o

./src/Output/apsr.c.o:  file format elf32-littlearm

Disassembly of section .text:

00000000 <get_apsr>:
       0: b082          sub     sp, #0x8
       2: f3ef 8300     mrs     r3, apsr
       6: 9301          str     r3, [sp, #0x4]
       8: b002          add     sp, #0x8
       a: 4770          bx      lr
```

This output is expected to match the test case

```c
    // CHECK: mrs {{r[0-9]+}}, apsr
    volatile uint32_t result = __get_APSR();
```

I.e., the test case expects the `mrs {{r[0-9]+}}, apsr` instruction, additional whitespace is ignored.
