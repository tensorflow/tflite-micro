# CMSIS-Compiler {#mainpage}

**CMSIS-Compiler** provides software components for retargeting I/O operations in standard C run-time libraries. Following interfaces are supported for retargeting:

 - File interface for reading and writing files.
 - I/O interface for standard I/O stream retargeting (stderr, stdin, stdout).
 - OS interface for multithread safety using an arbitrary RTOS.

![Overview of CMSIS-Compiler](./images/cmsis_compiler_overview.png)

Standard C library functions are platform independent, but the implementations of the low-level I/O and multithreading support are tailored to the target compiler toolchains.

## Access to CMSIS-Compiler

CMSIS-Compiler is maintained in a GitHub repository and is also released as a standalone package in CMSIS Pack format.

 - [**CMSIS-Compiler GitHub Repo**](https://github.com/Arm-Software/CMSIS-Compiler) provides the full source code of CMSIS-Compiler components.
 - [**CMSIS-Compiler Documentation**](https://arm-software.github.io/CMSIS-Compiler/latest/) explains how to use the library and describes the implemented functions in details.
 - [**CMSIS-Compiler Pack**](https://www.keil.arm.com/packs/cmsis-compiler-arm/versions/) delivers CMSIS-Compiler components and examples in [CMSIS-Pack format](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html).

## Key Features and Benefits

 - CMSIS-Compiler allows individual retargeting configuration for common I/O interfaces: file, stderr, stdin, stdout.
 - Implements several ready-to-use retarget variants: into File, Breakpoint, Event Recorder, ITM, as well as user-specific variant.
 - Supports common standard C libraries, such as from Arm Compiler and GCC (Newlib) toolchains. IAR support is planned.
 - Provides implementation templates and examples to get started.
