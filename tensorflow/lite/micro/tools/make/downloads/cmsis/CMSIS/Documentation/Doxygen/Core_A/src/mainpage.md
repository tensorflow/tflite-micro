# Overview {#mainpage}

The **CMSIS-Core (Cortex-A)** component implements the basic run-time system for a Cortex-A device and gives the user access to the processor core and the device peripherals.
In detail it defines:

 - **Hardware Abstraction Layer (HAL)** for Cortex-A processor registers with standardized  definitions for the GIC, FPU, MMU, Cache, and core access functions.
 - **System exception names** to interface to system exceptions without having compatibility issues.
 - **Methods to organize header files** that makes it easy to learn new Cortex-A microcontroller products and improve software portability. This includes naming conventions for device-specific interrupts.
 - **Methods for system initialization** to be used by each MCU vendor. For example, the standardized SystemInit() function is essential for configuring the clock system of the device.
 - **Intrinsic functions** used to generate CPU instructions that are not supported by standard C functions.
 - A variable to determine the **system clock frequency** which simplifies the setup of the system timers.

The following sections provide details about the CMSIS-Core (Cortex-A):

 - \ref using_pg describes the project setup and shows a simple program example.
 - \ref templates_pg describes the files of the CMSIS-Core (Cortex-A) in detail and explains how to adapt template files provided by Arm to silicon vendor devices.
 - \ref coreMISRA_Exceptions_pg describes the violations to the MISRA standard.
 - [**API Reference**](modules.html) describes the features and functions of the \ref device_h_pg in detail.
 - [**Data Structures**](annotated.html) describe the data structures of the \ref device_h_pg in detail.

## Access to CMSIS-Core (Cortex-A)

CMSIS-Core is actively maintained in the [**CMSIS 6 GitHub repository**](https://github.com/ARM-software/CMSIS_6) and released as part of the [**CMSIS Software Pack**](../General/cmsis_pack.html).

The following directories and files relevant to CMSIS-Core (Cortex-A) are present in the **ARM::CMSIS** Pack:

Directory                         | Content
:---------------------------------|:------------------------------------------------------------------------
📂 CMSIS                          | CMSIS Base software components folder
 ┣ 📂 Documentation/html/Core_A   | A local copy of this CMSIS-Core (A) documentation
 ┗ 📂 Core                        | CMSIS-Core files
 &emsp;&nbsp; ┣ 📂 Include        | \ref CMSIS_Processor_files.
 &emsp;&emsp;&nbsp; ┗ 📂 a-profile| Header files specific for Arm A-Profile.
 &emsp;&nbsp; ┗ 📂 Template       | \ref template_files_sec

## Processor Support {#ref_v7A}

CMSIS supports a selected subset of [Cortex-A processors](https://www.arm.com/products/silicon-ip-cpu?families=cortex-m&showall=true).

\anchor ref_man_ca_sec
**Cortex-A Technical Reference Manuals**

The following Technical Reference Manuals describe the various Arm Cortex-A processors:

 - [Cortex-A5](https://developer.arm.com/documentation/ddi0433) (Armv7-A architecture)
 - [Cortex-A7](https://developer.arm.com/documentation/ddi0464) (Armv7-A architecture)
 - [Cortex-A9](https://developer.arm.com/documentation/100511) (Armv7-A architecture)

## Tested and Verified Toolchains {#tested_tools_sec}

The \ref templates_pg delivered with this CMSIS-Core release have been tested and verified with the following toolchains:

 - Arm Compiler for Embedded 6.22
 - IAR C/C++ Compiler for Arm 9.40
 - GNU Arm Embedded Toolchain 13.2.1
 - LLVM/Clang 18.3.1
