[![Version](https://img.shields.io/github/v/release/arm-software/CMSIS_6)](https://github.com/ARM-software/CMSIS_6/releases/latest) [![License](https://img.shields.io/github/license/arm-software/CMSIS_6)](https://github.com/ARM-software/CMSIS_6/blob/main/LICENSE)

# CMSIS Version 6

> **Note:** The branch *main* of this GitHub repository contains our current state of development and gives you contiguous access to the CMSIS development for review, feedback, and contributions via pull requests. For stable versions ready for productive use please refer to tagged releases, like [![Version](https://img.shields.io/github/v/release/arm-software/CMSIS_6?display_name=release&label=%20&sort=semver)](https://github.com/ARM-software/CMSIS_6/releases/latest).

## Useful Links

- [**Documentation of latest release**](https://arm-software.github.io/CMSIS_6/) -  access to the CMSIS user's manual.
- [**CMSIS Components**](https://arm-software.github.io/CMSIS_6/latest/General/index.html#cmsis_components) - overview of software, tools, and specification.
- [**Raise Issues**](https://github.com/ARM-software/CMSIS_6#issues-and-labels) - to provide feedback or report problems.
- [**Documentation of main branch**](https://arm-software.github.io/CMSIS_6/main/General/index.html) - updated from time to time (use [Generate CMSIS Pack for Release](https://github.com/ARM-software/CMSIS_6#generate-cmsis-pack-for-release) for local generation).

## Other related GitHub repositories

| Repository                  | Description                                               |
|:--------------------------- |:--------------------------------------------------------- |
| [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP)                      | Compute library for various data types: fixed-point (fractional q7, q15, q31) and single precision floating-point (32-bit).
| [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)                        | Software library of efficient neural network kernels optimized for Arm Cortex-M processors.
| [CMSIS-FreeRTOS](https://github.com/arm-software/CMSIS-FreeRTOS)            | CMSIS adoption of FreeRTOS including CMSIS-RTOS2 API layer.
| [CMSIS-RTX](https://github.com/arm-software/CMSIS-rtx)                      | Keil RTX Real-Time Operating System (CMSIS-RTOS2 native implementation).
| [CMSIS-Driver](https://github.com/arm-software/CMSIS-Driver)                | Generic MCU driver implementations and templates for Ethernet MAC/PHY and Flash.  |
| [CMSIS-Driver_Validation](https://github.com/ARM-software/CMSIS-Driver_Validation) | CMSIS-Driver Validation can be used to verify CMSIS-Driver in a user system |
| [cmsis-pack-eclipse](https://github.com/ARM-software/cmsis-pack-eclipse)    | CMSIS-Pack Management for Eclipse reference implementation Pack support  |
| [CMSIS-Zone](https://github.com/ARM-software/CMSIS-Zone)                    | CMSIS-Zone Utility along with example projects and FreeMarker templates         |
| [NXP_LPC](https://github.com/ARM-software/NXP_LPC)                          | CMSIS Driver Implementations for the NXP LPC Microcontroller Series       |
| [mdk-packs](https://github.com/mdk-packs)                                   | IoT cloud connectors as trail implementations for MDK (help us to make it generic)|
| [trustedfirmware.org](https://www.trustedfirmware.org/)                     | Arm Trusted Firmware provides a reference implementation of secure world software for Armv8-A and Armv8-M.|

## Directory Structure

Directory                                      | Content
:----------------------------------------------|:---------------------------------------------------------
[CMSIS/Core](./CMSIS/Core)                     | CMSIS-Core related files (for release)
[CMSIS/CoreValidation](./CMSIS/CoreValidation) | Validation for Core(M) and Core(A) (NOT part of pack release)  
[CMSIS/Driver](./CMSIS/Driver)                 | CMSIS-Driver API headers and template files
[CMSIS/RTOS2](./CMSIS/RTOS2)                   | RTOS v2 related files (for Cortex-M & Armv8-M)
[CMSIS/Documentation](./CMSIS/Documentation)   | Doxygen source of the users guide (NOT part of pack release)  

## Generate CMSIS Pack for Release

This GitHub development repository lacks pre-built libraries of various software components (RTOS, RTOS2).
In order to generate a full pack one needs to have the build environment available to build these libraries.
This causes some sort of inconvenience. Hence the pre-built libraries may be moved out into separate pack(s)
in the future.

To build a complete CMSIS pack for installation the following additional tools are required:

- **doxygen.exe**    Version: 1.9.6 (Documentation Generator)
- **mscgen.exe**     Version: 0.20  (Message Sequence Chart Converter)
- **7z.exe (7-Zip)** Version: 16.02 (File Archiver)

Using these tools, you can generate on a Windows PC:

- **CMSIS Documentation** using the shell script **gen_doc.sh** (located in ./CMSIS/Documentation/Doxygen).
- **CMSIS Software Pack** using the shell script **gen_pack.sh**.

## License

Arm CMSIS is licensed under [![License](https://img.shields.io/github/license/arm-software/CMSIS_6?label)](https://github.com/ARM-software/CMSIS_6/blob/main/LICENSE).

## Contributions and Pull Requests

Contributions are accepted under [![License](https://img.shields.io/github/license/arm-software/CMSIS_6?label)](https://github.com/ARM-software/CMSIS_6/blob/main/LICENSE). Only submit contributions where you have authored all of the code.

### Issues and Labels

Please feel free to raise an [issue on GitHub](https://github.com/ARM-software/CMSIS_6/issues)
to report misbehavior (i.e. bugs) or start discussions about enhancements. This
is your best way to interact directly with the maintenance team and the community.
We encourage you to append implementation suggestions as this helps to decrease the
workload of the very limited maintenance team.

We will be monitoring and responding to issues as best we can.
Please attempt to avoid filing duplicates of open or closed items when possible.
In the spirit of openness we will be tagging issues with the following:

- **bug** – We consider this issue to be a bug that will be investigated.
- **wontfix** - We appreciate this issue but decided not to change the current behavior.
- **enhancement** – Denotes something that will be implemented soon.
- **future** - Denotes something not yet schedule for implementation.
- **out-of-scope** - We consider this issue loosely related to CMSIS. It might by implemented outside of CMSIS. Let us know about your work.
- **question** – We have further questions to this issue. Please review and provide feedback.
- **documentation** - This issue is a documentation flaw that will be improved in future.
- **review** - This issue is under review. Please be patient.
- **DONE** - We consider this issue as resolved - please review and close it. In case of no further activity this issues will be closed after a week.
- **duplicate** - This issue is already addressed elsewhere, see comment with provided references.
- **Important Information** - We provide essential information regarding planned or resolved major enhancements.
