# CMSIS-Core Device Files {#cmsis_device_files}

The **CMSIS-Core Device Files** implement CMSIS-Core support for a specific microcontroller device or family of devices. These files are typically provided by the device vendor and use \ref cmsis_standard_files via \#include directives.

CMSIS-Core specifies the organization of CMSIS-Core Device Files, defines their functionalities, and provides unified naming conventions. This brings   following benefits:
 - Simplified device support for vendors with fast scaling for device families and variants on all Arm Cortex-M cores.
 - Uniform workflows and experience for application developers.

Following CMSIS-Core Device Files are defined:
 - \subpage system_c_pg
 - \subpage device_h_pg
 - \subpage startup_c_pg
 - \subpage linker_sct_pg
\if ARMv8M
 - \subpage partition_h_pg
 - \subpage partition_gen_h_pg
\endif

\ref cmsis_files_dfps explains how to distribute \ref cmsis_device_files in CMSIS-Pack format.

## Device Template Files {#cmsis_template_files}

CMSIS-Core includes the template files that simplify the creation of CMSIS-Core Device Files for a specific device variant.

Silicon vendors typically need to add to these template files the following information:
 - **Device Peripheral Access Layer** that provides definitions for device-specific peripherals.
 - **Access Functions for Peripherals** (optional) that provides additional helper functions to access device-specific peripherals.
 - **Interrupt vectors** in the startup file that are device specific.

Template File                       | Description
:-----------------------------------|:----------------------------------------
📂 CMSIS/Core/Template/Device_M     | Folder with CMSIS-Core device file templates ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Template/Device_M/))
 ┣ 📂 Config                        | Template configuration files
 &emsp;&nbsp; ┣ 📄 Device_ac6.sct   | \ref linker_sct_pg template
 &emsp;&nbsp; ┣ 📄 Device_gcc.ld    | Linker description file for GNU GCC Arm Embedded Compiler
 &emsp;&nbsp; ┗ 📄 partition_Device.h |\ref partition_h_pg template
 ┣ 📂 Include                       | Template header files
 &emsp;&nbsp; ┣ 📄 Device.h         | \ref device_h_pg template
 &emsp;&nbsp; ┗ 📄 system_Device.h  | \ref system_Device_sec
 ┗ 📂 Source                        | Template C files
 &emsp;&nbsp; ┣ 📄 startup_Device.c | \ref startup_c_pg template
 &emsp;&nbsp; ┗ 📄 system_Device.c  | \ref system_Device_sec

**Adapt Template Files to a Device**

Each template file contains comments that start with **ToDo:** and describe required modifications.

The templates contain several placeholders that need to be replaced when creating CMSIS-Core device files:

Placeholder                | To be replaced with
:--------------------------|:----------------------------------------
`<Device>`                 | The specific device name or device family name, for example `LPC17xx`
`<DeviceInterrupt>`        | The specific interrupt name of the device, for example `TIM1` for Timer 1 interrupt
`<DeviceAbbreviation>`     | Short name or abbreviation of the device family, for example `LPC`
`Cortex-M#`                | The specific Cortex-M processor name, for example `Cortex-M3`

Device files for the \ref device_examples can be also a good reference point for implementing CMSIS-Core support for specific devices.
