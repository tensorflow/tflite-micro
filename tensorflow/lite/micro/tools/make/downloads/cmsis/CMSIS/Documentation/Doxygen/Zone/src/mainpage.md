# CMSIS-Zone {#mainpage}

**CMSIS-Zone** is an open source utility that helps to manage software configurations for partitions that have different access permissions to system resources in devices with Arm Cortex-M processors. It implements the infrastructure that is required for:

 - Split of a multi-processor system for single processor views
 - TrustZone setup (SAU, Interrupt assignment to Secure/Non-Secure)
 - Setup of Memory Protection Unit (MPU)
 - Setup of device specific Memory Protection Controller (MPC)
 - Setup of device specific Peripheral Protection Controller (PPC)

## Acess to CMSIS-Zone

CMSIS-Zone is maintained in a GitHub repository and is released as a standalone Eclipse plugin.

 - [**CMSIS-Zone GitHub Repo**](https://github.com/Arm-Software/CMSIS-Zone) contains examples, templates and documentation.
 - [**CMSIS-Pack Eclipse Plug-ins**](https://github.com/ARM-software/cmsis-pack-eclipse/releases/latest) implement the CMSIS-Zone utility.
 - [**CMSIS-Zone Documentation**](https://arm-software.github.io/CMSIS-Zone/latest/) explains in details how to use the CMSIS-Zone in a project.


## Key Features and Benefits

 - CMSIS-Zone reduces the complexity of configuring access permissions in embedded software.
 - Allows to setup of access permissions to memory and peripherals in secure/non-secure modes, and privilege execution levels.
 - Provides graphical (GUI) and command line (CLI) interfaces.
 - Generates the code for the setup of protection hardware such as SAU, MPC, PPC, MPU.
 - Generates the linker scripts for the defined partitions.
 - Includes multiple examples for real hardware.
