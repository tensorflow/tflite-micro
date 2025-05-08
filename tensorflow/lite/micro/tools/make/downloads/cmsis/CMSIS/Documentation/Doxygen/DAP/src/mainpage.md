# CMSIS-DAP {#mainpage}

**CMSIS-DAP** is a protocol specification and an open-source firmware implementation that provides standardized access to the CoreSight Debug Access Port ([DAP](https://developer.arm.com/documentation/102585/0000/what-is-a-debug-access-port))
 available on many Arm Cortex processors as part of the [CoreSight Debug and Trace](https://developer.arm.com/ip-products/system-ip/coresight-debug-and-trace) functionality.

![Overview of CMSIS-DAP](./images/cmsis_dap_interface.png)

CMSIS-DAP supports target devices that contain one or more Cortex processors.

A processor device exposes Debug Access Port (DAP) typically either with a 5-pin JTAG or with a 2-pin Serial Wired Debug (SWD) interface that is used as a physical communication with a debug unit.

CMSIS-DAP provides the interface firmware for a debug unit that implements the communication between the debug port of the device and USB port. With it, a software debug tool that runs on a host computer can connect via USB and the debug unit to the device where the application firwmare gets executed.

## Access to CMSIS-DAP

 - [**CMSIS-DAP GitHub Repo**](https://github.com/ARM-software/CMSIS-DAP) provides the full source code.
 - [**CMSIS-DAP Documentation**](https://arm-software.github.io/CMSIS-DAP/latest/) describes the implemented functions in details.

## Key Features and Benefits

 - Provides a standardized interface for debuggers. Interfaces to many standard debuggers is available.
 - Access to CoreSight registers of all Cortex processor architectures (Cortex-A/R/M).
 - Connects via 5-pin JTAG or 2-pin Serial Wire Debug (SWD).
 - Supports multi-core debugging.
 - Supports Serial Wire Output (SWO) of Cortex-M devices.
 - Easy to deploy to debug units based on Cortex-M microcontrollers.
 - Debug unit may be integrated on an evaluation board.
 - Using USB bulk transfers avoids driver installation on host PC.
 - Supports time-critical JTAG or SWD command execution.
 - Supports Test Domain Timer for time measurement using the debug unit.
 - Supports UART communication port, which can be routed to USB COM Port (optional) or native CMSIS-DAP commands.
