# CMSIS-Core Device Templates {#templates_pg}

Arm supplies CMSIS-Core device template files for the all supported Cortex-A processors and various compiler vendors. Refer to the list of \ref tested_tools_sec for compliance.

These CMSIS-Core device template files include the following:
 - Register names of the Core Peripherals and names of the Core Exception Vectors.
 - Functions to access core peripherals, cache, MMU and special CPU instructions 
 - Generic startup code and system configuration code.

## CMSIS-Core Processor Files {#CMSIS_Processor_files}

The CMSIS-Core processor files provided by Arm are in the directory .\\CMSIS\\Core\\Include. These header files define all processor specific attributes do not need any modifications.

The `core_<cpu>.h` defines the core peripherals and provides helper functions that access the core registers.

Header File            | Target Processor Core
:----------------------|:-------------------------------
📂 CMSIS/Core/Include  | CMSIS-Core include folder ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Include/))
 ┗ 📄 core_ca.h        | Generics for all supported Cortex-A processors

## Device Examples {#device_examples}

The [Cortex_DFP pack](https://github.com/ARM-software/Cortex_DFP) provides generic device definitions for supported Arm Cortex-A cores and contains corresponding CMSIS-Core device files.

## Template Files {#template_files_sec}

To simplify the creation of CMSIS-Core device files, the following template files are provided that should be extended by the silicon vendor to reflect the actual device and device peripherals.
Silicon vendors add to these template files the following information:
 - **Device Peripheral Access Layer** that provides definitions for device-specific peripherals.
 - **Access Functions for Peripherals** (optional) that provides additional helper functions to access device-specific peripherals.
 - **Interrupt vectors** in the startup file that are device specific.

Template File                       | Description
:-----------------------------------|:----------------------------------------
📂 CMSIS/Core/Template/Device_A     | Folder with CMSIS-Core device file templates ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Template/Device_A/))
 ┣ 📂 Config                        | Template configuration files
 &emsp;&nbsp; ┣ 📄 Device_ac6.sct   | Linker scatter file template for Arm C/C++ Compiler
 &emsp;&nbsp; ┗ 📄 mem_Device.h     |\ref mem_h_pg template
 ┣ 📂 Include                       | Template header files
 &emsp;&nbsp; ┣ 📄 Device.h         | \ref device_h_pg template
 &emsp;&nbsp; ┗ 📄 system_Device.h  | \ref system_Device_h_sec
 ┗ 📂 Source                        | Template C files
 &emsp;&nbsp; ┣ 📄 mmu_Device.c     | \ref mmu_c_pg template
 &emsp;&nbsp; ┣ 📄 startup_Device.c | \ref startup_c_pg template
 &emsp;&nbsp; ┗ 📄 system_Device.c  | \ref system_Device_sec

**Adapt Template Files to a Device**

The following steps describe how to adopt the template files to a specific device or device family.
Copy the complete all files in the template directory and replace:
 - directory name 'Vendor' with the abbreviation for the device vendor  e.g.: NXP.
 - directory name 'Device' with the specific device name e.g.: LPC17xx.
 - in the file names 'Device' with the specific device name e.g.: LPC17xx.

Each template file contains comments that start with \b ToDo: that describe a required modification.
The template files contain place holders:

Placeholder                | To be replaced with
:--------------------------|:----------------------------------------
`<Device>`                 | The specific device name or device family name, for example `LPC17xx`
`<DeviceInterrupt>`        | The specific interrupt name of the device, for example `TIM1` for Timer 1 interrupt
`<DeviceAbbreviation>`     | Short name or abbreviation of the device family, for example `LPC`
`Cortex-A#`                | The specific Cortex-A processor name, for example `Cortex-A9`

The device configuration of the template files is described in detail on the following pages:
 - \subpage startup_c_pg
 - \subpage system_c_pg
 - \subpage device_h_pg
 - \subpage mem_h_pg
 - \subpage mmu_c_pg

\page startup_c_pg Startup File startup_<Device>.c

The \ref startup_c_pg contains:
 - Exception vectors of the Cortex-A Processor with weak functions that implement default routines.
 - The reset handler which is executed after CPU reset and typically calls the \ref SystemInit function.
 - The setup values for the various stack pointers, i.e. per exceptional mode and main stack.

The file exists for each supported toolchain and is the only tool-chain specific CMSIS file.

\section startup_c_sec startup_Device.c Template File

An Arm Compiler specific startup file for an Armv7-A processor like Cortex-A9 is shown below.
The files for other compiler vendors differ slightly in the syntax, but not in the overall structure.

\verbinclude "Source/startup_Device.c"


\page system_c_pg System Configuration Files system_<Device>.c and system_<Device>.h

The \ref system_c_pg provides as a minimum the functions described under \ref system_init_gr.
These functions are device specific and need adaptations. In addition, the file might have
configuration settings for the device such as XTAL frequency or PLL prescaler settings.

For devices with external memory BUS the system_<Device>.c also configures the BUS system.

The silicon vendor might expose other functions (i.e. for power configuration) in the system_<Device>.c file.

In case of additional features the function prototypes need to be added to the system_<Device>.h header file.

\section system_Device_sec system_Device.c Template File

The \ref system_Device_sec is shown below.

\verbinclude "Source/system_Device.c"

\section system_Device_h_sec system_Device.h Template File

The system_<Device>.h header file contains prototypes to access the public functions in the system_<device>.c file. The \ref system_Device_h_sec is shown below.

\verbinclude "Include/system_Device.h"

\page device_h_pg Device Header File \<Device.h>

The \ref device_h_pg contains the following sections that are device specific:
 - \ref interrupt_number_sec provides interrupt numbers (IRQn) for all exceptions and interrupts of the device.
 - \ref core_config_sect reflect the features of the device.
 - \ref device_access definitions for the \ref peripheral_gr to all device peripherals. It contains all data structures and the address mapping for device-specific peripherals.
 - <b>Access Functions for Peripherals (optional)</b> provide additional helper functions for peripherals that are useful for programming of these peripherals. Access Functions may be provided as inline functions or can be extern references to a device-specific library provided by the silicon vendor.

<a href="modules.html">\b Reference </a> describes the standard features and functions of the \ref device_h_pg in detail.

\section interrupt_number_sec Interrupt Number Definition

\ref device_h_pg contains the enumeration \ref IRQn_ID_t that defines all exceptions and interrupts of the device.
For devices implementing an Arm GIC these are defined as:
  - IRQn 0-15 represents software generated interrupts (SGI), local to each processor core.
  - IRQn 16-31 represents private peripheral interrupts (PPI), local to each processor core.
  - IRQn 32-1019 represents shared peripheral interrupts (SPI), routable to all processor cores.
  - IRQn 1020-1023 represents special interrupts, refer to the GIC Architecture Specification.
  
**Example:**

The following example shows the extension of the interrupt vector table for Cortex-A9 class device.

```c
typedef enum IRQn
{
/******  SGI Interrupts Numbers                 ****************************************/
  SGI0_IRQn            =  0,      
  SGI1_IRQn            =  1,
  SGI2_IRQn            =  2,
       :                  :
  SGI15_IRQn           = 15,

/******  Cortex-A9 Processor Exceptions Numbers ****************************************/
  GlobalTimer_IRQn     = 27,        /*!< Global Timer Interrupt                        */
  PrivTimer_IRQn       = 29,        /*!< Private Timer Interrupt                       */
  PrivWatchdog_IRQn    = 30,        /*!< Private Watchdog Interrupt                    */

/******  Platform Exceptions Numbers ***************************************************/
  Watchdog_IRQn        = 32,        /*!< SP805 Interrupt        */
  Timer0_IRQn          = 34,        /*!< SP804 Interrupt        */
  Timer1_IRQn          = 35,        /*!< SP804 Interrupt        */
  RTClock_IRQn         = 36,        /*!< PL031 Interrupt        */
  UART0_IRQn           = 37,        /*!< PL011 Interrupt        */
       :                  :
       :                  :
} IRQn_Type;
```

\section core_config_sect Configuration of the Processor and Core Peripherals

The \ref device_h_pg configures the Cortex-A processor and the core peripherals with <i>\#defines</i>
that are set prior to including the file <b>core_<cpu>.h</b>.

The following tables list the <i>\#defines</i> along with the possible values for each processor core.
If these <i>\#defines</i> are missing default values are used.

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>__CM0_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>__CORTEX_A</td>
      <td>5, 7, 9</td>
      <td>(n/a)</td>
      <td>Core type number</td>
    </tr>
    <tr>
      <td>__FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if an FPU is present or not</td>
    </tr>
    <tr>
      <td>__GIC_PRESENT</td>
      <td>0 ..1 </td>
      <td>Defines if an GIC is present or not</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>__TIM_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a private timer is present or not</td>
    </tr>
    <tr>
      <td>__L2C_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a level 2 cache controller is present or not</td>
    </tr>
</table>

**Example**

The following code exemplifies the configuration of the Cortex-A9 Processor and Core Peripherals.

```c
#define __CA_REV        0x0000U    /*!< Core revision r0p0                          */
#define __CORTEX_A           9U    /*!< Cortex-A9 Core                              */
#define __FPU_PRESENT        1U    /*!< FPU present                                 */
#define __GIC_PRESENT        1U    /*!< GIC present                                 */
#define __TIM_PRESENT        0U    /*!< TIM not present                             */
#define __L2C_PRESENT        0U    /*!< L2C not present                             */
:
:
#include "core_ca.h"               /* Cortex-A processor and core peripherals       */
```


\section core_version_sect CMSIS Version and Processor Information

Defines in the core_<i>cpu</i>.h file identify the version of the CMSIS-Core-A and the processor used.
The following shows the defines in the various core_<i>cpu</i>.h files that may be used in the \ref device_h_pg
to verify a minimum version or ensure that the right processor core is used.

```c
#define __CA_CMSIS_VERSION_MAIN  (5U)                                 /* [31:16] CMSIS Core main version */
#define __CA_CMSIS_VERSION_SUB   (0U)                                 /* [15:0]  CMSIS Core sub version */
#define __CA_CMSIS_VERSION       ((__CA_CMSIS_VERSION_MAIN << 16U) | \
                                   __CA_CMSIS_VERSION_SUB          )  /* CMSIS Core version number */
```

\section device_access Device Peripheral Access Layer

The \ref device_h_pg contains for each peripheral:
 - Register Layout Typedef
 - Base Address
 - Access Definitions

The section \ref peripheral_gr shows examples for peripheral definitions.

\section device_h_sec Device.h Template File

The silicon vendor needs to extend the Device.h template file with the CMSIS features described above.
In addition the \ref device_h_pg may contain functions to access device-specific peripherals.
The \ref system_Device_h_sec which is provided as part of the CMSIS specification is shown below.

\verbinclude "Include/Device.h"


\page mem_h_pg Memory Configuration Files mem_<device>.h

\verbinclude "Config/mem_Device.h"


\page mmu_c_pg Memory Management Unit Files mmu_<device>.c

\verbinclude "Source/mmu_Device.c"
