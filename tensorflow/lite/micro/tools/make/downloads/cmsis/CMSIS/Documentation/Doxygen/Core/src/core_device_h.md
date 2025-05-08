# Device Header File <Device.h> {#device_h_pg}

The Device Header File contains the following functionalities that are device-specific:
 - \ref interrupt_number_sec provides interrupt numbers (IRQn) for all exceptions and interrupts of the device.
 - \ref core_config_sect reflect the features of the device.
 - \ref device_access provides definitions for the \ref peripheral_gr to all device peripherals. It contains all data structures and the address mapping for device-specific peripherals.
 - **Access Functions for Peripherals (optional)** provide additional helper functions for peripherals that are useful for programming of these peripherals. Access Functions may be provided as inline functions or can be extern references to a device-specific library provided by the silicon vendor.

[API Reference](modules.html) describes the standard features and functions of the \ref device_h_pg in details.

## Interrupt Number Definition {#interrupt_number_sec}

\ref device_h_pg contains the enumeration \ref IRQn_Type that defines all exceptions and interrupts of the device.
 - Negative IRQn values represent processor core exceptions (internal interrupts).
 - Positive IRQn values represent device-specific exceptions (external interrupts). The first device-specific interrupt has the IRQn value 0.
   The IRQn values needs extension to reflect the device-specific interrupt vector table in the \ref startup_c_pg.

**Example:**

The following example shows the extension of the interrupt vector table for the LPC1100 device family.

```c
typedef enum IRQn
{
/******  Cortex-M0 Processor Exceptions Numbers ***************************************************/
  NonMaskableInt_IRQn           = -14,      /*!< 2 Non Maskable Interrupt                         */
  HardFault_IRQn                = -13,      /*!< 3 Cortex-M0 Hard Fault Interrupt                 */
  SVCall_IRQn                   = -5,       /*!< 11 Cortex-M0 SVC Interrupt                       */
  PendSV_IRQn                   = -2,       /*!< 14 Cortex-M0 PendSV Interrupt                    */
  SysTick_IRQn                  = -1,       /*!< 15 Cortex-M0 System Tick Interrupt               */

/******  LPC11xx/LPC11Cxx Specific Interrupt Numbers **********************************************/
  WAKEUP0_IRQn                  = 0,        /*!< All I/O pins can be used as wakeup source.       */
  WAKEUP1_IRQn                  = 1,        /*!< There are 13 pins in total for LPC11xx           */
  WAKEUP2_IRQn                  = 2,
                 :       :
                 :       :
  EINT1_IRQn                    = 30,       /*!< External Interrupt 1 Interrupt                   */
  EINT0_IRQn                    = 31,       /*!< External Interrupt 0 Interrupt                   */
} IRQn_Type;
```

## Configuration of the Processor and Core Peripherals {#core_config_sect}

The \ref device_h_pg  configures the Cortex-M or SecurCore processors and the core peripherals with `#define` directives
that are set prior to including the file `core_<cpu>.h`.

The following tables list the <i>\#defines</i> along with the possible values for each processor core.
If these <i>\#defines</i> are missing default values are used.

**Cortex-M0 core** (core_cm0.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM0_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2</td>
      <td>2</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>

**Cortex-M0+ core** (core_cm0plus.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM0PLUS_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2</td>
      <td>2</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>

**Cortex-M3 core** (core_cm3.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM3_REV</td>
      <td>0x0101 | 0x0200</td>
      <td>0x0200</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>4</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>

**Cortex-M4 core** (core_cm4.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM4_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>4</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a FPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>

**Cortex-M7 core** (core_cm7.h)
<table class="cmtable" summary="">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM7_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>4</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>
        If this define is set to 1, then the default <b>SysTick_Config</b> function
        is excluded. In this case, the file <i><b>device.h</b></i>
        must contain a vendor specific implementation of this function.
      </td>
    </tr>
    <tr>
      <td>\ref __FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a FPU is present or not.</td>
    </tr>
    <tr>
      <td>\ref __FPU_DP</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>
        The combination of the defines \ref __FPU_PRESENT and \ref __FPU_DP
       determine whether the FPU is with single or double precision.
      </td>
    </tr>
    <tr>
      <td>\ref __ICACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Instruction Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __DCACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Data Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __DTCM_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Data Tightly Coupled Memory is present or not</td>
    </tr>
</table>

\if ARMSC
**SecurCore SC000 core** (core_sc000.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __SC000_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2</td>
      <td>2</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

\if ARMSC
**SecurCore SC300 core** (core_sc300.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __SC300_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>4</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

\if ARMv8M
**Cortex-M23 core, Armv8-M Baseline core** (core_cm23.h , core_armv8mbl.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __ARMv8MBL_REV or \ref __CM23_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __SAUREGION_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if SAU regions are present or not</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2</td>
      <td>2</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

\if ARMv8M
**Cortex-M33, Cortex-M35P, Armv8-M Mainline core** (core_cm33.h, core_cm35p.h, core_armv8mml.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __ARMv8MML_REV or \ref __CM33_REV or \ref __CM35P_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __SAUREGION_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if SAU regions are present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a FPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>3</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

\if ARMv8M
**Cortex-M55 core, Armv8.1-M Mainline core** (core_cm55.h, core_armv81mml.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __ARMv81MML_REV or \ref __CM55_REV</td>
      <td>0x0000</td>
      <td>0x0000</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __SAUREGION_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if SAU regions are present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a FPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_DP</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>
        The combination of the defines \ref __FPU_PRESENT and \ref __FPU_DP determine
        whether the FPU is with single or double precision.
      </td>
    </tr>
    <tr>
      <td>\ref __ICACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Instruction Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __DCACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Data Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>3</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

\if ARMv8M
**Cortex-M85 core** (core_cm85.h)
<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>\ref __CM85_REV</td>
      <td>0x0001</td>
      <td>0x0001</td>
      <td>Core revision number ([15:8] revision number, [7:0] patch number)</td>
    </tr>
    <tr>
      <td>\ref __MPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a MPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __SAUREGION_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if SAU regions are present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_PRESENT</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Defines if a FPU is present or not</td>
    </tr>
    <tr>
      <td>\ref __FPU_DP</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>
        The combination of the defines \ref __FPU_PRESENT and \ref __FPU_DP determine
        whether the FPU is with single or double precision.
      </td>
    </tr>
    <tr>
      <td>\ref __ICACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Instruction Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __DCACHE_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Data Chache present or not</td>
    </tr>
    <tr>
      <td>\ref __VTOR_PRESENT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Defines if a VTOR register is present or not</td>
    </tr>
    <tr>
      <td>\ref __NVIC_PRIO_BITS</td>
      <td>2 .. 8</td>
      <td>3</td>
      <td>Number of priority bits implemented in the NVIC (device specific)</td>
    </tr>
    <tr>
      <td>\ref __Vendor_SysTickConfig</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Vendor defined <b>SysTick_Config</b> function.</td>
    </tr>
</table>
\endif

**Example**

The following code exemplifies the configuration of the Cortex-M4 Processor and Core Peripherals.

```c
#define __CM4_REV                 0x0001U   /* Core revision r0p1                                 */
#define __MPU_PRESENT             1U        /* MPU present or not                                 */
#define __VTOR_PRESENT            1U        /* VTOR present */
#define __NVIC_PRIO_BITS          3U        /* Number of Bits used for Priority Levels            */
#define __Vendor_SysTickConfig    0U        /* Set to 1 if different SysTick Config is used       */
#define __FPU_PRESENT             1U        /* FPU present or not                                 */
.
.
#include <core_cm4.h>                       /* Cortex-M4 processor and core peripherals           */
#include "system_<Device>.h"                /* Device System Header                               */
```

## Device Peripheral Access Layer {#device_access}

The \ref device_h_pg contains for each peripheral:
 - Register Layout Typedef
 - Base Address
 - Access Definitions

The section \ref peripheral_gr shows examples for peripheral definitions.

## Device.h Template File {#device_h_sec}

CMSIS-Core \ref cmsis_template_files include `Device.h` file that can be used as a starting point for chip vendors to implement the device-specific features required in a Device header file as described above. But the may also contain other functions to access device-specific peripherals.
