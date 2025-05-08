# Using CMSIS-Core {#using_pg}

[TOC]

To use the CMSIS-Core (Cortex-M) in an embedded software project at the following \ref cmsis_device_files need to be added to the application:

 - \ref startup_c_pg with reset handler and exception vectors.
 - \ref system_c_pg with general device configuration (i.e. for clock and BUS setup).
 - \ref device_h_pg gives access to processor core and all peripherals.

![Using CMSIS-Core (Cortex-M) in a project](./images/CMSIS_CORE_Files_USER.png)

The \ref startup_c_pg is executed after reset and calls `SystemInit()` in the reset hander. After the system initialization the control is transferred to the C/C++ run-time library which performs initialization and calls the `main` function in the user code. In addition the \ref startup_c_pg contains all exception and interrupt vectors and implements a default function for every interrupt. It may also contain stack and heap configurations for the user application.

The \ref system_c_pg performs the setup for the processor clock. The variable \ref SystemCoreClock indicates the CPU clock speed. In addition the file may contain functions for the memory BUS setup and clock re-configuration.

> **Note**
> - The files \ref startup_c_pg and \ref system_c_pg may require application specific adaptations and therefore should be copied into the application project folder prior configuration.

The \ref device_h_pg provides access to the following device-specific functionalities:

 - \ref peripheral_gr provides a standardized register layout for all peripherals. Optionally functions for device-specific peripherals may be available.
 - \ref NVIC_gr can be accessed with standardized symbols and functions for the Nested Interrupt Vector Controller (NVIC) are provided.
 - \ref intrinsic_CPU_gr allow to access special instructions, for example for activating sleep mode or the NOP instruction.
 - \ref intrinsic_SIMD_gr provide access to the DSP-oriented instructions.
 - \ref SysTick_gr function to configure and start a periodic timer interrupt.
 - \ref ITM_Debug_gr are functions that allow printf-style I/O via the CoreSight Debug Unit and ITM communication.

## Usage in CMSIS-Packs {#using_packs}

The easiest way to use CMSIS-Core in a project is with CMSIS Packs.

The \ref cmsis_device_files are typically provided in a [CMSIS Device Family Pack (DFP)](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/cp_PackTutorial.html#createPack_DFP) that is maintained by the chip vendor for the target device family. The list of public CMSIS packs (including DFPs) can be found at [keil.arm.com/packs](https://www.keil.arm.com/packs/).

A Device Family Pack (DFP) usually has a requirement for using the **CMSIS:CORE** component from the [CMSIS Software pack](../General/cmsis_pack.html) that contains the \ref cmsis_standard_files. In such case the CMSIS Software pack needs to be installed as well.

The files \ref startup_c_pg and \ref system_c_pg are typically provided in the DFP as part of **Device** class in the **Startup** group and are defined as configuration files, meaning they are copied from the pack into a project folder and can be modifed there if necessary.

The use of \ref device_h_pg can be abstracted with the `#define CMSIS_header_file` provided in [RTE_Components.h](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/cp_Packs.html#cp_RTECompH). This allows to have uniform include code in the application independent of the target device.

```c
#include "RTE_Components.h"                      // include information about project configuration
#include CMSIS_device_header                     // include <Device>.h file
```

Thereafter, the functions described under [API Reference](modules.html) can be used in the application.

For example, the following files are provided by the STM32F10x device family pack:

File                         | Description
:---------------------------------|:------------------------------------------------------------------------
".\Device\Source\ARM\startup_stm32f10x_cl.s" | \ref startup_c_pg for the STM32F10x device variants
".\Device\Source\system_stmf10x.c"  | \ref system_c_pg for the STM32F10x device families
".\Device\Include\stm32f10x.h"      | \ref device_h_pg for the STM32F10x device families

\ref cmsis_files_dfps provides more information on how CMSIS-Core files can be delivered in CMSIS Packs.

## Usage Examples {#usage_examples}

**Examples**

 - \ref using_basic is a simple example that shows the usage of the CMSIS layer.
 - \ref using_vtor shows how to remap the interrupt vector table.
 - \ref using_arm explains how to use CMSIS-Core (Cortex-M) for Arm processors.
 - \ref using_ARM_Lib_sec explains how to create libraries that support various Cortex-M cores.

Also see \ref using_TrustZone_pg that details CMSIS-Core support for Arm TrusZone operation on Cortex-M.

### Basic CMSIS Example {#using_basic}

A typical example for using the CMSIS layer is provided below. The example is based on a STM32F10x Device.

```c
#include <stm32f10x.h>                           // File name depends on device used
 
uint32_t volatile msTicks;                       // Counter for millisecond Interval
 
void SysTick_Handler (void) {                    // SysTick Interrupt Handler
  msTicks++;                                     // Increment Counter
}
 
void WaitForTick (void)  {
  uint32_t curTicks;
 
  curTicks = msTicks;                            // Save Current SysTick Value
  while (msTicks == curTicks)  {                 // Wait for next SysTick Interrupt
    __WFE ();                                    // Power-Down until next Event/Interrupt
  }
}
 
void TIM1_UP_IRQHandler (void) {                 // Timer Interrupt Handler
  ;                                              // Add user code here
}
 
void timer1_init(int frequency) {                // Set up Timer (device specific)
  NVIC_SetPriority (TIM1_UP_IRQn, 1);            // Set Timer priority
  NVIC_EnableIRQ (TIM1_UP_IRQn);                 // Enable Timer Interrupt
}
 
 
void Device_Initialization (void)  {             // Configure & Initialize MCU
  if (SysTick_Config (SystemCoreClock / 1000)) { // SysTick 1mSec
       : // Handle Error 
  }
  timer1_init ();                                // setup device-specific timer
}
 
 
// The processor clock is initialized by CMSIS startup + system file
void main (void) {                               // user application starts here
  Device_Initialization ();                      // Configure & Initialize MCU
  while (1)  {                                   // Endless Loop (the Super-Loop)
    __disable_irq ();                            // Disable all interrupts
    Get_InputValues ();                          // Read Values
    __enable_irq ();                             // Enable all interrupts 
    Calculation_Response ();                     // Calculate Results
    Output_Response ();                          // Output Results
    WaitForTick ();                              // Synchronize to SysTick Timer
  }
}
```

### Using Interrupt Vector Remap {#using_vtor}

Most Cortex-M processors provide VTOR register for remapping interrupt vectors. The following example shows a typical use case where the interrupt vectors are copied to RAM and the `SysTick_Handler` is replaced.

```c
#include "ARMCM3.h"                     // Device header
 
#define VECTORTABLE_SIZE        (240)    /* size of the used vector tables    */
                                         /* see startup file startup_ARMCM3.c */
#define VECTORTABLE_ALIGNMENT   (0x100U) /* 16 Cortex + 32 ARMCM3 = 48 words  */
                                         /* next power of 2 = 256             */

/* externals from startup_ARMCM3.c */
extern uint32_t __VECTOR_TABLE[VECTORTABLE_SIZE];        /* vector table ROM  */

/* new vector table in RAM, same size as vector table in ROM */
uint32_t vectorTable_RAM[VECTORTABLE_SIZE] __attribute__(( aligned (VECTORTABLE_ALIGNMENT) ));

 
/*----------------------------------------------------------------------------
  SysTick_Handler
 *----------------------------------------------------------------------------*/
volatile uint32_t msTicks = 0;                        /* counts 1ms timeTicks */
void SysTick_Handler(void) {
  msTicks++;                                             /* increment counter */
}
 
/*----------------------------------------------------------------------------
  SysTick_Handler (RAM)
 *----------------------------------------------------------------------------*/
volatile uint32_t msTicks_RAM = 0;                    /* counts 1ms timeTicks */
void SysTick_Handler_RAM(void) {
  msTicks_RAM++;                                      /* increment counter */
}
 
/*----------------------------------------------------------------------------
  MAIN function
 *----------------------------------------------------------------------------*/
int main (void) {
  uint32_t i;
   
  for (i = 0; i < VECTORTABLE_SIZE; i++) {
    vectorTable_RAM[i] = __VECTOR_TABLE[i];       /* copy vector table to RAM */
  }
                                                   /* replace SysTick Handler */
  vectorTable_RAM[SysTick_IRQn + 16] = (uint32_t)SysTick_Handler_RAM;
  
  /* relocate vector table */ 
  __disable_irq();
    SCB->VTOR = (uint32_t)&vectorTable_RAM;
  __DSB();
  __enable_irq();
 
  SystemCoreClockUpdate();                        /* Get Core Clock Frequency */
  SysTick_Config(SystemCoreClock / 1000ul); /* Setup SysTick Timer for 1 msec */
   
  while(1);
}
```

### Use generic Arm Devices {#using_arm}

Test and example projects of many software components have a need for implementations that are independent from specific device vendors but still have adaptations for various Arm Cortex-M cores to benefit from their architectural differenceis.

The [Cortex_DFP pack](https://github.com/ARM-software/Cortex_DFP) provides generic device definitions for standard Arm Cortex-M cores and contains corresponding \ref cmsis_device_files. These generic Arm devices can be used as a target for embedded programs, with execution, for example, on processor simulation models.

Validation suits and example projects for such components as [CMSIS-DSP](../DSP/index.html), [CMSIS-RTOS](../RTOS2/index.html) and [CMSIS-Core](index.html) itself use that approach already.

### Create generic libraries {#using_ARM_Lib_sec}

The CMSIS Processor and Core Peripheral files allow also to create generic libraries. 
The [CMSIS-DSP libraries](../DSP/index.html) are an example for such a generic library.

To build a generic Library set the define `__CMSIS_GENERIC` and include the relevant `core_<cpu>.h` CMSIS CPU & Core Access header file for the processor.

The define `__CMSIS_GENERIC` disables device-dependent features such as the **SysTick** timer and the **Interrupt System**.

Refer to \ref core_config_sect for a list of the available `core_<cpu>.h` header files.

**Example:**

The following code section shows the usage of the `core_<cpu>.h` header files to build a generic library for Cortex-M0, Cortex-M3, Cortex-M4, or Cortex-M7 devices.

To select the processor, the source code uses the defines `CORTEX_M7`, `CORTEX_M4`, `CORTEX_M3`, `CORTEX_M0`, or `CORTEX_M0PLUS`. One of these defines needs to be provided on the compiler command line. By using this header file, the source code can access the correct implementations for \ref Core_Register_gr, \ref intrinsic_CPU_gr, \ref intrinsic_SIMD_gr, and \ref ITM_Debug_gr.

```c
#define __CMSIS_GENERIC              /* disable NVIC and Systick functions */

#if defined (CORTEX_M7)
  #include "core_cm7.h"
#elif defined (CORTEX_M4)
  #include "core_cm4.h"
#elif defined (CORTEX_M3)
  #include "core_cm3.h"
#elif defined (CORTEX_M0)
  #include "core_cm0.h"
#elif defined (CORTEX_M0PLUS)
  #include "core_cm0plus.h"
#else
  #error "Processor not specified or unsupported."
#endif
```
