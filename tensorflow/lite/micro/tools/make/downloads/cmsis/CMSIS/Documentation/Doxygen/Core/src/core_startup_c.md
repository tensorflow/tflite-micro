# Startup File startup_\<Device\>.c {#startup_c_pg}

The startup file defines device exceptions and interrupts, and provides their initial (weak) handler functions. The file has a naming convention `startup_<Device>.c` where `<Device>` corresponds to the device name.

Specifically, following functionalities are provided in the startup file:

 - The reset handler `Reset_Handler` which is executed upon CPU reset and typically calls the `SystemInit()` function. After the system init the control is transferred to the C/C++ run-time library which performs initialization and calls the `main` function in the user code.
 - The setup values for the Main Stack Pointer (MSP).
 - Exception vectors of the Cortex-M Processor with weak functions that implement default routines.
 - Interrupt vectors that are device specific with weak functions that implement default routines.

To adapt the file to a specific device only the interrupt vector table needs to be extended with the device-specific interrupt handlers. The naming convention for the interrupt handler names is `<interrupt_name>_IRQHandler`. This table needs to be consistent with \ref IRQn_Type that defines all the IRQ numbers for each interrupt.

Additional application-specific adaptations may be required in the startup code and therefore so the startup file shall be located in the application project. \ref cmsis_files_dfps explains how this can be achieved when device support is provided in [CMSIS pack format](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html).

**Example:**

The following example shows the extension of the interrupt vector table for the LPC1100 device family.

```c
/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
/* Exceptions */
void WAKEUP0_IRQHandler     (void) __attribute__ ((weak, alias("Default_Handler")));
void WAKEUP1_IRQHandler     (void) __attribute__ ((weak, alias("Default_Handler")));
void WAKEUP2_IRQHandler     (void) __attribute__ ((weak, alias("Default_Handler")));
// :
// :
void EINT1_IRQHandler       (void) __attribute__ ((weak, alias("Default_Handler")));
void EINT2_IRQHandler       (void) __attribute__ ((weak, alias("Default_Handler")));
// :
// :

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector table
 *----------------------------------------------------------------------------*/
extern const pFunc __VECTOR_TABLE[240];
       const pFunc __VECTOR_TABLE[240] __VECTOR_TABLE_ATTRIBUTE = {
  (pFunc)(&__INITIAL_SP),                   /*     Initial Stack Pointer */
  Reset_Handler,                            /*     Reset Handler */
  NMI_Handler,                              /* -14 NMI Handler */
  HardFault_Handler,                        /* -13 Hard Fault Handler */
  MemManage_Handler,                        /* -12 MPU Fault Handler */
  BusFault_Handler,                         /* -11 Bus Fault Handler */
  UsageFault_Handler,                       /* -10 Usage Fault Handler */
  0,                                        /*     Reserved */
  0,                                        /*     Reserved */
  0,                                        /*     Reserved */
  0,                                        /*     Reserved */
  SVC_Handler,                              /*  -5 SVC Handler */
  DebugMon_Handler,                         /*  -4 Debug Monitor Handler */
  0,                                        /*     Reserved */
  PendSV_Handler,                           /*  -2 PendSV Handler */
  SysTick_Handler,                          /*  -1 SysTick Handler */

  /* Interrupts */
  WAKEUP0_IRQHandler,                       /*   0 Wakeup PIO0.0 */
  WAKEUP1_IRQHandler,                       /*   1 Wakeup PIO0.1 */
  WAKEUP2_IRQHandler,                       /*   2 Wakeup PIO0.2 */
  // :
  // :
  EINT1_IRQHandler,                         /*  30 PIO INT1 */
  EINT2_IRQHandler,                         /*  31 PIO INT2 */
  // :
  // :
};
```

## startup_Device.c Template File {#startup_c_sec}

CMSIS-Core \ref cmsis_template_files include a `startup_Device.c` file that can be used as a starting point for chip vendors to implement own device-specific startup file.

The C startup file relys on certain compiler specific preprocessor defines specified in CMSIS compiler headers:

 - \ref __INITIAL_SP
 - \ref __STACK_LIMIT
 - \ref __PROGRAM_START
 - \ref __VECTOR_TABLE
 - \ref __VECTOR_TABLE_ATTRIBUTE
 - \ref __STACK_SEAL (for Armv8-M/v8.1-M)
 - \ref __TZ_set_STACKSEAL_S (for Armv8-M/v8.1-M)

The stack sealing and the initialization for the Stack Limit register is done in function ` Reset_Handler(void)`:

```c
/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
__NO_RETURN void Reset_Handler(void)
{
  __set_PSP((uint32_t)(&__INITIAL_SP));

  __set_MSPLIM((uint32_t)(&__STACK_LIMIT));
  __set_PSPLIM((uint32_t)(&__STACK_LIMIT));

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  __TZ_set_STACKSEAL_S((uint32_t *)(&__STACK_SEAL));
#endif

  SystemInit();                             /* CMSIS System Initialization */
  __PROGRAM_START();                        /* Enter PreMain (C library entry point) */
}
```

> **Note**
> - Stack Sealing also requires the application project to use a scatter file (or a linker script) as explained in \ref RTOS_TrustZone_stacksealing section.
