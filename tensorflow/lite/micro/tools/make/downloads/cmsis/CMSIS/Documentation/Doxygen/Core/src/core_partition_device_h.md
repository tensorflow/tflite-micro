# TrustZone setup: partition_<Device>.h {#partition_h_pg}

The TrustZone header file contains the initial setup of the TrustZone hardware in an Armv8-M system.

This file implements the function \ref TZ_SAU_Setup that is called from \ref SystemInit. It uses settings in these files:

 - \ref partition_h_pg "partition_<Device>.h" that defines the initial system configuration and during SystemInit in Secure state.
 - \ref partition_gen_h_pg "partition_gen.h" that contains SAU region and interrupt target assignments. This file may be generated using [CMSIS-Zone](../Zone/index.html).

> **Note**
> - \ref partition_gen_h_pg "partition_gen.h" is optional and can be generated using [CMSIS-Zone](../Zone/index.html). In previous versions of CMSIS-Core(M) this settings were part of \ref partition_h_pg "partition_<Device>.h".

The \ref partition_h_pg "partition_<Device>.h" file contains the following configuration settings for:

 - \ref sau_ctrlregister_sec provides settings for the SAU CTRL register.
 - \ref sau_sleepexception_sec provides device-specific deep-sleep and exception settings.
 - \ref sau_fpu_sec defines the usage of the Floating Point Unit in secure and non-secure state.

The \ref partition_h_pg "partition_<Device>.h" file includes the \ref partition_gen_h_pg "partition_gen.h" file with configuration settings for:

 - \ref sau_regions_sect provides configuration of the SAU Address Regions.
 - \ref sau_interrupttarget_sec provides device-specific interrupt target settings.

##  SAU CTRL register settings {#sau_ctrlregister_sec}

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>SAU_INIT_CTRL</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Initialize SAU CTRL register or not
           - 0: do not initialize SAU CTRL register
           - 1: initialize SAU CTRL register</td>
    </tr>
    <tr>
      <td>SAU_INIT_CTRL_ENABLE</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>enable/disable the SAU
           - 0: disable SAU
           - 1: enable SAU</td>
    </tr>
    <tr>
      <td>SAU_INIT_CTRL_ALLNS</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>value for SAU_CTRL register bit ALLNS
           - 0: all Memory is Secure
           - 1: all Memory is Non-Secure</td>
    </tr>
</table>

## Configuration of Sleep and Exception behaviour {#sau_sleepexception_sec}

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>SCB_CSR_AIRCR_INIT</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Setup behaviour of Sleep and Exception Handling
           - 0: not setup of CSR and AIRCR registers; the values below are not relevant
           - 1: setup of CSR and AIRCR registers with values below</td>
    </tr>
    <tr>
      <td>CSR_INIT_DEEPSLEEPS_VAL</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>value for SCB_CSR register bit DEEPSLEEPS
           - 0: Deep Sleep can be enabled by Secure and Non-Secure state
           - 1: Deep Sleep can be enabled by Secure state only</td>
    </tr>
    <tr>
      <td>AIRCR_INIT_SYSRESETREQS_VAL</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>value for SCB_AIRCR register bit SYSRESETREQS
           - 0: System reset request accessible from Secure and Non-Secure state
           - 1: System reset request accessible from Secure state only</td>
    </tr>
    <tr>
      <td>AIRCR_INIT_PRIS_VAL</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>value for SCB_AIRCR register bit PRIS
           - 0: Priority of Non-Secure exceptions is Not altered
           - 1: Priority of Non-Secure exceptions is Lowered to 0x80-0xFF</td>
    </tr>
    <tr>
      <td>AIRCR_INIT_BFHFNMINS_VAL</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>value for SCB_AIRCR register bit BFHFNMINS
           - 0: BusFault, HardFault, and NMI target are Secure state
           - 1: BusFault, HardFault, and NMI target are Non-Secure state</td>
    </tr>
</table>


## Configuration of Floating Point Unit {#sau_fpu_sec}

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>TZ_FPU_NS_USAGE</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Setup behaviour of Floating Point Unit
           - 0: not setup of NSACR and FPCCR registers; the values below are not relevant
           - 1: setup of NSACR and FPCCR registers with values below</td>
    </tr>
    <tr>
      <td>SCB_NSACR_CP10_11_VAL</td>
      <td>0 or 3</td>
      <td>3</td>
      <td>Floating Point Unit usage (Value for SCB->NSACR register bits CP10, CP11)
           - 0: Secure state only
           - 3: Secure and Non-Secure state</td>
    </tr>
    <tr>
      <td>FPU_FPCCR_TS_VAL</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Treat floating-point registers as Secure (value for FPU->FPCCR register bit TS)
           - 0: Disable
           - 1: Enabled</td>
    </tr>
    <tr>
      <td>FPU_FPCCR_CLRONRETS_VAL</td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>Clear on return (CLRONRET) accessibility (Value for FPU->FPCCR register bit CLRONRETS)
           - 0: Secure and Non-Secure state
           - 1: Secure state only</td>
    </tr>
    <tr>
      <td>FPU_FPCCR_CLRONRET_VAL</td>
      <td>0 .. 1</td>
      <td>1</td>
      <td>Clear floating-point caller saved registers on exception return (Value for FPU->FPCCR register bit CLRONRET)
           - 0: Disabled
           - 1: Enabled</td>
    </tr>
</table>

## Region/ISR setup: partition_gen.h {#partition_gen_h_pg}

The \ref partition_gen_h_pg "partition_gen.h" header file can be generated using [CMSIS-Zone](../Zone/index.html).

The \ref partition_h_pg "partition_<Device>.h" file includes the \ref partition_h_pg "partition_gen.h" file with configuration settings for:

  - \ref sau_regions_sect provides configuration of the SAU Address Regions.
  - \ref sau_interrupttarget_sec provides device-specific interrupt target settings.

> **Note**
> - In previous versions of CMSIS-Core(M) the above settings were part of \ref partition_h_pg "partition_<Device>.h"

### Configuration of the SAU Address Regions {#sau_regions_sect}

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>SAU_REGIONS_MAX</td>
      <td>0 .. tbd</td>
      <td>8</td>
      <td>maximum number of SAU regions</td>
    </tr>
    <tr>
      <td>SAU_INIT_REGION<number></td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>initialize SAU region or not
           - 0: do not initialize SAU region
           - 1: initialize SAU region</td>
    </tr>
    <tr>
      <td>SAU_INIT_START<number></td>
      <td>0x00000000 .. 0xFFFFFFE0\n
          [in steps of 32]</td>
      <td>0x00000000</td>
      <td>region start address</td>
    </tr>
    <tr>
      <td>SAU_INIT_END<number></td>
      <td>0x00000000 .. 0xFFFFFFE0\n
          [in steps of 32]</td>
      <td>0x00000000</td>
      <td>region start address</td>
    </tr>
    <tr>
      <td>SAU_INIT_NSC<number></td>
      <td>0 .. 1</td>
      <td>0</td>
      <td>SAU region attribute
           - 0: Non-Secure
           - 1: Secure, Non-Secure callable</td>
    </tr>
</table>

The range of \<number\> is from 0 .. SAU_REGIONS_MAX.
A set of these macros must exist for each \<number\>.

The following example shows a set of SAU region macros.

```c
#define SAU_REGIONS_MAX   8                 /* Max. number of SAU regions */

#define SAU_INIT_REGION0    1
#define SAU_INIT_START0     0x00000000      /* start address of SAU region 0 */
#define SAU_INIT_END0       0x001FFFE0      /* end address of SAU region 0 */
#define SAU_INIT_NSC0       1

#define SAU_INIT_REGION1    1
#define SAU_INIT_START1     0x00200000      /* start address of SAU region 1 */
#define SAU_INIT_END1       0x003FFFE0      /* end address of SAU region 1 */
#define SAU_INIT_NSC1       0

#define SAU_INIT_REGION2    1
#define SAU_INIT_START2     0x20200000      /* start address of SAU region 2 */
#define SAU_INIT_END2       0x203FFFE0      /* end address of SAU region 2 */
#define SAU_INIT_NSC2       0

#define SAU_INIT_REGION3    1
#define SAU_INIT_START3     0x40000000      /* start address of SAU region 3 */
#define SAU_INIT_END3       0x40040000      /* end address of SAU region 3 */
#define SAU_INIT_NSC3       0

#define SAU_INIT_REGION4    0
#define SAU_INIT_START4     0x00000000      /* start address of SAU region 4 */
#define SAU_INIT_END4       0x00000000      /* end address of SAU region 4 */
#define SAU_INIT_NSC4       0

#define SAU_INIT_REGION5    0
#define SAU_INIT_START5     0x00000000      /* start address of SAU region 5 */
#define SAU_INIT_END5       0x00000000      /* end address of SAU region 5 */
#define SAU_INIT_NSC5       0

#define SAU_INIT_REGION6    0
#define SAU_INIT_START6     0x00000000      /* start address of SAU region 6 */
#define SAU_INIT_END6       0x00000000      /* end address of SAU region 6 */
#define SAU_INIT_NSC6       0

#define SAU_INIT_REGION7    0
#define SAU_INIT_START7     0x00000000      /* start address of SAU region 7 */
#define SAU_INIT_END7       0x00000000      /* end address of SAU region 7 */
#define SAU_INIT_NSC7       0
```


### Configuration of Interrupt Target settings {#sau_interrupttarget_sec}

Each interrupt has a configuration bit that defines the execution in Secure or Non-secure state. The Non-Secure interrupts have a separate vector table.  Refer to \ref Model_TrustZone for more information.

<table class="cmtable">
    <tr>
      <th>\#define</th>
      <th>Value Range</th>
      <th>Default</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>NVIC_INIT_ITNS<number></td>
      <td>0x00000000 .. 0xFFFFFFFF\n
          [each bit represents an interrupt]</td>
      <td>0x00000000</td>
      <td>Interrupt vector target
           - 0: Secure state
           - 1: Non-Secure state</td>
    </tr>
</table>

The range of \<number\> is 0 .. (\<number of external interrupts\> + 31) / 32.

The following example shows the configuration for a maximum of 64 external interrupts.

```c
#define NVIC_INIT_ITNS0      0x0000122B
#define NVIC_INIT_ITNS1      0x0000003A
```
