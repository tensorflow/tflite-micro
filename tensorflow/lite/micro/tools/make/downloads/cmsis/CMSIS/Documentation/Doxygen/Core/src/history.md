# Revision History {#core_revisionHistory}

CMSIS-Core (M) component is maintained with its own versioning that gets officially updated upon releases of the [CMSIS Software Pack](../General/cmsis_pack.html).

The table below provides information about the changes delivered with specific versions of CMSIS-Core (M).

<table class="cmtable" summary="Revision History">
    <tr>
      <th>Version</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>V6.1.0</td>
      <td>
        <ul>
          <li>Added support for Cortex-M52</li>
          <li>Added deprecated CoreDebug symbols for CMSIS 5 compatibility</li>
          <li>Added define CMSIS_DISABLE_DEPRECATED to hide deprecated symbols</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V6.0.0</td>
      <td>
        <ul>
          <li>Core(M) and Core(A) joined into single Core component</li>
          <li>Core header files reworked, aligned with Cortex-M Technical Reference Manuals (TRMs).
              <br/>See \ref core6_changes "Breaking changes in CMSIS-Core v6 header files" for details, and [GitHub issue #122](https://github.com/ARM-software/CMSIS_6/issues/122).</li>
          <li>Previously deprecated features removed</li>
          <li>Dropped support for Arm Compiler 5</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.6.0</td>
      <td>
        <ul>
          <li>Added: Arm Cortex-M85 cpu support</li>
          <li>Added: Arm China Star-MC1 cpu support</li>
          <li>Updated: system_ARMCM55.c</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.5.0</td>
      <td>
        <ul>
          <li>Updated GCC LinkerDescription, GCC Assembler startup</li>
          <li>Added ARMv8-M Stack Sealing (to linker, startup) for toolchain ARM, GCC</li>
          <li>Changed C-Startup to default Startup.</li>
          </li>
            Updated Armv8-M Assembler startup to use GAS syntax<br>
            Note: Updating existing projects may need manual user interaction!
          </li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.4.0</td>
      <td>
        <ul>
          <li>Added: Cortex-M55 cpu support</li>
          <li>Enhanced: MVE support for Armv8.1-MML</li>
          <li>Fixed: Device config define checks</li>
          <li>Added: \ref cache_functions_m7 for Armv7-M and later</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.3.0</td>
      <td>
        <ul>
          <li>Added: Provisions for compiler-independent C startup code.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.2.1</td>
      <td>
        <ul>
          <li>Fixed: Compilation issue in cmsis_armclang_ltm.h introduced in 5.2.0</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.2.0</td>
      <td>
        <ul>
          <li>Added: Cortex-M35P support.</li>
          <li>Added: Cortex-M1 support.
          <li>Added: Armv8.1 architecture support.
          <li>Added: \ref __RESTRICT and \ref __STATIC_FORCEINLINE compiler control macros.
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.1.2</td>
      <td>
        <ul>
          <li>Removed using get/set built-ins FPSCR in GCC >= 7.2 due to shortcomings.</li>
          <li>Added __NO_RETURN to  __NVIC_SystemReset() to silence compiler warnings.</li>
          <li>Added support for Cortex-M1 (beta).</li>
          <li>Removed usage of register keyword.</li>
          <li>Added defines for EXC_RETURN, FNC_RETURN and integrity signature values.</li>
          <li>Enhanced MPUv7 API with defines for memory access attributes.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.1.1</td>
      <td>
        <ul>
          <li>Aligned MSPLIM and PSPLIM access functions along supported compilers.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.1.0</td>
      <td>
        <ul>
          <li>Added MPU Functions for ARMv8-M for Cortex-M23/M33.</li>
          <li>Moved __SSAT and __USAT intrinsics to CMSIS-Core.</li>
          <li>Aligned __REV, __REV16 and __REVSH intrinsics along supported compilers.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.0.2</td>
      <td>
        <ul>
          <li>Added macros  \ref \__UNALIGNED_UINT16_READ,  \ref \__UNALIGNED_UINT16_WRITE.</li>
          <li>Added macros  \ref \__UNALIGNED_UINT32_READ,  \ref \__UNALIGNED_UINT32_WRITE.</li>
          <li>Deprecated macro __UNALIGNED_UINT32.</li>
          <li>Changed \ref version_control_gr macros to be core agnostic.</li>
          <li>Added \ref mpu_functions for Cortex-M0+/M3/M4/M7.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.0.1</td>
      <td>
        <ul>
          <li>Added: macro \ref \__PACKED_STRUCT.</li>
          <li>Added: uVisor support.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00</td>
      <td>
        <ul>
          <li>Added: Cortex-M23, Cortex-M33 support.</li>
          <li>Added: macro __SAU_PRESENT with __SAU_REGION_PRESENT.</li>
          <li>Replaced: macro __SAU_PRESENT with __SAU_REGION_PRESENT.</li>
          <li>Reworked: SAU register and functions.</li>
          <li>Added: macro \ref \__ALIGNED.</li>
          <li>Updated: function \ref SCB_EnableICache.</li>
          <li>Added: cmsis_compiler.h with compiler specific CMSIS macros, functions, instructions.</li>
          <li>Added: macro \ref \__PACKED.</li>
          <li>Updated: compiler specific include files.</li>
          <li>Updated: core dependant include files.</li>
          <li>Removed: deprecated files core_cmfunc.h, core_cminstr.h, core_cmsimd.h.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00<br>Beta 6</td>
      <td>
        <ul>
          <li>Added: SCB_CFSR register bit definitions.</li>
          <li>Added: function \ref NVIC_GetEnableIRQ.</li>
          <li>Updated: core instruction macros \ref \__NOP, \ref \__WFI, \ref \__WFE, \ref \__SEV for toolchain GCC.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00<br>Beta 5</td>
      <td>
        <ul>
          <li>Moved: DSP libraries from CMSIS/DSP/Lib to CMSIS/Lib.</li>
          <li>Added: DSP libraries build projects to CMSIS pack.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00<br>Beta 4</td>
      <td>
        <ul>
          <li>Updated: ARMv8M device files.</li>
          <li>Corrected: ARMv8MBL interrupts.</li>
          <li>Reworked: NVIC functions.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00<br>Beta 2</td>
      <td>
        <ul>
          <li>Changed: ARMv8M SAU regions to 8.</li>
          <li>Changed: moved function \ref TZ_SAU_Setup to file partition_&lt;device&gt;.h.</li>
          <li>Changed: license under Apache-2.0.</li>
          <li>Added: check if macro is defined before use.</li>
          <li>Corrected: function \ref SCB_DisableDCache.</li>
          <li>Corrected: macros \ref \_VAL2FLD, \ref \_FLD2VAL.</li>
          <li>Added: NVIC function virtualization with macros \ref CMSIS_NVIC_VIRTUAL and \ref CMSIS_VECTAB_VIRTUAL.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V5.00<br>Beta 1</td>
      <td>
        <ul>
          <li>Renamed: cmsis_armcc_V6.h to cmsis_armclang.h.</li>
          <li>Renamed: core\_*.h to lower case.</li>
          <li>Added: function \ref SCB_GetFPUType to all CMSIS cores.</li>
          <li>Added: ARMv8-M support.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V4.30</td>
      <td>
        <ul>
          <li>Corrected: DoxyGen function parameter comments.</li>
          <li>Corrected: IAR toolchain: removed for \ref NVIC_SystemReset the attribute(noreturn).</li>
          <li>Corrected: GCC toolchain: suppressed irrelevant compiler warnings.</li>
          <li>Added: Support files for Arm Compiler v6 (cmsis_armcc_v6.h).</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V4.20</td>
      <td>
        <ul>
          <li>Corrected: MISRA-C:2004 violations.</li>
          <li>Corrected: predefined macro for TI CCS Compiler.</li>
          <li>Corrected: function \ref __SHADD16 in arm_math.h.</li>
          <li>Updated: cache functions for Cortex-M7.</li>
          <li>Added: macros \ref _VAL2FLD, \ref _FLD2VAL to core\_*.h.</li>
          <li>Updated: functions \ref __QASX, \ref __QSAX, \ref __SHASX, \ref __SHSAX.</li>
          <li>Corrected: potential bug in function \ref __SHADD16.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V4.10</td>
      <td>
        <ul>
          <li>Corrected: MISRA-C:2004 violations.</li>
          <li>Corrected: intrinsic functions \ref __DSB, \ref __DMB, \ref __ISB.</li>
          <li>Corrected: register definitions for ITCMCR register.</li>
          <li>Corrected: register definitions for \ref CONTROL_Type register.</li>
          <li>Added: functions \ref SCB_GetFPUType, \ref SCB_InvalidateDCache_by_Addr to core_cm7.h.</li>
          <li>Added: register definitions for \ref APSR_Type, \ref IPSR_Type, \ref xPSR_Type register.</li>
          <li>Added: \ref __set_BASEPRI_MAX function to core_cmFunc.h.</li>
          <li>Added: intrinsic functions \ref __RBIT, \ref __CLZ  for Cortex-M0/CortexM0+.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V4.00</td>
      <td>
        <ul>
          <li>Added: Cortex-M7 support.</li>
          <li>Added: intrinsic functions for \ref __RRX, \ref __LDRBT, \ref __LDRHT, \ref __LDRT, \ref __STRBT, \ref __STRHT, and \ref __STRT</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V3.40</td>
      <td>
       <ul>
         <li>Corrected: C++ include guard settings.</li>
       </ul>
     </td>
    </tr>
    <tr>
      <td>V3.30</td>
      <td>
        <ul>
          <li>Added: COSMIC tool chain support.</li>
          <li>Corrected: GCC __SMLALDX instruction intrinsic for Cortex-M4.</li>
          <li>Corrected: GCC __SMLALD instruction intrinsic for Cortex-M4.</li>
          <li>Corrected: GCC/CLang warnings.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V3.20</td>
      <td>
        <ul>
          <li>Added: \ref __BKPT instruction intrinsic.</li>
          <li>Added: \ref __SMMLA instruction intrinsic for Cortex-M4.</li>
          <li>Corrected: \ref ITM_SendChar.</li>
          <li>Corrected: \ref __enable_irq, \ref __disable_irq and inline assembly for GCC Compiler.</li>
          <li>Corrected: \ref NVIC_GetPriority and VTOR_TBLOFF for Cortex-M0/M0+, SC000.</li>
          <li>Corrected: rework of in-line assembly functions to remove potential compiler warnings.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V3.01</td>
      <td>
       <ul>
         <li>Added support for Cortex-M0+ processor.</li>
       </ul>
     </td>
    </tr>
    <tr>
      <td>V3.00</td>
      <td>
        <ul>
          <li>Added support for GNU GCC ARM Embedded Compiler.</li>
          <li>Added function \ref __ROR.</li>
          <li>Added \ref regMap_pg for TPIU, DWT.</li>
          <li>Added support for \ref core_config_sect "SC000 and SC300 processors".</li>
          <li>Corrected \ref ITM_SendChar function.</li>
          <li>Corrected the functions \ref __STREXB, \ref __STREXH, \ref __STREXW for the GNU GCC compiler section.</li>
          <li>Documentation restructured.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V2.10</td>
      <td>
        <ul>
          <li>Updated documentation.</li>
          <li>Updated CMSIS core include files.</li>
          <li>Changed CMSIS/Device folder structure.</li>
          <li>Added support for Cortex-M0, Cortex-M4 w/o FPU to CMSIS DSP library.</li>
          <li>Reworked CMSIS DSP library examples.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V2.00</td>
      <td>
       <ul>
         <li>Added support for Cortex-M4 processor.</li>
       </ul>
     </td>
    </tr>
    <tr>
      <td>V1.30</td>
      <td>
        <ul>
          <li>Reworked Startup Concept.</li>
          <li>Added additional Debug Functionality.</li>
          <li>Changed folder structure.</li>
          <li>Added doxygen comments.</li>
          <li>Added definitions for bit.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.01</td>
      <td>
       <ul>
         <li>Added support for Cortex-M0 processor.</li>
       </ul>
      </td>
    </tr>
    <tr>
      <td>V1.01</td>
      <td>
       <ul>
         <li>Added intrinsic functions for \ref __LDREXB, \ref __LDREXH, \ref __LDREXW, \ref __STREXB, \ref __STREXH, \ref __STREXW, and \ref __CLREX</li>
       </ul>
     </td>
    </tr>
    <tr>
      <td>V1.00</td>
      <td>
       <ul>
         <li>Initial Release for Cortex-M3 processor.</li>
       </ul>
     </td>
    </tr>
</table>

\section core6_changes Breaking changes in CMSIS-Core 6

\ref cmsis_standard_files in CMSIS-Core v6.0.0 have received a number of changes that are incompatible with CMSIS-Core v5.6.0.

In summary, following types of incompatible changes are present:

 - struct member is renamed in an existing structure (e.g. NVIC->PR -> NVIC->IPR)
 - struct name is changed (e.g. CoreDebug_Type -> DCB_Type)
 - define name is changed (e.g. CoreDebug_DEMCR_TRCENA_Msk -> DCB_DEMCR_TRCENA_Msk)

For the latest two types, CMSIS-Core v6.1 and higher provide also the original CMSIS 5 symbols as deprecated and so improve the backward compatibility. See section \ref deprecated_gr.

Additionally, the [GitHub issue #122](https://github.com/ARM-software/CMSIS_6/issues/122) discusses how to resolve such incompatibilities.

Below is detailed information about the changes relevant for each Cortex-M core.

**Cortex-M0, Cortex-M0+, Cortex-M1:**

- struct NVIC_Type
  - member IP renamed to IPR
- struct SCB_Type
  - member SHP renamed to SHPR

**Cortex-M3, Cortex-M4:**

- struct NVIC_Type
  - member IP renamed to IPR
- struct SCB_Type
  - member SHP renamed to SHPR
  - member PFR renamed to ID_PFR
  - member PFR renamed to ID_PFR
  - member DFR renamed to ID_PFR
  - member ADR renamed to ID_AFR
  - member MMFR renamed to ID_MMFR
  - member ISAR renamed to ID_ISAR
  - member STIR added
- struct ITM_Type:
  - members PIDx and CIDx removed
- define names for ITM_TCR_* changed
- define names for ITM_LSR_* changed
- struct TPI_Type renamed to TPIU_Type
- define names for TPI_* renamed to TPIU_*
- define names for FPU_MVFR0/1_* changed (Cortex-M4)
- struct CoreDebug_Type renamed to DCB_Type
- defines for CoreDebug_* renamed to DCB_*

**Cortex-M7:**

- struct NVIC_Type
  - member IP renamed to IPR
- struct SCB_Type
  - member ID_MFR renamed to ID_MMFR
- struct ITM_Type:
  - members PIDx and CIDx removed
- define names for ITM_TCR_* changed
- define names for ITM_LSR_* changed
- struct TPI_Type renamed to TPIU_Type
- define names for TPI_* renamed to TPIU_*
- define names for FPU_MVFR0/1_* changed
- struct CoreDebug_Type renamed to DCB_Type
- defines for CoreDebug_* renamed to DCB_*

**Cortex-M23:**

- struct DWT_Type
  - member RESERVED0[6] replaced by CYCCNT, CPICNT, EXCCNT, SLEEPCNT, LSUCNT, FOLDCNT
  - other RESERVED members mainly removed
- struct TPI_Type renamed to TPIU_Type
- define names for TPI_* renamed to TPIU_*
- struct CoreDebug_Type removed (struct DCB_Type already existed)
- defines CoreDebug_* removed (defines DCB_* already existed)

**Cortex-M33:**

- struct ITM_Type:
  - members LAR, LSR removed
  - members PIDx and CIDx removed
- struct TPI_Type renamed to TPIU_Type
- define names for TPI_* renamed to TPIU_*
- define names for FPU_MVFR0/1_* changed
- struct CoreDebug_Type removed (struct DCB_Type already existed)
- defines CoreDebug_* removed (defines DCB_* already existed)

**Cortex-M55, Cortex-M85:**

- struct ITM_Type:
  - members LAR, LSR removed
  - members PIDx and CIDx removed
- struct DWT_Type:
  - members PIDx and CIDx removed
- struct EWIC_Type
  - all members renamed
- define names EWIC_* changed
- struct TPI_Type renamed to TPIU_Type
  - members LAR, LSR replaced
- define names for TPI_* renamed to TPIU_*
- struct PMU_Type
  - members PIDx and CIDx removed
- struct CoreDebug_Type removed (struct DCB_Type already existed)
- defines CoreDebug_* removed (defines DCB_* already existed)
- struct DIB_Type
  - members DLAR, DLSR removed (replaced by RESERVED0[2])
- defines for DIB_DLAR_* and DIB_DLSR_* removed
