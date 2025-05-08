/*-----------------------------------------------------------------------------
 *      Name:         CV_Config.h
 *      Purpose:      CV Config header
 *----------------------------------------------------------------------------
 *      Copyright (c) 2017 - 2021 ARM Limited. All rights reserved.
 *----------------------------------------------------------------------------*/
#ifndef __CV_CONFIG_H
#define __CV_CONFIG_H

#include "RTE_Components.h"
#include CMSIS_device_header

#define RTE_CV_COREINSTR  1
#define RTE_CV_COREFUNC   1
#define RTE_CV_L1CACHE    1

//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------

// <h> Common Test Settings
// <o> Print Output Format <0=> Plain Text <1=> XML
// <i> Set the test results output format to plain text or XML
#ifndef PRINT_XML_REPORT
#define PRINT_XML_REPORT            1
#endif
// <o> Buffer size for assertions results
// <i> Set the buffer size for assertions results buffer
#define BUFFER_ASSERTIONS           128U
// </h>

// <h> Disable Test Cases
// <i> Uncheck to disable an individual test case
// <q0> TC_CoreInstr_NOP
#define TC_COREINSTR_NOP_EN                   1
// <q0> TC_CoreInstr_REV
#define TC_COREINSTR_REV_EN                   1
// <q0> TC_CoreInstr_REV16
#define TC_COREINSTR_REV16_EN                 1
// <q0> TC_CoreInstr_REVSH
#define TC_COREINSTR_REVSH_EN                 1
// <q0> TC_CoreInstr_ROR
#define TC_COREINSTR_ROR_EN                   1
// <q0> TC_CoreInstr_RBIT
#define TC_COREINSTR_RBIT_EN                  1
// <q0> TC_CoreInstr_CLZ
#define TC_COREINSTR_CLZ_EN                   1
// <q0> TC_CoreInstr_Exclusives
#define TC_COREINSTR_EXCLUSIVES_EN            1
// <q0> TC_CoreInstr_SSAT
#define TC_COREINSTR_SSAT_EN                  1
// <q0> TC_CoreInstr_USAT
#define TC_COREINSTR_USAT_EN                  1

// <q0> TC_CoreAFunc_IRQ
#define TC_COREAFUNC_IRQ                      1
// <q0> TC_CoreAFunc_FaultIRQ
#define TC_COREAFUNC_FAULTIRQ                 1
// <q0> TC_CoreAFunc_FPSCR
#define TC_COREAFUNC_FPSCR                    1
// <q0> TC_CoreAFunc_CPSR
#define TC_COREAFUNC_CPSR                     1
// <q0> TC_CoreAFunc_Mode
#define TC_COREAFUNC_MODE                     1
// <q0> TC_CoreAFunc_FPEXC
#define TC_COREAFUNC_FPEXC                    1
// <q0> TC_CoreAFunc_ACTLR
#define TC_COREAFUNC_ACTLR                    1
// <q0> TC_CoreAFunc_CPACR
#define TC_COREAFUNC_CPACR                    1
// <q0> TC_CoreAFunc_DFSR
#define TC_COREAFUNC_DFSR                     1
// <q0> TC_CoreAFunc_IFSR
#define TC_COREAFUNC_IFSR                     1
// <q0> TC_CoreAFunc_ISR
#define TC_COREAFUNC_ISR                      1
// <q0> TC_CoreAFunc_CBAR
#define TC_COREAFUNC_CBAR                     1
// <q0> TC_CoreAFunc_TTBR0
#define TC_COREAFUNC_TTBR0                    1
// <q0> TC_CoreAFunc_DACR
#define TC_COREAFUNC_DACR                     1
// <q0> TC_CoreAFunc_SCTLR
#define TC_COREAFUNC_SCTLR                    1
// <q0> TC_CoreAFunc_MPIDR
#define TC_COREAFUNC_MPIDR                    1
// <q0> TC_CoreAFunc_VBAR
#define TC_COREAFUNC_VBAR                     1
// <q0> TC_CoreAFunc_MVBAR
#define TC_COREAFUNC_MVBAR                    1
// <q0> TC_CoreAFunc_FPU_Enable
#define TC_COREAFUNC_FPU_ENABLE               1

// <q0> TC_GenTimer_CNTFRQ
#define TC_GENTIMER_CNTFRQ                    1
// <q0> TC_GenTimer_CNTP_TVAL
#define TC_GENTIMER_CNTP_TVAL                 1
// <q0> TC_GenTimer_CNTP_CTL
#define TC_GENTIMER_CNTP_CTL                  1
// <q0> TC_GenTimer_CNTPCT
#define TC_GENTIMER_CNTPCT                    1
// <q0> TC_GenTimer_CNTP_CVAL
#define TC_GENTIMER_CNTP_CVAL                 1

// <q0> TC_CAL1Cache_EnDisable
#define TC_CAL1CACHE_ENDISABLE                1
// <q0> TC_CAL1Cache_EnDisableBTAC
#define TC_CAL1CACHE_ENDISABLEBTAC            1
// <q0> TC_CAL1Cache_log2_up
#define TC_CAL1CACHE_LOG2_UP                  1
// <q0> TC_CAL1Cache_InvalidateDCacheAll
#define TC_CAL1CACHE_INVALIDATEDCACHEALL      1
// <q0> TC_CAL1Cache_CleanDCacheAll
#define TC_CAL1CACHE_CLEANDCACHEALL           1
// <q0> TC_CAL1Cache_CleanInvalidateDCacheAll
#define TC_CAL1CACHE_CLEANINVALIDATEDCACHEALL 1
// </h>

#endif /* __CV_CONFIG_H */

