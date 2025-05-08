/******************************************************************************
 * @file     startup_ARMCR5.c
 * @brief    CMSIS Device System Source File for Arm Cortex-A9 Device Series
 * @version  V1.0.0
 * @date     31. March 2024
 ******************************************************************************/
/*
 * Copyright (c) 2024 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ARMCR5.h>

/*----------------------------------------------------------------------------
  Internal References
 *----------------------------------------------------------------------------*/
void Vectors        (void) __attribute__ ((naked, section("RESET")));
void Reset_Handler  (void) __attribute__ ((naked));
void Default_Handler(void) __attribute__ ((noreturn));

/*----------------------------------------------------------------------------
  Exception / Interrupt Handler
 *----------------------------------------------------------------------------*/
void Undef_Handler (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));
void SVC_Handler   (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));
void PAbt_Handler  (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));
void DAbt_Handler  (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));
void IRQ_Handler   (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));
void FIQ_Handler   (void) __attribute__ ((weak, noreturn, alias("Default_Handler")));

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector Table
 *----------------------------------------------------------------------------*/
void Vectors (void)
{
  __ASM volatile(
  "LDR    PC, =Reset_Handler                        \n"
  "LDR    PC, =Undef_Handler                        \n"
  "LDR    PC, =SVC_Handler                          \n"
  "LDR    PC, =PAbt_Handler                         \n"
  "LDR    PC, =DAbt_Handler                         \n"
  "NOP                                              \n"
  "LDR    PC, =IRQ_Handler                          \n"
  "LDR    PC, =FIQ_Handler                          \n"
  );
}

/*----------------------------------------------------------------------------
  Reset Handler called on controller reset
 *----------------------------------------------------------------------------*/
void Reset_Handler (void)
{
  __ASM volatile(

  // Mask interrupts
  "CPSID   if                                      \n"

  // Put any cores other than 0 to sleep
  "MRC     p15, 0, R0, c0, c0, 5                   \n"  // Read MPIDR
  "ANDS    R0, R0, #3                              \n"
  "goToSleep:                                      \n"
  "WFINE                                           \n"
  "BNE     goToSleep                               \n"

  // In the Cortex-R5, the Z-bit of the SCTLR does not control the program flow prediction.
  // Some control bits in the ACTLR control the program flow and prefetch features instead.
  // These are enabled by default, but are shown here for completeness.
  "MRC     p15, 0, R0, c1, c0, 0                   \n"  // Read CP15 System Control register
  "BIC     R0, R0, #(0x1 << 12)                    \n"  // Clear I bit 12 to disable I Cache
  "BIC     R0, R0, #(0x1 <<  2)                    \n"  // Clear C bit  2 to disable D Cache
  "BIC     R0, R0, #0x1                            \n"  // Clear M bit  0 to disable MPU
  "DSB                                             \n"  // Ensure all previous loads/stores have completed
  "MCR     p15, 0, R0, c1, c0, 0                   \n"  // Write value back to CP15 System Control register
  "ISB                                             \n"  // Ensure subsequent insts execute with new MPU settings

  // Configure ACTLR
  "MRC     p15, 0, r0, c1, c0, 1                   \n"  // Read CP15 Auxiliary Control Register
  "BIC     r0, r0, #(0x1 << 17)                    \n"  // Clear RSDIS bit 17 to enable return stack
  "BIC     r0, r0, #(0x1 << 16)                    \n"  // Clear BP bit 15 and BP bit 16:
  "BIC     r0, r0, #(0x1 << 15)                    \n"  // Normal operation, BP is taken from the global history table.
  "MCR     p15, 0, r0, c1, c0, 1                   \n"  // Write ACTLR
  "ISB                                             \n"  

  // Setup Stack for each exceptional mode
  "CPS    #0x11                                    \n"
  "LDR    SP, =Image$$FIQ_STACK$$ZI$$Limit         \n"
  "CPS    #0x12                                    \n"
  "LDR    SP, =Image$$IRQ_STACK$$ZI$$Limit         \n"
  "CPS    #0x13                                    \n"
  "LDR    SP, =Image$$SVC_STACK$$ZI$$Limit         \n"
  "CPS    #0x17                                    \n"
  "LDR    SP, =Image$$ABT_STACK$$ZI$$Limit         \n"
  "CPS    #0x1B                                    \n"
  "LDR    SP, =Image$$UND_STACK$$ZI$$Limit         \n"
  "CPS    #0x1F                                    \n"
  "LDR    SP, =Image$$ARM_LIB_STACK$$ZI$$Limit     \n"

  // Call SystemInit
  "BL     SystemInit                               \n"

  // Unmask interrupts
  "CPSIE  if                                       \n"

  // Call __main
  "BL     __main                                   \n"
  );
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void Default_Handler (void)
{
  while(1);
}
