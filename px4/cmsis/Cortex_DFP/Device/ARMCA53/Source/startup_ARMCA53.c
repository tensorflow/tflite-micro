/******************************************************************************
 * @file     startup_ARMCA53.c
 * @brief    CMSIS Device System Source File for Arm Cortex-A9 Device Series
 * @version  V1.0.0
 * @date     31. March 2024
 ******************************************************************************/
/*
 * Copyright (c) 2009-2021 Arm Limited. All rights reserved.
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

#include <ARMCA53.h>

/*----------------------------------------------------------------------------
  Definitions
 *----------------------------------------------------------------------------*/
#define USR_MODE 0x10            // User mode
#define FIQ_MODE 0x11            // Fast Interrupt Request mode
#define IRQ_MODE 0x12            // Interrupt Request mode
#define SVC_MODE 0x13            // Supervisor mode
#define ABT_MODE 0x17            // Abort mode
#define UND_MODE 0x1B            // Undefined Instruction mode
#define SYS_MODE 0x1F            // System mode

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
void Vectors(void) {
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
void Reset_Handler(void) {
  __ASM volatile(

  // Call __main
  "BL     __main                                   \n"
  );
}

/*----------------------------------------------------------------------------
  Default Handler for Exceptions / Interrupts
 *----------------------------------------------------------------------------*/
void Default_Handler(void) {
  while(1);
}
