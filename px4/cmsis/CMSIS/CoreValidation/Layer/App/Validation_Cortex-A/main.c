/*
 * Copyright (C) 2022 ARM Limited or its affiliates. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>

#include "RTE_Components.h"
#include  CMSIS_device_header

#ifdef RTE_Compiler_EventRecorder
#include "EventRecorder.h"
#endif

#include "cmsis_cv.h"
#include "CV_Report.h"

//lint -e970 allow using int for main

int main (void)
{

  // System Initialization
  SystemCoreClockUpdate();

#ifdef RTE_Compiler_EventRecorder
  // Initialize and start Event Recorder
  (void)EventRecorderInitialize(EventRecordError, 1U);
  (void)EventRecorderEnable(EventRecordAll, 0xFEU, 0xFEU);
#endif

  cmsis_cv();

  #ifdef __MICROLIB
  for(;;) {}
  #else
  exit(0);
  #endif
}

#if defined(__CORTEX_A)
#include "irq_ctrl.h"

#if (defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)) || \
    (defined ( __GNUC__ ))
  #define __IRQ __attribute__((interrupt("IRQ")))
#elif defined ( __CC_ARM )
  #define __IRQ __irq
#elif defined ( __ICCARM__ )
  #define __IRQ __irq __arm
#else
  #error "Unsupported compiler!"
#endif


__IRQ
void IRQ_Handler(void);
__IRQ
void IRQ_Handler(void) {
  const IRQn_ID_t irqn = IRQ_GetActiveIRQ();
  IRQHandler_t const handler = IRQ_GetHandler(irqn);
  if (handler != NULL) {
    __enable_irq();
    handler();
    __disable_irq();
  }
  IRQ_EndOfInterrupt(irqn);
}

__IRQ __NO_RETURN
void Undef_Handler (void);
__IRQ __NO_RETURN
void Undef_Handler (void) {
  cmsis_cv_abort(__FILENAME__, __LINE__, "Undefined Instruction!");
  exit(0);
}

__IRQ
void SVC_Handler   (void);
__IRQ
void SVC_Handler   (void) {
}

__IRQ __NO_RETURN
void PAbt_Handler  (void);
__IRQ __NO_RETURN
void PAbt_Handler  (void) {
  cmsis_cv_abort(__FILENAME__, __LINE__, "Prefetch Abort!");
  exit(0);
}

__IRQ __NO_RETURN
void DAbt_Handler  (void);
__IRQ __NO_RETURN
void DAbt_Handler  (void) {
  cmsis_cv_abort(__FILENAME__, __LINE__, "Data Abort!");
  exit(0);
}

__IRQ
void FIQ_Handler   (void);
__IRQ
void FIQ_Handler   (void) {
}
#endif

#if defined(__CORTEX_M)
__NO_RETURN
void HardFault_Handler(void);
__NO_RETURN
void HardFault_Handler(void) {
  cmsis_cv_abort(__FILENAME__, __LINE__, "HardFault!");
  #ifdef __MICROLIB
  for(;;) {}
  #else
  exit(0);
  #endif
}
#endif
