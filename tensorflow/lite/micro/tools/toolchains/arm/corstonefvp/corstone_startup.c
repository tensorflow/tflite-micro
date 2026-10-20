/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Standalone, self-contained startup_ARMCM55.c matching Official ARM CMSIS
   Template. No dependencies on external Keil/third_party vendor libraries. */

#include <stdint.h>
#include <stdio.h>

extern uint32_t __INITIAL_SP;
extern uint32_t __STACK_LIMIT;

extern void _start(void);      // GCC Newlib C runtime entry point
extern void SystemInit(void);  // CMSIS System Initialization

void Reset_Handler(void);
void Default_Handler(void);
void Fault_Handler(void);

extern void EthosU_IRQHandler(void);

typedef void (*VECTOR_TABLE_Type)(void);

__attribute__((section(".vectors"), used))
const VECTOR_TABLE_Type __VECTOR_TABLE[80] = {
    (VECTOR_TABLE_Type)(&__INITIAL_SP), /*     Initial Stack Pointer */
    Reset_Handler,   /*     Reset Handler (Linker sets Thumb relocation bit 0
                        automatically) */
    Fault_Handler,   /* -14 NMI Handler */
    Fault_Handler,   /* -13 Hard Fault Handler */
    Fault_Handler,   /* -12 MPU Fault Handler */
    Fault_Handler,   /* -11 Bus Fault Handler */
    Fault_Handler,   /* -10 Usage Fault Handler */
    Fault_Handler,   /*  -9 Secure Fault Handler */
    0,               /*     Reserved */
    0,               /*     Reserved */
    0,               /*     Reserved */
    Default_Handler, /*  -5 SVCall Handler */
    Default_Handler, /*  -4 Debug Monitor Handler */
    0,               /*     Reserved */
    Default_Handler, /*  -2 PendSV Handler */
    Default_Handler, /*  -1 SysTick Handler */
    [16 + 56] = EthosU_IRQHandler, /*  56 Ethos-U55 NPU Interrupt */
};

__attribute__((section(".vectors_sram"), used))
const VECTOR_TABLE_Type __VECTOR_TABLE_SRAM[80] = {
    (VECTOR_TABLE_Type)(&__INITIAL_SP), /*     Initial Stack Pointer */
    Reset_Handler, /*     Reset Handler (Linker sets Thumb relocation bit 0
                      automatically) */
    Fault_Handler, /* -14 NMI Handler */
    Fault_Handler, /* -13 Hard Fault Handler */
    Fault_Handler, /* -12 MPU Fault Handler */
    Fault_Handler, /* -11 Bus Fault Handler */
    Fault_Handler, /* -10 Usage Fault Handler */
    Fault_Handler, /*  -9 Secure Fault Handler */
    0,
    0,
    0,
    Default_Handler, /*  -5 SVCall Handler */
    Default_Handler, /*  -4 Debug Monitor Handler */
    0,
    Default_Handler,               /*  -2 PendSV Handler */
    Default_Handler,               /*  -1 SysTick Handler */
    [16 + 56] = EthosU_IRQHandler, /*  56 Ethos-U55 NPU Interrupt */
};

void Default_Handler(void) {
  volatile uint32_t* const SCB_ICSR = (volatile uint32_t*)0xE000ED04;
  uint32_t vectactive = *SCB_ICSR & 0x1FF;
  printf(
      "\r\n\r\nFATAL EXCEPTION: Default_Handler triggered for Vector %u "
      "(External IRQ %d)\r\n",
      (unsigned)vectactive, (int)vectactive - 16);
  fflush(stdout);
  while (1) {
  }
}

void Fault_Handler(void) {
  volatile uint32_t* const SCB_ICSR = (volatile uint32_t*)0xE000ED04;
  volatile uint32_t* const SCB_CFSR = (volatile uint32_t*)0xE000ED28;
  volatile uint32_t* const SCB_HFSR = (volatile uint32_t*)0xE000ED2C;
  uint32_t vectactive = *SCB_ICSR & 0x1FF;
  printf(
      "\r\n\r\nFATAL FAULT: Fault_Handler (Vector %u)! CFSR=0x%08X "
      "HFSR=0x%08X\r\n",
      (unsigned)vectactive, (unsigned)*SCB_CFSR, (unsigned)*SCB_HFSR);
  fflush(stdout);
  while (1) {
  }
}

/* Reset Handler called on controller reset */
__attribute__((noreturn, naked)) void Reset_Handler(void) {
  __asm__ volatile(
      // 1. Initialize MSPLIM register if available on Cortex-M55
      "ldr r0, =__STACK_LIMIT\n"
      "msr msplim, r0\n"
      // 2. Perform exact CMSIS System Initialization (VTOR, Clocks, CPACR)
      "bl SystemInit\n"
      // 3. Enable CMSDK APB UART0 Transmit (CTRL at 0x49303008 = 0x01)
      "ldr r0, =0x49303008\n"
      "movs r1, #1\n"
      "str r1, [r0]\n"
      // 4. Enter GCC Newlib C library entry point (_start inside crt0.o)
      "b _start\n"
      :
      :
      : "memory");
}
