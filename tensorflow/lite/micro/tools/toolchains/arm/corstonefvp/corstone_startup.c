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

extern uint32_t __INITIAL_SP;
extern uint32_t __STACK_LIMIT;

extern void _start(void);  // GCC Newlib C runtime entry point
extern void _exit(int status) __attribute__((noreturn));
extern void SystemInit(void);  // CMSIS System Initialization

extern void corstone_uart_puts(const char* s);
extern void corstone_uart_put_dec(int val);
extern void corstone_uart_put_hex32(uint32_t val);

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
    [16 ... 71] = Default_Handler,
    [16 + 56] = EthosU_IRQHandler, /*  56 Ethos-U55 NPU Interrupt */
    [73 ... 79] = Default_Handler,
};

/* Uses only general-purpose registers and raw polling UART0 (never stdio or
   malloc) so faults caused by stack overflow (STKOF), lazy FP state
   preservation, or heap corruption still emit diagnostics and terminate FVP
   cleanly without double-faulting into Cortex-M55 LOCKUP. */
__attribute__((noreturn, noinline, used, target("general-regs-only"))) void
Fault_Handler_C(const uint32_t* fault_sp, uint32_t exc_return) {
  volatile uint32_t* const SCB_ICSR = (volatile uint32_t*)0xE000ED04;
  volatile uint32_t* const SCB_CFSR = (volatile uint32_t*)0xE000ED28;
  volatile uint32_t* const SCB_HFSR = (volatile uint32_t*)0xE000ED2C;
  volatile uint32_t* const SCB_MMFAR = (volatile uint32_t*)0xE000ED34;
  volatile uint32_t* const SCB_BFAR = (volatile uint32_t*)0xE000ED38;
  uint32_t vectactive = *SCB_ICSR & 0x1FFU;

  if (vectactive >= 16U) {
    corstone_uart_puts("\r\n\r\nFATAL EXCEPTION: Default_Handler (Vector ");
    corstone_uart_put_dec((int)vectactive);
    corstone_uart_puts(", External IRQ ");
    corstone_uart_put_dec((int)vectactive - 16);
    corstone_uart_puts(")\r\n");
  } else {
    uint32_t cfsr = *SCB_CFSR;
    uint32_t hfsr = *SCB_HFSR;
    corstone_uart_puts("\r\n\r\nFATAL FAULT: Fault_Handler (Vector ");
    corstone_uart_put_dec((int)vectactive);
    corstone_uart_puts(")! CFSR=0x");
    corstone_uart_put_hex32(cfsr);
    corstone_uart_puts(" HFSR=0x");
    corstone_uart_put_hex32(hfsr);
    if ((cfsr & 0x00000080U) != 0U) {
      corstone_uart_puts(" MMFAR=0x");
      corstone_uart_put_hex32(*SCB_MMFAR);
    }
    if ((cfsr & 0x00008000U) != 0U) {
      corstone_uart_puts(" BFAR=0x");
      corstone_uart_put_hex32(*SCB_BFAR);
    }

    uintptr_t sp_addr = (uintptr_t)fault_sp;
    uintptr_t limit_addr = (uintptr_t)(&__STACK_LIMIT);
    uintptr_t top_addr = (uintptr_t)(&__INITIAL_SP);
    if (((exc_return & (1UL << 2)) == 0U) && ((sp_addr & 0x3U) == 0U) &&
        (sp_addr >= limit_addr) && (sp_addr + 32U <= top_addr)) {
      corstone_uart_puts(" PC=0x");
      corstone_uart_put_hex32(fault_sp[6]);
      corstone_uart_puts(" LR=0x");
      corstone_uart_put_hex32(fault_sp[5]);
      corstone_uart_puts(" xPSR=0x");
      corstone_uart_put_hex32(fault_sp[7]);
    }
    corstone_uart_puts("\r\n");
  }

  _exit(128 + (int)(vectactive & 0x7FU));
}

__attribute__((noreturn, naked)) void Fault_Handler(void) {
  __asm__ volatile(
      "mrs r0, msp\n"
      "mov r1, lr\n"
      "movs r2, #0\n"
      "msr msplim, r2\n"
      "ldr r3, =0xE000EF34\n"
      "ldr r2, [r3]\n"
      "bic r2, r2, #1\n"
      "str r2, [r3]\n"
      "ldr r2, =__INITIAL_SP\n"
      "msr msp, r2\n"
      "dsb\n"
      "isb\n"
      "b Fault_Handler_C\n");
}

__attribute__((noreturn, alias("Fault_Handler"))) void Default_Handler(void);

/* Reset Handler called on controller reset */
__attribute__((noreturn, naked)) void Reset_Handler(void) {
  __asm__ volatile(
      // 1. Initialize MSPLIM register on Cortex-M55
      "ldr r0, =__STACK_LIMIT\n"
      "msr msplim, r0\n"
      // 2. Perform CMSIS System Initialization (VTOR, CPACR, PMU, UART0)
      "bl SystemInit\n"
      // 3. Enter GCC Newlib C library entry point (_start inside crt0.o)
      "b _start\n"
      :
      :
      : "memory");
}
