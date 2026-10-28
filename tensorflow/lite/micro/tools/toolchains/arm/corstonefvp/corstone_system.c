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

/* Standalone, self-contained system_ARMCM55.c matching Official ARM CMSIS
   Template. No dependencies on external Keil/third_party vendor libraries. */

#include <stdint.h>
#include <stdio.h>

#define SYSTEM_CLOCK 25000000U
uint32_t SystemCoreClock = SYSTEM_CLOCK;

extern const uint32_t __VECTOR_TABLE[];
extern void _exit(int status) __attribute__((noreturn));
extern void corstone_uart_puts(const char* s);

#if __has_include(<ethosu_driver.h>)
#include <ethosu_driver.h>

struct ethosu_driver ethosu_drv_instance;

void EthosU_IRQHandler(void) { ethosu_irq_handler(&ethosu_drv_instance); }
#else
void EthosU_IRQHandler(void) {
  while (1) {
  }
}
#endif

void SystemCoreClockUpdate(void) { SystemCoreClock = SYSTEM_CLOCK; }

void SystemInit(void) {
  // 1. Set Vector Table Offset Register (VTOR) to ITCM base address
  // (0x00000000)
  volatile uint32_t* const SCB_VTOR = (volatile uint32_t*)0xE000ED08;
  *SCB_VTOR = 0x00000000UL;

  // 2. Enable FPU (CP10, CP11) and MVE/Helium (CP0..CP3) completely in CPACR
  // and NSACR
  volatile uint32_t* const SCB_CPACR = (volatile uint32_t*)0xE000ED88;
  volatile uint32_t* const SCB_NSACR = (volatile uint32_t*)0xE000ED8C;
  *SCB_CPACR |=
      (0xFFUL << 0) |
      (0xFUL << 20);  // Full access to CP0..CP3 (MVE) & CP10..CP11 (FPU)
  *SCB_NSACR |= (0xFFUL << 0) | (0xFUL << 20);

  // 3. Enable Instruction and Data Caches / Loop and Branch info cache inside
  // SCB->CCR
  volatile uint32_t* const SCB_CCR = (volatile uint32_t*)0xE000ED14;
  *SCB_CCR |= (1UL << 18) | (1UL << 17) | (1UL << 16);  // LOB, IC, DC bits

  // 4. Enable all fault handlers (UsageFault, BusFault, MemFault)
  volatile uint32_t* const SCB_SHCSR = (volatile uint32_t*)0xE000ED24;
  *SCB_SHCSR |= (1UL << 18) | (1UL << 17) | (1UL << 16);

  // 5. Enable trace/PMU subsystem (DEMCR.TRCENA) and Armv8.1-M PMU cycle
  // counter (PMU_CCNTR at 0xE000307C) for guest timing.
  volatile uint32_t* const DCB_DEMCR = (volatile uint32_t*)0xE000EDFC;
  volatile uint32_t* const PMU_CNTENSET = (volatile uint32_t*)0xE0003C00;
  volatile uint32_t* const PMU_CTRL = (volatile uint32_t*)0xE0003E04;
  *DCB_DEMCR |= (1UL << 24);
  *PMU_CTRL |= (1UL << 0) | (1UL << 2);  // ENABLE | CYCCNT_RESET
  *PMU_CNTENSET = (1UL << 31);           // CCNTR_ENABLE
  __asm__ volatile("dsb\nisb" : : : "memory");

  // 6. Enable CMSDK APB UART0 Transmit (CTRL at 0x49303008 = Bit 0 TXEN) right
  // at system init
  volatile uint32_t* const UART0_CTRL = (volatile uint32_t*)0x49303008;
  *UART0_CTRL = 0x01UL;
  SystemCoreClock = SYSTEM_CLOCK;
}

__attribute__((constructor)) void CorstoneHardwareInit(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
#if __has_include(<ethosu_driver.h>)
  if (ethosu_init(&ethosu_drv_instance, /*base_address=*/(void*)0x48102000,
                  /*fast_memory=*/(void*)0, /*fast_memory_size=*/0,
                  /*secure=*/1, /*privileged=*/1) != 0) {
    corstone_uart_puts("FATAL: ethosu_init failed\r\n");
    _exit(1);
  }

  // Enable NVIC IRQ 56 (Ethos-U55 NPU completion interrupt) in ISER[1] bit 24
  volatile uint32_t* const NVIC_ISER1 = (volatile uint32_t*)0xE000E104;
  *NVIC_ISER1 = (1UL << (56 - 32));
#endif
}
