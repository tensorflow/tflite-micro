/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/debug_log.h"
#include <stdarg.h> 
#include <stdio.h>  
#include "tensorflow/lite/micro/riscv32_baremetal/lightweight_snprintf.cc"

// UART definitions- Datasheet: https://pdos.csail.mit.edu/6.S081/2024/lec/16550.pdf
#define UART0_BASE 0x10000000
#define REG(base, offset) ((*((volatile unsigned char *)(base + offset))))
#define UART0_DR    REG(UART0_BASE, 0x00)   // Data register (read/write
#define UART0_FCR   REG(UART0_BASE, 0x02)   // FIFO Control register
#define UART0_LSR   REG(UART0_BASE, 0x05)   // Line Status register

#define UARTFCR_FFENA 0x01   // FIFO Enable bit
#define UARTLSR_THRE  0x20   // Transmitter Holding Register Empty bit
#define UART0_FF_THR_EMPTY (UART0_LSR & UARTLSR_THRE)   // Macro: check if the transmitter is ready to accept a new character

// uart_putc()
// Waits until the UART transmitter is ready, then writes one character.

static void uart_putc(char c) {
  while (!UART0_FF_THR_EMPTY);
  UART0_DR = c;
}

// uart_puts()
// Sends a null-terminated string over UART.

static void uart_puts(const char *str) {
  while (*str) {
    uart_putc(*str++);
  }
}

// DebugLog()
// TFLM's platform hook for debug logging.

extern "C" void DebugLog(const char* format, va_list args) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  char buffer[256];  // Temporary buffer for formatted string
  int len = mini_vsnprintf(buffer, sizeof(buffer), format, args);
  if (len > 0) {
    UART0_FCR = UARTFCR_FFENA;
    uart_puts(buffer);
  }
#endif
}

// DebugVsnprintf()
// A utility for TFLM internal formatting, using the same lightweight function.

#ifndef TF_LITE_STRIP_ERROR_STRINGS
extern "C" int DebugVsnprintf(char* buffer, size_t buf_size,
                              const char* format, va_list vlist) {
  return mini_vsnprintf(buffer, buf_size, format, vlist);
}
#endif

