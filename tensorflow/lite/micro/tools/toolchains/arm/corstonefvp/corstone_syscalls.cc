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

#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <time.h>

#include <cstddef>
#include <cstdint>

namespace {
constexpr uint64_t kSystemClockHz = 25000000ULL;
constexpr uint64_t kCyclesPerUs = kSystemClockHz / 1000000ULL;
}  // namespace

extern "C" {

extern uint8_t __HeapBase[];
extern uint8_t __HeapLimit[];
extern void _start(void);

__attribute__((target("general-regs-only"))) void corstone_uart_putc(char c) {
  volatile uint32_t* const UART0_DATA =
      reinterpret_cast<volatile uint32_t*>(0x49303000);
  volatile uint32_t* const UART0_STATE =
      reinterpret_cast<volatile uint32_t*>(0x49303004);
  volatile uint32_t* const UART0_CTRL =
      reinterpret_cast<volatile uint32_t*>(0x49303008);
  *UART0_CTRL |= 0x01U;
  while ((*UART0_STATE & 1U) != 0U) {
  }
  *UART0_DATA = static_cast<uint32_t>(static_cast<uint8_t>(c));
}

__attribute__((target("general-regs-only"))) void corstone_uart_puts(
    const char* s) {
  while (*s != '\0') {
    corstone_uart_putc(*s++);
  }
}

__attribute__((target("general-regs-only"))) void corstone_uart_put_dec(
    int val) {
  char buf[16];
  int i = 0;
  uint32_t u = (val < 0) ? (0U - static_cast<uint32_t>(val))
                         : static_cast<uint32_t>(val);
  if (val < 0) {
    corstone_uart_putc('-');
  }
  do {
    buf[i++] = static_cast<char>('0' + (u % 10U));
    u /= 10U;
  } while (u > 0U);
  while (i > 0) {
    corstone_uart_putc(buf[--i]);
  }
}

__attribute__((target("general-regs-only"))) void corstone_uart_put_hex32(
    uint32_t val) {
  constexpr char kHex[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4) {
    corstone_uart_putc(kHex[(val >> shift) & 0xFU]);
  }
}

// Reads the 32-bit Armv8.1-M PMU cycle counter (PMU_CCNTR at 0xE000307C) and
// extends it to 64 bits. Callers measuring spans >171.8s (2^32 cycles at
// 25 MHz) must sample at least once per wrap window. Note that arm-none-eabi
// Newlib defines CLOCKS_PER_SEC = 100 (10 ms clock() ticks); use
// gettimeofday() or corstone_get_cycles() for microsecond / cycle precision.
uint64_t corstone_get_cycles(void) {
  static uint32_t last_ccntr = 0;
  static uint64_t high_cycles = 0;
  volatile uint32_t* const PMU_CCNTR =
      reinterpret_cast<volatile uint32_t*>(0xE000307C);
  uint32_t primask;
  __asm__ volatile(
      "mrs %0, primask\n"
      "cpsid i\n"
      : "=r"(primask)
      :
      : "memory");
  uint32_t cur = *PMU_CCNTR;
  if (cur < last_ccntr) {
    high_cycles += (1ULL << 32);
  }
  last_ccntr = cur;
  uint64_t total = high_cycles | cur;
  __asm__ volatile("msr primask, %0" : : "r"(primask) : "memory");
  return total;
}

int _write(int file, char* ptr, int len) {
  if (file == 1 || file == 2) {
    for (int i = 0; i < len; ++i) {
      corstone_uart_putc(ptr[i]);
    }
    return len;
  }
  return -1;
}

__attribute__((noreturn, target("general-regs-only"))) void _exit(int status) {
  corstone_uart_puts("\r\n__TFLM_EXIT_CODE=");
  corstone_uart_put_dec(status);
  corstone_uart_puts("__\r\n");
  corstone_uart_putc('\x04');  // EOT triggers shutdown when shutdown_on_eot=1
  for (;;) {
    __asm__ volatile("wfi");
  }
}

void* _sbrk(ptrdiff_t incr) {
  static uint8_t* heap_end = __HeapBase;
  uint8_t* prev_heap_end = heap_end;
  if (incr >= 0) {
    if (static_cast<size_t>(__HeapLimit - heap_end) <
        static_cast<size_t>(incr)) {
      errno = ENOMEM;
      return reinterpret_cast<void*>(-1);
    }
    heap_end += incr;
  } else {
    size_t decr = static_cast<size_t>(0) - static_cast<size_t>(incr);
    if (static_cast<size_t>(heap_end - __HeapBase) < decr) {
      errno = ENOMEM;
      return reinterpret_cast<void*>(-1);
    }
    heap_end -= decr;
  }
  return prev_heap_end;
}

int _close(int file) {
  (void)file;
  return -1;
}

int _fstat(int file, struct stat* st) {
  (void)file;
  if (st != nullptr) {
    st->st_mode = S_IFCHR;
  }
  return 0;
}

int _isatty(int file) {
  (void)file;
  return 1;
}

int _lseek(int file, int ptr, int dir) {
  (void)file;
  (void)ptr;
  (void)dir;
  return 0;
}

int _read(int file, char* ptr, int len) {
  (void)file;
  (void)ptr;
  (void)len;
  return 0;
}

int _kill(int pid, int sig) {
  (void)pid;
  (void)sig;
  return -1;
}

int _getpid(void) { return 1; }

clock_t _times(struct tms* buf) {
  uint64_t cycles = corstone_get_cycles();
  clock_t ticks = static_cast<clock_t>(
      (cycles * static_cast<uint64_t>(CLOCKS_PER_SEC)) / kSystemClockHz);
  if (ticks == static_cast<clock_t>(-1)) {
    ticks = 0;
  }
  if (buf != nullptr) {
    buf->tms_utime = ticks;
    buf->tms_stime = 0;
    buf->tms_cutime = 0;
    buf->tms_cstime = 0;
  }
  return ticks;
}

int _gettimeofday(struct timeval* tv, void* tz) {
  (void)tz;
  if (tv != nullptr) {
    uint64_t cycles = corstone_get_cycles();
    tv->tv_sec = static_cast<time_t>(cycles / kSystemClockHz);
    tv->tv_usec =
        static_cast<suseconds_t>((cycles % kSystemClockHz) / kCyclesPerUs);
  }
  return 0;
}

}  // extern "C"
