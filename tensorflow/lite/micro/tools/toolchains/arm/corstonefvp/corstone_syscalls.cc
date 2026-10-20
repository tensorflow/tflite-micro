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

#include <sys/stat.h>
#include <sys/time.h>

#include <cstddef>
#include <cstdint>

// 32KB bare-metal C++ heap inside SRAM
__attribute__((aligned(16),
               section(".heap_pool"))) static uint8_t heap_memory[32 * 1024];
static size_t heap_pointer = 0;

extern "C" {

extern void _start(void);

int _write(int file, char* ptr, int len) {
  if (file == 1 || file == 2) {
    volatile uint32_t* const UART0_DATA =
        reinterpret_cast<volatile uint32_t*>(0x49303000);
    volatile uint32_t* const UART0_STATE =
        reinterpret_cast<volatile uint32_t*>(0x49303004);
    volatile uint32_t* const UART0_CTRL =
        reinterpret_cast<volatile uint32_t*>(0x49303008);
    *UART0_CTRL |= 0x01;
    for (int i = 0; i < len; ++i) {
      while ((*UART0_STATE & 1) != 0) {
      }
      *UART0_DATA = static_cast<uint32_t>(static_cast<uint8_t>(ptr[i]));
    }
    return len;
  }
  return -1;
}

void _exit(int status) {
  volatile uint32_t* const UART0_DATA =
      reinterpret_cast<volatile uint32_t*>(0x49303000);
  volatile uint32_t* const UART0_STATE =
      reinterpret_cast<volatile uint32_t*>(0x49303004);
  while ((*UART0_STATE & 1) != 0) {
  }
  *UART0_DATA = 0x04;  // ASCII EOT triggers FVP shutdown when shutdown_on_eot=1
  while (1) {
  }
}

void* _sbrk(int incr) {
  if (incr == 0) {
    return &heap_memory[heap_pointer];
  }
  if (heap_pointer + incr > sizeof(heap_memory)) {
    return reinterpret_cast<void*>(-1);
  }
  void* prev = &heap_memory[heap_pointer];
  heap_pointer += incr;
  return prev;
}

int _close(int file) { return -1; }
int _fstat(int file, struct stat* st) {
  if (st) st->st_mode = S_IFCHR;
  return 0;
}
int _isatty(int file) { return 1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _read(int file, char* ptr, int len) { return 0; }
int _kill(int pid, int sig) { return -1; }
int _getpid(void) { return 1; }

int _gettimeofday(struct timeval* tv, void* tz) {
  if (tv) {
    volatile uint32_t* const DWT_CYCCNT =
        reinterpret_cast<volatile uint32_t*>(0xE0001004);
    uint32_t cycles = *DWT_CYCCNT;
    tv->tv_sec = cycles / 25000000;
    tv->tv_usec = (cycles % 25000000) / 25;
  }
  return 0;
}

}  // extern "C"
