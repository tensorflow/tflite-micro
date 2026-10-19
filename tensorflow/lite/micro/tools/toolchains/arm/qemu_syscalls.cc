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

#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

void* __dso_handle = nullptr;

extern "C" void __libc_init_array(void);
extern int main(int argc, char** argv);

namespace {

inline int32_t syscall1(int32_t num, intptr_t a1) {
  register intptr_t r0 asm("r0") = a1;
  register int32_t r7 asm("r7") = num;
  asm volatile("svc #0" : "+r"(r0) : "r"(r7) : "memory");
  return r0;
}

inline int32_t syscall2(int32_t num, intptr_t a1, intptr_t a2) {
  register intptr_t r0 asm("r0") = a1;
  register intptr_t r1 asm("r1") = a2;
  register int32_t r7 asm("r7") = num;
  asm volatile("svc #0" : "+r"(r0) : "r"(r1), "r"(r7) : "memory");
  return r0;
}

inline int32_t syscall3(int32_t num, intptr_t a1, intptr_t a2, intptr_t a3) {
  register intptr_t r0 asm("r0") = a1;
  register intptr_t r1 asm("r1") = a2;
  register intptr_t r2 asm("r2") = a3;
  register int32_t r7 asm("r7") = num;
  asm volatile("svc #0" : "+r"(r0) : "r"(r1), "r"(r2), "r"(r7) : "memory");
  return r0;
}

#ifdef __PICOLIBC__
static int qemu_putc(char c, FILE* /*file*/) {
  syscall3(4, 1, (intptr_t)&c, 1);
  return c;
}

static FILE qemu_stdout =
    FDEV_SETUP_STREAM(qemu_putc, nullptr, nullptr, _FDEV_SETUP_WRITE);
#endif

}  // namespace

#ifdef __PICOLIBC__
extern "C" {
FILE* const stdin = &qemu_stdout;
FILE* const stdout = &qemu_stdout;
FILE* const stderr = &qemu_stdout;
}
#endif

extern "C" {

int _write(int file, const void* ptr, int len) {
  return syscall3(4, file, (intptr_t)ptr, len);
}
int write(int file, const void* ptr, int len) { return _write(file, ptr, len); }

void _exit(int status) {
  syscall1(1, status);
  while (1);
}

void* _sbrk(intptr_t incr) {
  intptr_t current_brk = syscall1(45, 0);
  if (incr == 0) return (void*)current_brk;

  intptr_t ret_brk = syscall1(45, current_brk + incr);
  if (ret_brk == current_brk) return (void*)-1;
  return (void*)current_brk;
}
void* sbrk(intptr_t incr) { return _sbrk(incr); }

int _close(int file) { return -1; }
int close(int file) { return _close(file); }

int _fstat(int file, struct stat* st) {
  if (st) {
    st->st_mode = S_IFCHR;
  }
  return 0;
}
int fstat(int file, struct stat* st) { return _fstat(file, st); }

int _isatty(int file) { return 1; }
int isatty(int file) { return _isatty(file); }

int _lseek(int file, int ptr, int dir) { return 0; }
int lseek(int file, int ptr, int dir) { return _lseek(file, ptr, dir); }

int _read(int file, void* ptr, int len) { return 0; }
int read(int file, void* ptr, int len) { return _read(file, ptr, len); }

int _kill(int pid, int sig) { return -1; }
int kill(int pid, int sig) { return _kill(pid, sig); }

int _getpid(void) { return 1; }
int getpid(void) { return _getpid(); }

int _gettimeofday(struct timeval* tv, void* tz) {
  return syscall2(78, (intptr_t)tv, (intptr_t)tz);
}
int gettimeofday(struct timeval* tv, void* tz) { return _gettimeofday(tv, tz); }

void* __aeabi_read_tp(void) {
  static char dummy_tls[64];
  return dummy_tls;
}

void _init() {}
void _fini() {}

void _start() {
  __libc_init_array();
  _exit(main(0, nullptr));
}

}  // extern "C"
