// REQUIRES: sat
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t s32 = 10;
static volatile uint32_t u32 = 10U;

void ssat() {
    // CHECK-LABEL: <ssat>:
    // CHECK: ssat {{r[0-9]+}}, #0x2, {{r[0-9]+}}
    volatile uint32_t c = __SSAT(s32, 2u);
    // CHECK: ssat {{r[0-9]+}}, #0x5, {{r[0-9]+}}
    volatile uint32_t d = __SSAT(s32, 5u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usat() {
    // CHECK-LABEL: <usat>:
    // CHECK: usat {{r[0-9]+}}, #0x2, {{r[0-9]+}}
    volatile uint32_t c = __USAT(u32, 2u);
    // CHECK: usat {{r[0-9]+}}, #0x5, {{r[0-9]+}}
    volatile uint32_t d = __USAT(u32, 5u);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

