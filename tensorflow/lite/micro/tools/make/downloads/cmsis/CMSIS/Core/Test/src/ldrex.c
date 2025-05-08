// REQUIRES: ldrex
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint8_t v8 = 0x7u;
static volatile uint16_t v16 = 0x7u;
static volatile uint32_t v32 = 0x7u;

void ldrexb() {
    // CHECK-LABEL: <ldrexb>:
    // CHECK: ldrexb {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint8_t result = __LDREXB(&v8);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldrexh() {
    // CHECK-LABEL: <ldrexh>:
    // CHECK: ldrexh {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint16_t result = __LDREXH(&v16);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldrexw() {
    // CHECK-LABEL: <ldrexw>:
    // CHECK: ldrex {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __LDREXW(&v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
