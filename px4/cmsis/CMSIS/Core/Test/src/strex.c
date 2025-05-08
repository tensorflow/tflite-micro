// REQUIRES: ldrex
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint8_t v8;
static volatile uint16_t v16;
static volatile uint32_t v32;

void strexb() {
    // CHECK-LABEL: <strexb>:
    // CHECK: strexb {{r[0-9]+}}, {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __STREXB(0x7u, &v8);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void strexh() {
    // CHECK-LABEL: <strexh>:
    // CHECK: strexh {{r[0-9]+}}, {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __STREXH(0x7u, &v16);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void strexw() {
    // CHECK-LABEL: <strexw>:
    // CHECK: strex {{r[0-9]+}}, {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __STREXW(0x7u, &v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
