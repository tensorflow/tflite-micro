
// REQUIRES: thumb-2
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint8_t v8 = 0x7u;
static volatile uint16_t v16 = 0x7u;
static volatile uint32_t v32 = 0x7u;

void ldrbt() {
    // CHECK-LABEL: <ldrbt>:
    // CHECK: ldrbt {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint8_t result = __LDRBT(&v8);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldrht() {
    // CHECK-LABEL: <ldrht>:
    // CHECK: ldrht {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint16_t result = __LDRHT(&v16);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldrt() {
    // CHECK-LABEL: <ldrt>:
    // CHECK: ldrt {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __LDRT(&v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
