// REQUIRES: armv8m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint8_t v8 = 0x7u;
static volatile uint16_t v16 = 0x7u;
static volatile uint32_t v32 = 0x7u;

void ldaexb() {
    // CHECK-LABEL: <ldaexb>:
    // CHECK: ldaexb {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint8_t result = __LDAEXB(&v8);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldaexh() {
    // CHECK-LABEL: <ldaexh>:
    // CHECK: ldaexh {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint16_t result = __LDAEXH(&v16);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ldaex() {
    // CHECK-LABEL: <ldaex>:
    // CHECK: ldaex {{r[0-9]+}}, [{{r[0-9]+}}]
    volatile uint32_t result = __LDAEX(&v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
