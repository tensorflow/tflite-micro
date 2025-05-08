// REQUIRES: armv8m
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint8_t v8;
static volatile uint16_t v16;
static volatile uint32_t v32;

void stlb() {
    // CHECK-LABEL: <stlb>:
    // CHECK: stlb {{r[0-9]+}}, [{{r[0-9]+}}]
    __STLB(0x7u, &v8);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void stlh() {
    // CHECK-LABEL: <stlh>:
    // CHECK: stlh {{r[0-9]+}}, [{{r[0-9]+}}]
    __STLH(0x7u, &v16);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void stl() {
    // CHECK-LABEL: <stl>:
    // CHECK: stl {{r[0-9]+}}, [{{r[0-9]+}}]
    __STL(0x7u, &v32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
