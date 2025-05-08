// REQUIRES: armv7a
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

static volatile uint32_t u32;

void get_actlr() {
    // CHECK-LABEL: <get_actlr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c1, c0, #0x1
    volatile uint32_t result = __get_ACTLR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_actlr() {
    // CHECK-LABEL: <set_actlr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c1, c0, #0x1
    __set_ACTLR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_cpacr() {
    // CHECK-LABEL: <get_cpacr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c1, c0, #0x2
    volatile uint32_t result = __get_CPACR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_cpacr() {
    // CHECK-LABEL: <set_cpacr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c1, c0, #0x2
    __set_CPACR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_dfsr() {
    // CHECK-LABEL: <get_dfsr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c5, c0, #0x0
    volatile uint32_t result = __get_DFSR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dfsr() {
    // CHECK-LABEL: <set_dfsr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c5, c0, #0x0
    __set_DFSR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_ifsr() {
    // CHECK-LABEL: <get_ifsr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c5, c0, #0x1
    volatile uint32_t result = __get_IFSR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_ifsr() {
    // CHECK-LABEL: <set_ifsr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c5, c0, #0x1
    __set_IFSR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_isr() {
    // CHECK-LABEL: <get_isr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c12, c1, #0x0
    volatile uint32_t result = __get_ISR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_cbar() {
    // CHECK-LABEL: <get_cbar>:
    // CHECK: mrc p15, #0x4, {{r[0-9]+}}, c15, c0, #0x0
    volatile uint32_t result = __get_CBAR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_ttbr0() {
    // CHECK-LABEL: <get_ttbr0>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c2, c0, #0x0
    volatile uint32_t result = __get_TTBR0();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_ttbr0() {
    // CHECK-LABEL: <set_ttbr0>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c2, c0, #0x0
    __set_TTBR0(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_dacr() {
    // CHECK-LABEL: <get_dacr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c3, c0, #0x0
    volatile uint32_t result = __get_DACR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dacr() {
    // CHECK-LABEL: <set_dacr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c3, c0, #0x0
    __set_DACR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_sctlr() {
    // CHECK-LABEL: <get_sctlr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c1, c0, #0x0
    volatile uint32_t result = __get_SCTLR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_sctlr() {
    // CHECK-LABEL: <set_sctlr>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c1, c0, #0x0
    __set_SCTLR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_mpidr() {
    // CHECK-LABEL: <get_mpidr>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c0, c0, #0x5
    volatile uint32_t result = __get_MPIDR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_vbar() {
    // CHECK-LABEL: <get_vbar>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c12, c0, #0x0
    volatile uint32_t result = __get_VBAR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_vbar() {
    // CHECK-LABEL: <set_vbar>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c12, c0, #0x0
    __set_VBAR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_mvbar() {
    // CHECK-LABEL: <get_mvbar>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c12, c0, #0x1
    volatile uint32_t result = __get_MVBAR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_mvbar() {
    // CHECK-LABEL: <set_mvbar>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c12, c0, #0x1
    __set_MVBAR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_cntfrq() {
    // CHECK-LABEL: <get_cntfrq>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c14, c0, #0x0
    volatile uint32_t result = __get_CNTFRQ();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_cntfrq() {
    // CHECK-LABEL: <set_cntfrq>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c14, c0, #0x0
    __set_CNTFRQ(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_cntp_tval() {
    // CHECK-LABEL: <get_cntp_tval>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c14, c2, #0x0
    volatile uint32_t result = __get_CNTP_TVAL();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_cntp_tval() {
    // CHECK-LABEL: <set_cntp_tval>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c14, c2, #0x0
    __set_CNTP_TVAL(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_cntp_ctl() {
    // CHECK-LABEL: <get_cntp_ctl>:
    // CHECK: mrc p15, #0x0, {{r[0-9]+}}, c14, c2, #0x1
    volatile uint32_t result = __get_CNTP_CTL();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_cntp_ctl() {
    // CHECK-LABEL: <set_cntp_ctl>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c14, c2, #0x1
    __set_CNTP_CTL(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_tlbiall() {
    // CHECK-LABEL: <set_tlbiall>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c8, c7, #0x0
    __set_TLBIALL(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_bpiall() {
    // CHECK-LABEL: <set_bpiall>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c5, #0x6
    __set_BPIALL(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_iciallu() {
    // CHECK-LABEL: <set_iciallu>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c5, #0x0
    __set_ICIALLU(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_icimvac() {
    // CHECK-LABEL: <set_icimvac>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c5, #0x1
    __set_ICIMVAC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dccmvac() {
    // CHECK-LABEL: <set_dccmvac>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c10, #0x1
    __set_DCCMVAC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dcimvac() {
    // CHECK-LABEL: <set_dcimvac>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c6, #0x1
    __set_DCIMVAC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dccimvac() {
    // CHECK-LABEL: <set_dccimvac>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c14, #0x1
    __set_DCCIMVAC(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_csselr() {
    // CHECK-LABEL: <get_csselr>:
    // CHECK: mrc p15, #0x2, {{r[0-9]+}}, c0, c0, #0x0
    volatile uint32_t result = __get_CSSELR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_csselr() {
    // CHECK-LABEL: <set_csselr>:
    // CHECK: mcr p15, #0x2, {{r[0-9]+}}, c0, c0, #0x0
    __set_CSSELR(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_ccsidr() {
    // CHECK-LABEL: <get_ccsidr>:
    // CHECK: mrc p15, #0x1, {{r[0-9]+}}, c0, c0, #0x0
    volatile uint32_t result = __get_CCSIDR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void get_clidr() {
    // CHECK-LABEL: <get_clidr>:
    // CHECK: mrc p15, #0x1, {{r[0-9]+}}, c0, c0, #0x1
    volatile uint32_t result = __get_CLIDR();
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dcisw() {
    // CHECK-LABEL: <set_dcisw>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c6, #0x2
    __set_DCISW(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dccsw() {
    // CHECK-LABEL: <set_dccsw>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c10, #0x2
    __set_DCCSW(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void set_dccisw() {
    // CHECK-LABEL: <set_dccisw>:
    // CHECK: mcr p15, #0x0, {{r[0-9]+}}, c7, c14, #0x2
    __set_DCCISW(u32);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
