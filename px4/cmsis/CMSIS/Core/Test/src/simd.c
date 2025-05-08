// REQUIRES: dsp
// RUN: %cc% %ccflags% %ccout% %T/%basename_t.o %s; llvm-objdump --mcpu=%mcpu% -d %T/%basename_t.o | FileCheck --allow-unused-prefixes --check-prefixes %prefixes% %s

#include "cmsis_compiler.h"

volatile static int32_t s32_1 = 0x47;
volatile static int32_t s32_2 = 0x11;
volatile static int32_t s32_3 = 0x15;
volatile static uint8_t u8 = 5u;

/* ADD8 */

void sadd8() {
    // CHECK-LABEL: <sadd8>:
    // CHECK: sadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qadd8() {
    // CHECK-LABEL: <qadd8>:
    // CHECK: qadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shadd8() {
    // CHECK-LABEL: <shadd8>:
    // CHECK: shadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uadd8() {
    // CHECK-LABEL: <uadd8>:
    // CHECK: uadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqadd8() {
    // CHECK-LABEL: <uqadd8>:
    // CHECK: uqadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhadd8() {
    // CHECK-LABEL: <uhadd8>:
    // CHECK: uhadd8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHADD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* SUB8 */

void ssub8() {
    // CHECK-LABEL: <ssub8>:
    // CHECK: ssub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SSUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qsub8() {
    // CHECK-LABEL: <qsub8>:
    // CHECK: qsub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QSUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shsub8() {
    // CHECK-LABEL: <shsub8>:
    // CHECK: shsub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHSUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usub8() {
    // CHECK-LABEL: <usub8>:
    // CHECK: usub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __USUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqsub8() {
    // CHECK-LABEL: <uqsub8>:
    // CHECK: uqsub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQSUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhsub8() {
    // CHECK-LABEL: <uhsub8>:
    // CHECK: uhsub8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHSUB8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* ADD16 */

void sadd16() {
    // CHECK-LABEL: <sadd16>:
    // CHECK: sadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qadd16() {
    // CHECK-LABEL: <qadd16>:
    // CHECK: qadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shadd16() {
    // CHECK-LABEL: <shadd16>:
    // CHECK: shadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uadd16() {
    // CHECK-LABEL: <uadd16>:
    // CHECK: uadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqadd16() {
    // CHECK-LABEL: <uqadd16>:
    // CHECK: uqadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhadd16() {
    // CHECK-LABEL: <uhadd16>:
    // CHECK: uhadd16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHADD16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* SUB16 */

void ssub16() {
    // CHECK-LABEL: <ssub16>:
    // CHECK: ssub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SSUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qsub16() {
    // CHECK-LABEL: <qsub16>:
    // CHECK: qsub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QSUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shsub16() {
    // CHECK-LABEL: <shsub16>:
    // CHECK: shsub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHSUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usub16() {
    // CHECK-LABEL: <usub16>:
    // CHECK: usub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __USUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqsub16() {
    // CHECK-LABEL: <uqsub16>:
    // CHECK: uqsub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQSUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhsub16() {
    // CHECK-LABEL: <uhsub16>:
    // CHECK: uhsub16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHSUB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* ASX */

void sasx() {
    // CHECK-LABEL: <sasx>:
    // CHECK: sasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qasx() {
    // CHECK-LABEL: <qasx>:
    // CHECK: qasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shasx() {
    // CHECK-LABEL: <shasx>:
    // CHECK: shasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uasx() {
    // CHECK-LABEL: <uasx>:
    // CHECK: uasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqasx() {
    // CHECK-LABEL: <uqasx>:
    // CHECK: uqasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhasx() {
    // CHECK-LABEL: <uhasx>:
    // CHECK: uhasx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHASX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* SAX */

void ssax() {
    // CHECK-LABEL: <ssax>:
    // CHECK: ssax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SSAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qsax() {
    // CHECK-LABEL: <qsax>:
    // CHECK: qsax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QSAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void shsax() {
    // CHECK-LABEL: <shsax>:
    // CHECK: shsax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SHSAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usax() {
    // CHECK-LABEL: <usax>:
    // CHECK: usax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __USAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uqsax() {
    // CHECK-LABEL: <uqsax>:
    // CHECK: uqsax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UQSAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uhsax() {
    // CHECK-LABEL: <uhsax>:
    // CHECK: uhsax {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UHSAX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* SAT */

void usad8() {
    // CHECK-LABEL: <usad8>:
    // CHECK: usad8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __USAD8(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usada8() {
    // CHECK-LABEL: <usada8>:
    // CHECK: usada8 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __USADA8(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void ssat16() {
    // CHECK-LABEL: <ssat16>:
    // CHECK: ssat16 {{r[0-9]+}}, #0x5, {{r[0-9]+}}
    volatile uint32_t result = __SSAT16(s32_1, 0x05);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void usat16() {
    // CHECK-LABEL: <usat16>:
    // CHECK: usat16 {{r[0-9]+}}, #0x5, {{r[0-9]+}}
    volatile uint32_t result = __USAT16(s32_1, 0x05);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uxtb16() {
    // CHECK-LABEL: <uxtb16>:
    // CHECK: uxtb16 {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UXTB16(s32_1);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void uxtab16() {
    // CHECK-LABEL: <uxtab16>:
    // CHECK: uxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __UXTAB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void sxtb16() {
    // CHECK-LABEL: <sxtb16>:
    // CHECK: sxtb16 {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SXTB16(s32_1);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void sxtab16() {
    // CHECK-LABEL: <sxtab16>:
    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SXTAB16(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

/* MUL */

void smuad() {
    // CHECK-LABEL: <smuad>:
    // CHECK: smuad {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMUAD(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smuadx() {
    // CHECK-LABEL: <smuadx>:
    // CHECK: smuadx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMUADX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlad() {
    // CHECK-LABEL: <smlad>:
    // CHECK: smlad {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLAD(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smladx() {
    // CHECK-LABEL: <smladx>:
    // CHECK: smladx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLADX(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlald() {
    // CHECK-LABEL: <smlald>:
    // CHECK: smlald {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLALD(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlaldx() {
    // CHECK-LABEL: <smlaldx>:
    // CHECK: smlaldx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLALDX(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smusd() {
    // CHECK-LABEL: <smusd>:
    // CHECK: smusd {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMUSD(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smusdx() {
    // CHECK-LABEL: <smusdx>:
    // CHECK: smusdx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMUSDX(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlsd() {
    // CHECK-LABEL: <smlsd>:
    // CHECK: smlsd {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLSD(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlsdx() {
    // CHECK-LABEL: <smlsdx>:
    // CHECK: smlsdx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLSDX(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlsld() {
    // CHECK-LABEL: <smlsld>:
    // CHECK: smlsld {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLSLD(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smlsldx() {
    // CHECK-LABEL: <smlsldx>:
    // CHECK: smlsldx {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SMLSLDX(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void sel() {
    // CHECK-LABEL: <sel>:
    // CHECK: sel {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __SEL(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qadd() {
    // CHECK-LABEL: <qadd>:
    // CHECK: qadd {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QADD(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void qsub() {
    // CHECK-LABEL: <qsub>:
    // CHECK: qsub {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile uint32_t result = __QSUB(s32_1, s32_2);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void pkhbt0() {
    // CHECK-LABEL: <pkhbt0>:
    // CHECK: {{pkhtb|pkhbt}} {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-NOT: , lsl
    // CHECK-NOT: , asr
    volatile uint32_t result = __PKHBT(s32_1, s32_2, 0);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void pkhbt() {
    // CHECK-LABEL: <pkhbt>:
    // CHECK: pkhbt {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, lsl #11
    volatile uint32_t result = __PKHBT(s32_1, s32_2, 11);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void pkhtb0() {
    // CHECK-LABEL: <pkhtb0>:
    // CHECK: {{pkhtb|pkhbt}} {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-NOT: , lsl
    // CHECK-NOT: , asr
    volatile uint32_t result = __PKHTB(s32_1, s32_2, 0);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void pkhtb() {
    // CHECK-LABEL: <pkhtb>:
    // CHECK: pkhtb {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, asr #11
    volatile uint32_t result = __PKHTB(s32_1, s32_2, 11);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void sxtb16_ror() {    
    // CHECK-LABEL: <sxtb16_ror>:
    // CHECK: sxtb16 {{r[0-9]+}}, {{r[0-9]+}}, ror #8
    volatile uint32_t result = __SXTB16_RORn(s32_1, 8);

    // CHECK: sxtb16 {{r[0-9]+}}, {{r[0-9]+}}, ror #16
    result = __SXTB16_RORn(s32_1, 16);

    // CHECK: sxtb16 {{r[0-9]+}}, {{r[0-9]+}}, ror #24
    result = __SXTB16_RORn(s32_1, 24);

    // CHECK-THUMB: ror.w [[REG:r[0-9]+]], {{r[0-9]+}}, {{#5|#0x5}}
    // CHECK-ARM: ror [[REG:r[0-9]+]], {{r[0-9]+}}, {{#5|#0x5}}
    // CHECK: sxtb16 {{r[0-9]+}}, [[REG]]
    // CHECK-NOT: , ror
    result = __SXTB16_RORn(s32_1, 5);

    // CHECK-THUMB: ror{{.w|ne|s}} {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-ARM: ror{{(ne)?}} {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK: sxtb16 {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-NOT: , ror
    result = __SXTB16_RORn(s32_1, u8);

    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void sxtab16_ror() {
    // CHECK-LABEL: <sxtab16_ror>:

    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, ror #8
    volatile uint32_t result = __SXTAB16_RORn(s32_1, s32_2, 8);

    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, ror #16
    result = __SXTAB16_RORn(s32_1, s32_2, 16);

    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, ror #24
    result = __SXTAB16_RORn(s32_1, s32_2, 24);

    // CHECK-THUMB: ror.w [[REG:r[0-9]+]], {{r[0-9]+}}, {{#5|#0x5}}
    // CHECK-ARM: ror [[REG:r[0-9]+]], {{r[0-9]+}}, {{#5|#0x5}}
    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, [[REG]]
    // CHECK-NOT: , ror
    result = __SXTAB16_RORn(s32_1, s32_2, 5);

    // CHECK-THUMB: ror{{.w|ne|s}} {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-ARM: ror{{(ne)?}} {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK: sxtab16 {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    // CHECK-NOT: , ror
    result = __SXTAB16_RORn(s32_1, s32_2, u8);

    // CHECK: {{(bx lr)|(pop {.*pc})}}
}

void smmla() {
    // CHECK-LABEL: <smmla>:
    // CHECK: smmla {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}, {{r[0-9]+}}
    volatile int32_t result = __SMMLA(s32_1, s32_2, s32_3);
    // CHECK: {{(bx lr)|(pop {.*pc})}}
}
