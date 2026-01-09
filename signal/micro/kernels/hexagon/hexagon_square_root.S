/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

.section        .note.GNU-stack,"",@progbits

// SignalHexagonSqrt32
//      input:  R0 unsigned 32-bit
//      output: R0 unsigned 32-bit
//
// The assembly routine below implements the following:
//
//   uint16_t Sqrt32(uint32_t num) {
//     uint32_t res = 0;
//     uint32_t bit = ((int32_t)1) << 30U;
//     while (bit > num)
//       bit >>= 2;
//     while (bit != 0) {
//       if (num >= res + bit) {
//         num -= res + bit;
//         res = (res >> 1U) + bit;
//       } else {
//         res >>= 1U;
//       }
//       bit >>= 2U;
//     }
//     // Do rounding
//     if (num > res && num != 0xFFFF)
//       ++res;
//     return res;
//   }

.text
.p2align 2
.p2align 4,,15
.globl   SignalHexagonSqrt32
.type    SignalHexagonSqrt32, @function

                                // Register mnemonics
#define num R0                  // input - as in loop above
#define res R1                  // as in loop aboe
#define bit R2                  // as in loop above
#define temp R3                 // the quantity bit + res
#define zcount R4               // leading zeroes
#define res_shift R5            // the quantity res >> 1
#define bit_shift R6            // the quantity bit >> 2

SignalHexagonSqrt32:
                // Set bit to the largest even-power of two
                // that is less than or equal to the input
        {
                res = #0                                // return value
                bit = ##1073741824                      // 2^30
                zcount = cl0(num)                       // count leading zeroes
        }
                zcount = clrbit(zcount, #0)             // even power of 2
        {
                bit = lsr(bit, zcount)                  // 2^30 right shifted
                if (cmp.eq(bit.new, #0)) jump:nt .done  // return if bit == 0
        }
.falign
.loop:
        {
                // Calculate quantities to be used in the conditional below
                temp = add(bit, res)
                res_shift = lsr(res, #1)
                bit_shift = lsr(bit, #2)
        }
        {
                // Conditionally assign to num and res
                p0 = cmp.ltu(temp, num)
                if (p0.new) num = sub(num, temp)
                if (p0.new) res = add(res_shift, bit)
                if (!p0.new) res = res_shift
        }
        {
                // Advance bit >> 2 and exit loop if done
                bit = bit_shift
                if (cmp.gt(bit.new, #0)) jump:t .loop
        }
.falign
.done:
                // if (num > res && res != 0xffff) {
                //   ++res
                // }
                // return res in num (R0)
        {
                temp = ##65535
        }
        {
                p0 = cmp.gt(num, res)
                p0 = !cmp.eq(res, temp)
                if (p0.new) num = add(res, #1)
                if (!p0.new) num = res
        }
        {
                jumpr r31
        }
.size   SignalHexagonSqrt32, .-SignalHexagonSqrt32
