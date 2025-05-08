// Copyright 2018 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// kernel_msa.h: a collection of MSA optimized kernels.
// Check in kernel_default.h which one(s) are actually used by default.
// Others are mere experiments; they are still covered by tests
// in case they might be useful some day.

#ifndef GEMMLOWP_INTERNAL_KERNEL_MSA_H_
#define GEMMLOWP_INTERNAL_KERNEL_MSA_H_

#include "kernel.h"

#include <msa.h>
#include <cassert>

namespace gemmlowp {

#ifdef GEMMLOWP_MSA

// Some convenience macros to hide differences between MIPS32 and MIPS64.
#ifdef GEMMLOWP_MIPS_64
#define GEMMLOWP_MIPS_XADDU "daddu"
#define GEMMLOWP_MIPS_XADDIU "daddiu"
#define GEMMLOWP_MIPS_XSLL "dsll"
#else
#define GEMMLOWP_MIPS_XADDU "addu"
#define GEMMLOWP_MIPS_XADDIU "addiu"
#define GEMMLOWP_MIPS_XSLL "sll"
#endif

// Our main GEMM kernel.
struct MSA_Kernel12x8Depth2 : KernelBase {
  typedef KernelFormat<KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 3>,
                       KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 2> >
      Format;

  const char* Name() const override { return "MSA, 12x8, depth 2"; }

  // TODO(benoitjacob): reorder function arguments so dst comes last
  void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
           std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
           const std::uint8_t* rhs_ptr, std::size_t start_depth,
           std::size_t run_depth) const override {
    ScopedProfilingLabel label("optimized kernel (MSA 12x8)");
// See comments above for why we need local numerical labels in our asm.
#define GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "1"
#define GEMMLOWP_LABEL_BEFORE_LOOP "2"
#define GEMMLOWP_LABEL_LOOP "3"
#define GEMMLOWP_LABEL_AFTER_LOOP "4"

    assert(dst_row_stride == 1);
    asm volatile(
        // Multiply dst_col_stride by 4 == sizeof(int32) to use
        // it as a byte offset below.
        GEMMLOWP_MIPS_XSLL
        " %[dst_col_stride], %[dst_col_stride], 2\n"

        // Check if start_depth==0 to decide whether we will clear
        // accumulators or load existing accumulators.
        "beqz   %[start_depth], " GEMMLOWP_LABEL_CLEAR_ACCUMULATORS "f\n"

        // Load accumulators (start_depth != 0).
        GEMMLOWP_MIPS_XADDU " $a0, %[dst_ptr], %[dst_col_stride]\n"
        "ld.w   $w0,  (0*16)(%[dst_ptr])\n"
        "ld.w   $w4,  (1*16)(%[dst_ptr])\n"
        "ld.w   $w8,  (2*16)(%[dst_ptr])\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "ld.w   $w1,  (0*16)($a0)\n"
        "ld.w   $w5,  (1*16)($a0)\n"
        "ld.w   $w9,  (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "ld.w   $w2,  (0*16)($a1)\n"
        "ld.w   $w6,  (1*16)($a1)\n"
        "ld.w   $w10, (2*16)($a1)\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "ld.w   $w3,  (0*16)($a0)\n"
        "ld.w   $w7,  (1*16)($a0)\n"
        "ld.w   $w11, (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "ld.w   $w12, (0*16)($a1)\n"
        "ld.w   $w16, (1*16)($a1)\n"
        "ld.w   $w20, (2*16)($a1)\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "ld.w   $w13, (0*16)($a0)\n"
        "ld.w   $w17, (1*16)($a0)\n"
        "ld.w   $w21, (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "ld.w   $w14, (0*16)($a1)\n"
        "ld.w   $w18, (1*16)($a1)\n"
        "ld.w   $w22, (2*16)($a1)\n"
        "ld.w   $w15, (0*16)($a0)\n"
        "ld.w   $w19, (1*16)($a0)\n"
        "ld.w   $w23, (2*16)($a0)\n"
        "b " GEMMLOWP_LABEL_BEFORE_LOOP "f\n"

        GEMMLOWP_LABEL_CLEAR_ACCUMULATORS ":\n"
        // Clear accumulators (start_depth == 0).
        "ldi.w  $w0,  0\n"
        "ldi.w  $w4,  0\n"
        "ldi.w  $w8,  0\n"
        "ldi.w  $w1,  0\n"
        "ldi.w  $w5,  0\n"
        "ldi.w  $w9,  0\n"
        "ldi.w  $w2,  0\n"
        "ldi.w  $w6,  0\n"
        "ldi.w  $w10, 0\n"
        "ldi.w  $w3,  0\n"
        "ldi.w  $w7,  0\n"
        "ldi.w  $w11, 0\n"
        "ldi.w  $w12, 0\n"
        "ldi.w  $w16, 0\n"
        "ldi.w  $w20, 0\n"
        "ldi.w  $w13, 0\n"
        "ldi.w  $w17, 0\n"
        "ldi.w  $w21, 0\n"
        "ldi.w  $w14, 0\n"
        "ldi.w  $w18, 0\n"
        "ldi.w  $w22, 0\n"
        "ldi.w  $w15, 0\n"
        "ldi.w  $w19, 0\n"
        "ldi.w  $w23, 0\n"

        GEMMLOWP_LABEL_BEFORE_LOOP ":\n"

        GEMMLOWP_LABEL_LOOP ":\n"
        // Overview of register layout:
        //
        // A half of the 2 2x4 cells of Rhs is stored in 16bit in w28-w31
        // (each register contains 4 replicas of a pair of elements).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in w24-w26.
        // A 12x8 block of accumulators is stored in 32bit in w0-w23.
        //
        //                    +------+------+------+------+
        //               Rhs  |w28   |w29   |w30   |w31   |
        //                    +------+------+------+------+
        //
        //                    |      |      |      |      |
        //
        //       Lhs          |      |      |      |      |
        //
        //      +---+ - - - - +------+------+------+------+
        //      |w24|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //      |w24|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //      |w24|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //      |w24|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //      +---+ - - - - +------+------+------+------+
        //      |w25|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //      |w25|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //      |w25|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //      |w25|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //      +---+ - - - - +------+------+------+------+
        //      |w26|         |w8/20 |w9/21 |w10/22|w11/23|
        //      |w26|         |w8/20 |w9/21 |w10/22|w11/23|
        //      |w26|         |w8/20 |w9/21 |w10/22|w11/23|
        //      |w26|         |w8/20 |w9/21 |w10/22|w11/23|
        //      +---+ - - - - +------+------+------+------+
        //
        //                             Accumulators

        // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
        "ld.b   $w24, 0(%[lhs_ptr])\n"
        "ld.b   $w25, 8(%[lhs_ptr])\n"

        // Load 2 x 8 bytes of rhs[].
        "ld.b   $w27, 0(%[rhs_ptr])\n"

        // Zero-extend 8-bit elements of lhs[] to 16 bits.
        "ldi.b  $w31, 0\n"
        "ilvr.b $w24, $w31, $w24\n"
        "ilvl.b $w26, $w31, $w25\n"
        "ilvr.b $w25, $w31, $w25\n"

        // First half of depths 0 and 1.
        // Zero-extend 8-bit elements of rhs[] to 16 bits.
        "ilvr.b    $w31, $w31, $w27\n"
        // Make 4 replicas of every pair of rhs[] elements.
        "splati.w  $w28, $w31[0]\n"
        "splati.w  $w29, $w31[1]\n"
        "splati.w  $w30, $w31[2]\n"
        "splati.w  $w31, $w31[3]\n"
        // Dot-product-(and)-add doubles multiplicand width.
        "dpadd_u.w  $w0, $w24, $w28\n"
        "dpadd_u.w  $w4, $w25, $w28\n"
        "dpadd_u.w  $w8, $w26, $w28\n"
        "dpadd_u.w  $w1, $w24, $w29\n"
        "dpadd_u.w  $w5, $w25, $w29\n"
        "dpadd_u.w  $w9, $w26, $w29\n"
        "dpadd_u.w  $w2, $w24, $w30\n"
        "dpadd_u.w  $w6, $w25, $w30\n"
        "dpadd_u.w $w10, $w26, $w30\n"
        "dpadd_u.w  $w3, $w24, $w31\n"
        "dpadd_u.w  $w7, $w25, $w31\n"
        "dpadd_u.w $w11, $w26, $w31\n"

        // Second half of depths 0 and 1.
        // Zero-extend 8-bit elements of rhs[] to 16 bits.
        "ldi.b     $w31, 0\n"
        "ilvl.b    $w31, $w31, $w27\n"
        // Make 4 replicas of every pair of rhs[] elements.
        "splati.w  $w28, $w31[0]\n"
        "splati.w  $w29, $w31[1]\n"
        "splati.w  $w30, $w31[2]\n"
        "splati.w  $w31, $w31[3]\n"
        // Dot-product-(and)-add doubles multiplicand width.
        "dpadd_u.w $w12, $w24, $w28\n"
        "dpadd_u.w $w16, $w25, $w28\n"
        "dpadd_u.w $w20, $w26, $w28\n"
        "dpadd_u.w $w13, $w24, $w29\n"
        "dpadd_u.w $w17, $w25, $w29\n"
        "dpadd_u.w $w21, $w26, $w29\n"
        "dpadd_u.w $w14, $w24, $w30\n"
        "dpadd_u.w $w18, $w25, $w30\n"
        "dpadd_u.w $w22, $w26, $w30\n"
        "dpadd_u.w $w15, $w24, $w31\n"
        "dpadd_u.w $w19, $w25, $w31\n"
        "dpadd_u.w $w23, $w26, $w31\n"

        GEMMLOWP_MIPS_XADDIU " %[run_depth], -2\n" GEMMLOWP_MIPS_XADDIU
        " %[lhs_ptr], 24\n" GEMMLOWP_MIPS_XADDIU " %[rhs_ptr], 16\n"
        "bnez   %[run_depth]," GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP ":\n"

        // Store accumulators.
        GEMMLOWP_MIPS_XADDU " $a0, %[dst_ptr], %[dst_col_stride]\n"
        "st.w   $w0,  (0*16)(%[dst_ptr])\n"
        "st.w   $w4,  (1*16)(%[dst_ptr])\n"
        "st.w   $w8,  (2*16)(%[dst_ptr])\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "st.w   $w1,  (0*16)($a0)\n"
        "st.w   $w5,  (1*16)($a0)\n"
        "st.w   $w9,  (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "st.w   $w2,  (0*16)($a1)\n"
        "st.w   $w6,  (1*16)($a1)\n"
        "st.w   $w10, (2*16)($a1)\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "st.w   $w3,  (0*16)($a0)\n"
        "st.w   $w7,  (1*16)($a0)\n"
        "st.w   $w11, (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "st.w   $w12, (0*16)($a1)\n"
        "st.w   $w16, (1*16)($a1)\n"
        "st.w   $w20, (2*16)($a1)\n" GEMMLOWP_MIPS_XADDU " $a1, $a0, %[dst_col_stride]\n"
        "st.w   $w13, (0*16)($a0)\n"
        "st.w   $w17, (1*16)($a0)\n"
        "st.w   $w21, (2*16)($a0)\n" GEMMLOWP_MIPS_XADDU " $a0, $a1, %[dst_col_stride]\n"
        "st.w   $w14, (0*16)($a1)\n"
        "st.w   $w18, (1*16)($a1)\n"
        "st.w   $w22, (2*16)($a1)\n"
        "st.w   $w15, (0*16)($a0)\n"
        "st.w   $w19, (1*16)($a0)\n"
        "st.w   $w23, (2*16)($a0)\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr), [run_depth] "+r"(run_depth),
        [dst_col_stride] "+r"(dst_col_stride)
        :  // inputs
        [dst_ptr] "r"(dst_ptr),
        [start_depth] "r"(start_depth)
        :  // clobbers
        "memory", "a0", "a1", "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7", "$f8", "$f9",
        "$f10", "$f11", "$f12", "$f13", "$f14", "$f15", "$f16", "$f17", "$f18", "$f19", "$f20",
        "$f21", "$f22", "$f23", "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31");

#undef GEMMLOWP_LABEL_CLEAR_ACCUMULATORS
#undef GEMMLOWP_LABEL_BEFORE_LOOP
#undef GEMMLOWP_LABEL_LOOP
#undef GEMMLOWP_LABEL_AFTER_LOOP
  }
};

#undef GEMMLOWP_MIPS_XADDU
#undef GEMMLOWP_MIPS_XADDIU
#undef GEMMLOWP_MIPS_XSLL

#endif  // GEMMLOWP_MSA

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_MSA_H_
