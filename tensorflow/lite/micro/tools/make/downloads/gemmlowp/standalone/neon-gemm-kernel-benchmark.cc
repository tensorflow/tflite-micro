// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
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

// This is a standalone testbed and benchmark for gemmlowp-style GEMM kernels,
// either doing integer or float arithmetic.
// It verifies that a kernel produces correct results, then benchmarks it.
//
// Some benchmark results are recorded in this spreadsheet:
//
// https://docs.google.com/spreadsheets/d/1UPbzbp9rdsD6RXxOr5q6AZ0n1omgEknLYO2ogiw6Kqk/edit?usp=sharing
//
// This program is entirely self-contained, and can be compiled manually
// such as suggested in the command lines below.
// It currently supports only Android/ARM but would trivially generalize to
// other OSes (it's mostly standard POSIX) or architectures (each kernel
// targets a specific architecture, one may simply add more).

/*
 Build and run this benchmark on Android/ARM/32bit:
 ~/android/toolchains/arm-linux-androideabi/bin/arm-linux-androideabi-clang++ \
 -fPIE -pie -O3 --std=c++11 standalone/neon-gemm-kernel-benchmark.cc -o \
 /tmp/benchmark -mfloat-abi=softfp -mfpu=neon-vfpv4 && adb push /tmp/benchmark \
 /data/local/tmp && adb shell /data/local/tmp/benchmark
 Build and run this benchmark on Android/ARM/64bit:
 ~/android/toolchains/aarch64-linux-android/bin/aarch64-linux-android-clang++ \
 -fPIE -static -O3 --std=c++11 standalone/neon-gemm-kernel-benchmark.cc -o \
 /tmp/benchmark && adb push /tmp/benchmark /data/local/tmp && adb shell \
 /data/local/tmp/benchmark
 */

// For big.LITTLE devices, use 'taskset' to select which cores to benchmark.
//
// The syntax is: taskset <mask> <commandline>
// where mask is a binary mask where each bit corresponds to a core,
// and low bits are little cores.
//
// Examples:
// Nexus 5X big cores: taskset 30
// Nexus 5X little cores: taskset 0f
// Pixel XL big cores: taskset 0c
// Pixel XL little cores: taskset 03
//
// Full example:
// adb shell taskset 0c /data/local/tmp/benchmark

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <type_traits>

#if !defined(__arm__) && !defined(__aarch64__) && \
  !(defined(__mips) && (__mips_isa_rev >= 5) && defined(__mips_msa))
#error This benchmark assumes ARM or MIPS (for intrinsics and inline assembly sections).
#endif

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__mips)
#include <msa.h>

// Some convenience macros to hide differences between MIPS32 and MIPS64.
#ifdef __LP64__
#define GEMMLOWP_MIPS_XADDIU     "daddiu"
#else
#define GEMMLOWP_MIPS_XADDIU     "addiu"
#endif
#endif

// Typically one wants to fit in L1 cache, and GEMM implementations
// are carefully optimized to tune their access patterns to that effect.
// Most devices have at least 16k of L1 cache. The Kraits have exactly 16k.
const int kDefaultCacheSizeK = 16;

const int kCacheLineSize = 64;

// These definitions are used for labels within assembly code. Required for
// iOS toolchain compatibility.
#define GEMMLOWP_LABEL_AFTER_LOOP "1"
#define GEMMLOWP_LABEL_LOOP "2"
#define GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES "3"
#define GEMMLOWP_LABEL_STORE "4"

// BEGIN code copied from gemmlowp/internal/kernel.h

// Explanation of general gemmlowp terminology
// ===========================================
//
// We use the following abbreviations:
// LHS = "left-hand side"
// RHS = "right-hand side"
// Sometimes when referring to either LHS or RHS, we just say a "Side".
//
// In a matrix product of a MxK matrix times a KxN matrix,
// we call K the 'depth'. Note that M is the number of rows
// of the result (and of the LHS), and N is the number of columns
// of the result (and of the RHS).
//
// In each of the LHS and RHS matrices, we call 'width' the
// other dimension, besides the depth. So in the LHS, 'width'
// is the number of rows, while in the RHS, 'width' is the number
// of columns.
//
//  So in the LHS MxK matrix, the depth is K and the width in M.
// And in the RHS KxN matrix, the depth is K and the width in N.
//
// This is illustrated in this picture:
//
//                             RHS width
//                        <----------------->
//                        +-----------------+ ^
//                        |       RHS       | | Depth
//                        +-----------------+ v
//                 ^ +--+ +-----------------+
//                 | |L | |                 |
//       LHS width | |H | |      Result     |
//                 | |S | |                 |
//                 v +--+ +-----------------+
//                   <-->
//                   Depth

// Explanation of gemmlowp kernel formats and "cells"
// ==================================================
//
// Kernels operate on small LHS and RHS blocks that fit in registers.
// These blocks are stored contiguously in memory, but not always
// in a traditional column-major or row-major order; instead,
// they consist of a number of sub-blocks, which we call "cells",
// that are stored in column-major or row-major order. However,
// what really matters to us is not so much rows vs columns, but
// rather width vs depth. So we refer to "width-major" and "depth-major"
// storage orders. In the LHS, width-major means row-major,
// while in the RHS, width-major means column-major.
// There is also a third possibility, "diagonal order",
// which is unused at the moment.
//
// We aim to treat both sides, LHS and RHS, on an equal footing,
// so we call them both 'sides'. A KernelFormat thus is just a pair
// of KernelSideFormat's, one for LHS and one for RHS; each KernelSideFormat
// contains a CellFormat and a number of cells; cells are only ever
// stacked in the width dimension, which means stacked vertically in the
// LHS and stacked horizondally in the RHS.
//
// Example
// =======
//
// Let's work out the data layout expected by a kernel having the
// following format (the struct names here are defined below in this file):
//
// KernelFormat<
//   KernelSideFormat<CellFormat<3, 4>, 3>,
//   KernelSideFormat<CellFormat<5, 4>, 2>
// >
//
// The LHS format, KernelSideFormat<CellFormat<3, 4>, 3>, means:
// 3 cells, each cell having dimensions (width=3, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the LHS,
// DepthMajor means column-major, so the LHS cells are of size 3x4 in
// column-major order, so the LHS layout is:
//
// 0  3  6  9
// 1  4  7  10
// 2  5  8  11
// 12 15 18 21
// 13 16 19 22
// 14 17 20 23
// 24 27 30 33
// 25 28 31 34
// 26 29 32 35
//
// The RHS format, KernelSideFormat<CellFormat<5, 4>, 2>, means:
// 2 cells each having dimensions (width=5, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the RHS,
// DepthMajor means row-major, so the RHS cells are of size 4x5 in
// row-major order, so the RHS layout is:
//
// 0  1  2  3  4  20 21 22 23 24
// 5  6  7  8  9  25 26 27 28 29
// 10 11 12 13 14 30 31 32 33 34
// 15 16 17 18 19 35 36 37 38 39

// CellOrder enumerates the possible storage orders (=layouts) for
// a cell (see explanation above).
enum class CellOrder { DepthMajor, WidthMajor, Diagonal };

// CellFormat describes how data is laid
// out in a cell. That is, a CellOrder together with actual dimensions.
template <int tWidth, int tDepth, CellOrder tOrder>
struct CellFormat {
  static const int kWidth = tWidth;
  static const int kDepth = tDepth;
  static const CellOrder kOrder = tOrder;

  static const int kSize = kWidth * kDepth;
};

// KernelSideFormat describes how data is laid out in a kernel side
// (i.e. LHS or RHS). That is, a CellFormat together with a number of
// cells. These cells are always stacked in the Width dimension.
// For example, in the LHS case, the Width dimension is the rows dimension,
// se we're saying that in the LHS, cells are stacked vertically.
// We never stack cells in the Depth dimension.
template <typename tCellFormat, int tCells>
struct KernelSideFormat {
  typedef tCellFormat Cell;
  static const int kCells = tCells;
  static const int kWidth = kCells * Cell::kWidth;
  static const int kDepth = Cell::kDepth;
};

// KernelFormat describes fully the input data layout that a kernel expects.
// It consists of two KernelSideFormat's, one for LHS and one for RHS.
template <typename tLhs, typename tRhs>
struct KernelFormat {
  typedef tLhs Lhs;
  typedef tRhs Rhs;

  static_assert(Lhs::Cell::kDepth == Rhs::Cell::kDepth, "");
  static const int kDepth = Lhs::Cell::kDepth;
  static const int kRows = Lhs::Cell::kWidth * Lhs::kCells;
  static const int kCols = Rhs::Cell::kWidth * Rhs::kCells;
};

// KernelOperandRanges specifies the minimum and maximum values an operand can
// take. It consists of two ranges: one for the LHS and one for the RHS. The
// default values are the minimum and maximum values of the operand data type.
template <typename Kernel, typename OperandType = typename Kernel::OperandType>
struct KernelOperandRanges {
  static OperandType LhsMin() {
    return std::numeric_limits<OperandType>::lowest();
  }
  static OperandType LhsMax() {
    return std::numeric_limits<OperandType>::max();
  }
  static OperandType RhsMin() {
    return std::numeric_limits<OperandType>::lowest();
  }
  static OperandType RhsMax() {
    return std::numeric_limits<OperandType>::max();
  }
};

template <typename Kernel>
struct KernelOperandRanges<Kernel, float> {
  static float LhsMin() { return -100.f; }
  static float LhsMax() { return 100.f; }
  static float RhsMin() { return -100.f; }
  static float RhsMax() { return 100.f; }
};

#define SET_7BIT_RANGES(kernel) \
template <> \
struct KernelOperandRanges<kernel, std::int8_t> { \
  static std::int8_t LhsMin() { return -63; } \
  static std::int8_t LhsMax() { return 63; } \
  static std::int8_t RhsMin() { return -64; } \
  static std::int8_t RhsMax() { return 63; } \
};

#define SET_425BIT_RANGES(kernel) \
template <> \
struct KernelOperandRanges<kernel, std::int8_t> { \
  static std::int8_t LhsMin() { return -7; } \
  static std::int8_t LhsMax() { return 7; } \
  static std::int8_t RhsMin() { return -9; } \
  static std::int8_t RhsMax() { return 9; } \
};

inline const char* CellOrderName(CellOrder o) {
  switch (o) {
    case CellOrder::DepthMajor:
      return "DepthMajor";
    case CellOrder::WidthMajor:
      return "WidthMajor";
    case CellOrder::Diagonal:
      return "Diagonal";
    default:
      assert(false);
      return nullptr;
  }
}

// Returns the offset into a cell, at which a given coefficient is stored.
template <typename CellFormat>
inline int OffsetIntoCell(int w, int d) {
  switch (CellFormat::kOrder) {
    case CellOrder::DepthMajor:
      return w + d * CellFormat::kWidth;
    case CellOrder::WidthMajor:
      return d + w * CellFormat::kDepth;
    case CellOrder::Diagonal:
      assert(CellFormat::kWidth == CellFormat::kDepth);
      static const int size = CellFormat::kWidth;
      return ((size + w - d) * size + d) % (size * size);
    default:
      assert(false);
      return 0;
  }
}

// END code copied from gemmlowp/internal/kernel.h

#ifdef __arm__

// This is the current standard kernel in gemmlowp, see:
// https://github.com/google/gemmlowp/blob/b1e2a29ff866680028f3080efc244e10e8dd7f46/internal/kernel_neon.h#L33
struct NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load 1 Rhs cell of size 2x4
        "vld1.8 {d0}, [%[rhs_ptr]]!\n"
        // Load 3 Lhs cells of size 4x2 each
        "vld1.8 {d2}, [%[lhs_ptr]]!\n"
        "vld1.8 {d4}, [%[lhs_ptr]]!\n"
        "vld1.8 {d6}, [%[lhs_ptr]]!\n"
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        "subs %[depth], #2\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"
        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in d0--d1 (q0).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in 32bit in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //                   |d0[0]|d0[1]|d0[2]|d0[3]|
        //              Rhs  +-----+-----+-----+-----+
        //                   |d1[0]|d1[1]|d1[2]|d1[3]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //    Lhs            |     |     |     |     |
        //
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  |d2|d3|          | q4  | q5  | q6  | q7  |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  |d4|d5|          | q8  | q9  | q10 | q11 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  |d6|d7|          | q12 | q13 | q14 | q15 |
        //  +--+--+ - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Expand Lhs/Rhs cells to 16 bit.
        // Note: moving theses vmovls further down to allow for
        // longer data pipelining helps a little on A57 but is
        // harmful on A53 --- It looks as if A53 doesn't like
        // interleaving vmovl's into the vmlal's.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vldr d2, [%[lhs_ptr]]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vldr d4, [%[lhs_ptr], #8]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vldr d0, [%[rhs_ptr]]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "add %[lhs_ptr], #24\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "add %[rhs_ptr], #8\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "subs %[depth], #2\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "vmovl.u8 q0, d0\n"
        "vmovl.u8 q1, d2\n"
        "vmovl.u8 q2, d4\n"
        "vmovl.u8 q3, d6\n"

        // Multiply-accumulate, level of depth 0
        "vmlal.u16 q4, d2, d0[0]\n"
        "vmlal.u16 q5, d2, d0[1]\n"
        "vmlal.u16 q6, d2, d0[2]\n"
        "vmlal.u16 q7, d2, d0[3]\n"
        "vmlal.u16 q8, d4, d0[0]\n"
        "vmlal.u16 q9, d4, d0[1]\n"
        "vmlal.u16 q10, d4, d0[2]\n"
        "vmlal.u16 q11, d4, d0[3]\n"
        "vmlal.u16 q12, d6, d0[0]\n"
        "vmlal.u16 q13, d6, d0[1]\n"
        "vmlal.u16 q14, d6, d0[2]\n"
        "vmlal.u16 q15, d6, d0[3]\n"

        // Multiply-accumulate, level of depth 1
        "vmlal.u16 q4, d3, d1[0]\n"
        "vmlal.u16 q5, d3, d1[1]\n"
        "vmlal.u16 q6, d3, d1[2]\n"
        "vmlal.u16 q7, d3, d1[3]\n"
        "vmlal.u16 q8, d5, d1[0]\n"
        "vmlal.u16 q9, d5, d1[1]\n"
        "vmlal.u16 q10, d5, d1[2]\n"
        "vmlal.u16 q11, d5, d1[3]\n"
        "vmlal.u16 q12, d7, d1[0]\n"
        "vmlal.u16 q13, d7, d1[1]\n"
        "vmlal.u16 q14, d7, d1[2]\n"
        "vmlal.u16 q15, d7, d1[3]\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// This is Maciek Chociej's fast kernel not expanding operands,
// from gemmlowp/meta/. Search for
//      mul_3x8_3x8_int32_lhsadd_rhsadd
// in this file:
// https://raw.githubusercontent.com/google/gemmlowp/e4b9d858b6637d5d0058bfa3d869d2b95864251b/meta/single_thread_gemm.h
struct NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<3, 8, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<3, 8, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Clear aggregators.
        "vmov.i32 q0, #0\n"
        "vmov.i32 q1, #0\n"
        "vmov.i32 q2, #0\n"
        "vmov.i32 q3, q0\n"
        "vmov.i32 q4, q1\n"
        "vmov.i32 q5, q2\n"
        "vmov.i32 q6, q3\n"
        "vmov.i32 q7, q4\n"
        "vmov.i32 q8, q5\n"

        // Loop head
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Subtract counter.
        "subs %[depth], %[depth], #8\n"

        "vld1.8 {d18, d19, d20}, [%[rhs_ptr]]!\n"
        "vld1.8 {d21, d22, d23}, [%[lhs_ptr]]!\n"
        "vmull.u8 q12, d18, d21\n"
        "vmull.u8 q13, d18, d22\n"
        "vmull.u8 q14, d18, d23\n"
        "vmull.u8 q15, d19, d21\n"
        "vpadal.u16 q0, q12\n"
        "vpadal.u16 q1, q13\n"
        "vpadal.u16 q2, q14\n"
        "vpadal.u16 q3, q15\n"
        "vmull.u8 q12, d19, d22\n"
        "vmull.u8 q13, d19, d23\n"
        "vmull.u8 q14, d20, d21\n"
        "vmull.u8 q15, d20, d22\n"
        "vmull.u8 q9, d20, d23\n"
        "vpadal.u16 q4, q12\n"
        "vpadal.u16 q5, q13\n"
        "vpadal.u16 q6, q14\n"
        "vpadal.u16 q7, q15\n"
        "vpadal.u16 q8, q9\n"

        // Loop branch
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Horizontal reduce aggregators, step 1
        "vpadd.u32 d0, d0, d1\n"
        "vpadd.u32 d2, d2, d3\n"
        "vpadd.u32 d4, d4, d5\n"
        "vpadd.u32 d6, d6, d7\n"
        "vpadd.u32 d8, d8, d9\n"
        "vpadd.u32 d10, d10, d11\n"
        "vpadd.u32 d12, d12, d13\n"
        "vpadd.u32 d14, d14, d15\n"
        "vpadd.u32 d16, d16, d17\n"

        // Horizontal reduce aggregators, step 2
        "vpadd.u32 d0, d0, d2\n"
        "vpadd.u32 d1, d4, d4\n"
        "vpadd.u32 d6, d6, d8\n"
        "vpadd.u32 d7, d10, d10\n"
        "vpadd.u32 d12, d12, d14\n"
        "vpadd.u32 d13, d16, d16\n"

        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d2}, [r0]!\n"
        "vld1.32 {d3[0]}, [r0]!\n"

        "vld1.32 {d8}, [r0]!\n"
        "vld1.32 {d9[0]}, [r0]!\n"

        "vld1.32 {d14}, [r0]!\n"
        "vld1.32 {d15[0]}, [r0]!\n"

        // Accumulate
        "vadd.s32 q0, q0, q1\n"
        "vadd.s32 q3, q3, q4\n"
        "vadd.s32 q6, q6, q7\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d0}, [r0]!\n"
        "vst1.32 {d1[0]}, [r0]!\n"

        "vst1.32 {d6}, [r0]!\n"
        "vst1.32 {d7[0]}, [r0]!\n"

        "vst1.32 {d12}, [r0]!\n"
        "vst1.32 {d13[0]}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// Fast kernel operating on int8 operands.
// It is assumed that one of the two int8 operands only takes values
// in [-127, 127], while the other may freely range in [-128, 127].
// The issue with both operands taking the value -128 is that:
// -128*-128 + -128*-128 == -32768 overflows int16.
// Every other expression a*b + c*d, for any int8 a,b,c,d, fits in int16
// range. That is the basic idea of this kernel.
struct NEON_32bit_GEMM_Int8Operands_AccumTwoWithin16Bits {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<2, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    std::size_t start_depth = 123;
    std::size_t run_depth = depth;
    std::size_t dst_col_stride = 4;
    AccumulatorType* dst_ptr = accum_ptr;
    asm volatile(

        // Overview of register layout:
        //
        // A 2x16 block of Rhs is stored in 8 bit in d0--d3.
        // A 4x16 block of Lhs is stored in 8 bit in d4--d7. That is only
        // half of the register space required, so we loop over these registers
        // twice. Only half of it, a 2x16 block, is stored in d4--d7 at
        // any given time.
        //
        // A 4x2 block of accumulators is stored in q8--q15 (as 4x32 bit
        // components which need to be horizontally-added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        // Here comes the special trick: since the operands are signed int8,
        // their range being [ -2^7 , 2^7 ), their products are in range
        // [ -2^14 , 2^14 - 1 ), meaning that we can add two such values
        // without any risk of overflowing int16.
        // We thus proceed with the 8 next levels of depth, multiplying
        // again Lhs by Rhs, accumulating into this existing int16x8 vector.
        //
        // Only then, having processed 16 levels of depth, do we need to
        // horizontally add these int16x8 accumulators into the final
        // int32x4 accumulators.
        //
        // As we do not have enough registers to store all 16 int16x8
        // temporary-16bit-accumulators, we have them cycle through q4--q7.
        //
        //
        // Register layout (ignoring the q4--q7 temporary 16bit accumulators):
        //
        //                               +----+----+
        //                               | d0 | d2 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                       Rhs     +----+----+
        //                               | d1 | d3 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               +----+----+
        //
        //                               |    |    |
        //
        //    Lhs                        |    |    |
        //
        //  +--------+--------+ - - - -  +----+----+
        //  | d4 ... | d5 ... |          | q8 | q9 |
        //  | d6 ... | d7 ... |          | q10| q11|
        //  | d4 ... | d5 ... |          | q12| q13|
        //  | d6 ... | d7 ... |          | q14| q15|
        //  +--------+--------+ - - - -  +----+----+
        //
        //                               Accumulator
        //

        // Clear accumulators, and, interleaved with it,
        // initial loads of the first loop iteration,
        // taken out of the loop so that in the loop itself we have
        // optimal streaming of data from memory.
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmov.i32 q8, #0\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmov.i32 q9, #0\n"
        "vldr d2, [%[rhs_ptr], #16]\n"
        "vmov.i32 q10, q8\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmov.i32 q11, q8\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmov.i32 q12, q8\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmov.i32 q13, q8\n"
        "vldr d3, [%[rhs_ptr], #24]\n"
        "vmov.i32 q14, q8\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vmov.i32 q15, q8\n"

        // General loop.
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Multiply 8 first levels of depth.
        "vmull.s8    q4,  d0,  d4\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #32]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d6, [%[lhs_ptr], #48]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #40]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d7, [%[lhs_ptr], #56]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q8,  q4\n"
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "vpadal.s16   q9,  q5\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "vpadal.s16   q10, q6\n"
        "vpadal.s16   q11, q7\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP
        "f\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d2, [%[rhs_ptr], #16]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d3, [%[rhs_ptr], #24]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"

        "b " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"
        "cmp %[start_depth], #0\n"

        // Reduce 32bit accumulators horizontally.
        "vpadd.s32 d0, d16, d17\n"
        "vpadd.s32 d1, d18, d19\n"
        "vpadd.s32 d2, d20, d21\n"
        "vpadd.s32 d3, d22, d23\n"
        "vpadd.s32 d4, d24, d25\n"
        "vpadd.s32 d5, d26, d27\n"
        "vpadd.s32 d6, d28, d29\n"
        "vpadd.s32 d7, d30, d31\n"

        "bne " GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        "f\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        "b " GEMMLOWP_LABEL_STORE "f\n"

        GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        ":\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise),
        // and load destination values from memory.
        "mov r0, %[dst_ptr]\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vld1.32 {d18, d19}, [r0]\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        // Add horizontally-reduced accumulators into
        // the values loaded from memory
        "vadd.s32 q4, q8, q4\n"
        "vadd.s32 q5, q9, q5\n"

        GEMMLOWP_LABEL_STORE
        ":\n"
        // Store back into memory
        "mov r0, %[dst_ptr]\n"
        "vst1.32 {d8, d9}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// We don't actually use int32*int32 in production. This is just an
// experiment to help dissociate the effect of integer-vs-float, from the
// effect of operands width.
struct NEON_32bit_GEMM_Int32_WithScalar {
  typedef std::int32_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 1 Rhs cell of size 1x4
        "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vmla.s32 q4, q1, d0[0]\n"
        "vmla.s32 q5, q1, d0[1]\n"
        "vmla.s32 q6, q1, d1[0]\n"
        "vmla.s32 q7, q1, d1[1]\n"
        "vmla.s32 q8, q2, d0[0]\n"
        "vmla.s32 q9, q2, d0[1]\n"
        "vmla.s32 q10, q2, d1[0]\n"
        "vmla.s32 q11, q2, d1[1]\n"
        "vmla.s32 q12, q3, d0[0]\n"
        "vmla.s32 q13, q3, d0[1]\n"
        "vmla.s32 q14, q3, d1[0]\n"
        "vmla.s32 q15, q3, d1[1]\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
struct NEON_32bit_GEMM_Float32_MLA_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vmla.f32 q4, q1, q0\n"
        "vmla.f32 q8, q2, q0\n"
        "vmla.f32 q12, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vmla.f32 q5, q1, q0\n"
        "vmla.f32 q9, q2, q0\n"
        "vmla.f32 q13, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vmla.f32 q6, q1, q0\n"
        "vmla.f32 q10, q2, q0\n"
        "vmla.f32 q14, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vmla.f32 q7, q1, q0\n"
        "vmla.f32 q11, q2, q0\n"
        "vmla.f32 q15, q3, q0\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
// This variant is relevant as on ARMv7 FMA does not have a with-scalar variant.
struct NEON_32bit_GEMM_Float32_FMA_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vfma.f32 q4, q1, q0\n"
        "vfma.f32 q8, q2, q0\n"
        "vfma.f32 q12, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vfma.f32 q5, q1, q0\n"
        "vfma.f32 q9, q2, q0\n"
        "vfma.f32 q13, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vfma.f32 q6, q1, q0\n"
        "vfma.f32 q10, q2, q0\n"
        "vfma.f32 q14, q3, q0\n"
        "vld1.32 {d0[], d1[]}, [%[rhs_ptr]]!\n"
        "vfma.f32 q7, q1, q0\n"
        "vfma.f32 q11, q2, q0\n"
        "vfma.f32 q15, q3, q0\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// This is the "most natural" kernel, using NEON multiply-with-scalar
// instructions.
struct NEON_32bit_GEMM_Float32_MLA_WithScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 1 Rhs cell of size 1x4
        "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vmla.f32 q4, q1, d0[0]\n"
        "vmla.f32 q5, q1, d0[1]\n"
        "vmla.f32 q6, q1, d1[0]\n"
        "vmla.f32 q7, q1, d1[1]\n"
        "vmla.f32 q8, q2, d0[0]\n"
        "vmla.f32 q9, q2, d0[1]\n"
        "vmla.f32 q10, q2, d1[0]\n"
        "vmla.f32 q11, q2, d1[1]\n"
        "vmla.f32 q12, q3, d0[0]\n"
        "vmla.f32 q13, q3, d0[1]\n"
        "vmla.f32 q14, q3, d1[0]\n"
        "vmla.f32 q15, q3, d1[1]\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// Faster kernel contributed by ARM in 64bit form
// (see NEON_64bit_GEMM_Float32_WithScalar_A53) then ported to 32bit code.
// Tuned for A53.
struct NEON_32bit_GEMM_Float32_WithScalar_A53 {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        // Overview of register layout:
        //
        // A 1x4 cell of Rhs is stored in d0--d1 (q0).
        // A 12x1 block of 3 4x1 cells Lhs is stored in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //             Rhs   |d0[0]|d0[1]|d1[0]|d1[1]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //  Lhs              |     |     |     |     |
        //
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //  |d2|             | q4  | q5  | q6  | q7  |
        //  |d2|             | q4  | q5  | q6  | q7  |
        //  |d3|             | q4  | q5  | q6  | q7  |
        //  |d3|             | q4  | q5  | q6  | q7  |
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //  |d4|             | q8  | q9  | q10 | q11 |
        //  |d4|             | q8  | q9  | q10 | q11 |
        //  |d5|             | q8  | q9  | q10 | q11 |
        //  |d5|             | q8  | q9  | q10 | q11 |
        //  +--+ - - - - - - +-----+-----+-----+-----+
        //  |d6|             | q12 | q13 | q14 | q15 |
        //  |d6|             | q12 | q13 | q14 | q15 |
        //  |d7|             | q12 | q13 | q14 | q15 |
        //  |d7|             | q12 | q13 | q14 | q15 |
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load Rhs cell
        "vldr d0, [%[rhs_ptr]]\n"
        "ldr r2, [%[rhs_ptr], #8]\n"
        "ldr r3, [%[rhs_ptr], #12]\n"

        // Load 1st Lhs Cell
        "vld1.32 {d2, d3}, [%[lhs_ptr]]\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        "vldr d4, [%[lhs_ptr], #16]\n"  // Load 1st half of 2nd Lhs cell
        "vmov d1, r2, r3\n"             // Prepare 2nd half of Rhs cell
        "vmla.f32 q4, q1, d0[0]\n"      // Multiply 1st Lhs cell with column 0
        "ldr r2, [%[lhs_ptr], #24]\n"   // Load 2nd half of 2nd Lhs cell, part 1
        "vmla.f32 q5, q1, d0[1]\n"      // Multiply 1st Lhs cell with column 1
        "ldr r3, [%[lhs_ptr], #28]\n"   // Load 2nd half of 2nd Lhs cell, part 2
        "vmla.f32 q6, q1, d1[0]\n"      // Multiply 1st Lhs cell with column 2
        "subs %[depth], #1\n"

        "vldr d6, [%[lhs_ptr], #32]\n"  // Load 1st half of 3rd Lhs cell
        "vmov d5, r2, r3\n"             // Prepare 2nd half of 2nd Lhs cell
        "vmla.f32 q7, q1, d1[1]\n"      // Multiply 1st Lhs cell with column 3
        "ldr r2, [%[lhs_ptr], #40]\n"   // Load 2nd half of 3rd Lhs cell, part 1
        "vmla.f32 q8, q2, d0[0]\n"      // Multiply 2nd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #44]\n"   // Load 2nd half of 3rd Lhs cell, part 2
        "vmla.f32 q9, q2, d0[1]\n"      // Multiply 2nd Lhs cell with column 1
        "add %[rhs_ptr], %[rhs_ptr], #16\n"  // Move forward by 1 Rhs cell

        "vldr d2, [%[lhs_ptr], #48]\n"  // Load 1st half of 1st Lhs cell of next
        // iteration
        "vmov d7, r2, r3\n"            // Prepare 2nd half of 3rd Lhs cell
        "vmla.f32 q10, q2, d1[0]\n"    // Multiply 2nd Lhs cell with column 2
        "ldr r2, [%[lhs_ptr], #56]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 1
        "vmla.f32 q12, q3, d0[0]\n"    // Multiply 3rd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #60]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 2
        "vmla.f32 q13, q3, d0[1]\n"  // Multiply 3rd Lhs cell with column 1
        "add %[lhs_ptr], %[lhs_ptr], #48\n"  // Move forward by 3 Lhs cells

        "vldr d0, [%[rhs_ptr]]\n"  // Load 1st half of Rhs cell of next
        // iteration
        "vmov d3, r2, r3\n"  // Prepare 2nd half of 1st Lhs cell of next
        // iteration
        "vmla.f32 q11, q2, d1[1]\n"   // Multiply 2nd Lhs cell with column 3
        "ldr r2, [%[rhs_ptr], #8]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 1
        "vmla.f32 q14, q3, d1[0]\n"    // Multiply 3rd Lhs cell with column 2
        "ldr r3, [%[rhs_ptr], #12]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 2
        "vmla.f32 q15, q3, d1[1]\n"  // Multiply 3rd Lhs cell with column 3

        // Loop branch.  This will dual issue in fmla cycle 3 of the 4th block.
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "d28", "d29", "d30", "d31");
  }
};

struct NEON_32bit_GEMM_Float32_WithScalar_A53_depth2 {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        // Overview of register layout:
        //
        // A 1x4 cell of Rhs is stored in d0--d1 (q0).
        // A 12x1 block of 3 4x1 cells Lhs is stored in d2--d7
        // (q1--q3).
        // A 12x4 block of accumulators is stored in q4--q15.
        //
        //                   +-----+-----+-----+-----+
        //             Rhs   |d0[0]|d0[1]|d1[0]|d1[1]|
        //                   +-----+-----+-----+-----+
        //
        //                   |     |     |     |     |
        //
        //  Lhs              |     |     |     |     |
        //
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //  |d2|             | q4  | q5  | q6  | q7  |
        //  |d2|             | q4  | q5  | q6  | q7  |
        //  |d3|             | q4  | q5  | q6  | q7  |
        //  |d3|             | q4  | q5  | q6  | q7  |
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //  |d4|             | q8  | q9  | q10 | q11 |
        //  |d4|             | q8  | q9  | q10 | q11 |
        //  |d5|             | q8  | q9  | q10 | q11 |
        //  |d5|             | q8  | q9  | q10 | q11 |
        //  +--+ - - - - - - +-----+-----+-----+-----+
        //  |d6|             | q12 | q13 | q14 | q15 |
        //  |d6|             | q12 | q13 | q14 | q15 |
        //  |d7|             | q12 | q13 | q14 | q15 |
        //  |d7|             | q12 | q13 | q14 | q15 |
        //  +--+- - - - - -  +-----+-----+-----+-----+
        //
        //                            Accumulator

        // Load Rhs cell
        "vldr d0, [%[rhs_ptr]]\n"
        "ldr r2, [%[rhs_ptr], #8]\n"
        "ldr r3, [%[rhs_ptr], #12]\n"

        // Load 1st Lhs Cell
        "vld1.32 {d2, d3}, [%[lhs_ptr]]\n"

        // Loop head - handling 2 levels of depth at once
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Level of depth 1

        "vldr d4, [%[lhs_ptr], #32]\n"  // Load 1st half of 2nd Lhs cell
        "vmov d1, r2, r3\n"             // Prepare 2nd half of Rhs cell
        "vmla.f32 q4, q1, d0[0]\n"      // Multiply 1st Lhs cell with column 0
        "ldr r2, [%[lhs_ptr], #40]\n"   // Load 2nd half of 2nd Lhs cell, part 1
        "vmla.f32 q5, q1, d0[1]\n"      // Multiply 1st Lhs cell with column 1
        "ldr r3, [%[lhs_ptr], #44]\n"   // Load 2nd half of 2nd Lhs cell, part 2
        "vmla.f32 q6, q1, d1[0]\n"      // Multiply 1st Lhs cell with column 2

        "vldr d6, [%[lhs_ptr], #64]\n"  // Load 1st half of 3rd Lhs cell
        "vmov d5, r2, r3\n"             // Prepare 2nd half of 2nd Lhs cell
        "vmla.f32 q7, q1, d1[1]\n"      // Multiply 1st Lhs cell with column 3
        "ldr r2, [%[lhs_ptr], #72]\n"   // Load 2nd half of 3rd Lhs cell, part 1
        "vmla.f32 q8, q2, d0[0]\n"      // Multiply 2nd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #76]\n"   // Load 2nd half of 3rd Lhs cell, part 2
        "vmla.f32 q9, q2, d0[1]\n"      // Multiply 2nd Lhs cell with column 1

        "vldr d2, [%[lhs_ptr], #16]\n"  // Load 1st half of 1st Lhs cell of next
        // iteration
        "vmov d7, r2, r3\n"            // Prepare 2nd half of 3rd Lhs cell
        "vmla.f32 q10, q2, d1[0]\n"    // Multiply 2nd Lhs cell with column 2
        "ldr r2, [%[lhs_ptr], #24]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 1
        "vmla.f32 q12, q3, d0[0]\n"    // Multiply 3rd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #28]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 2
        "vmla.f32 q13, q3, d0[1]\n"  // Multiply 3rd Lhs cell with column 1

        "vldr d0, [%[rhs_ptr], #16]\n"  // Load 1st half of Rhs cell of next
        // iteration
        "vmov d3, r2, r3\n"  // Prepare 2nd half of 1st Lhs cell of next
        // iteration
        "vmla.f32 q11, q2, d1[1]\n"    // Multiply 2nd Lhs cell with column 3
        "ldr r2, [%[rhs_ptr], #24]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 1
        "vmla.f32 q14, q3, d1[0]\n"    // Multiply 3rd Lhs cell with column 2
        "ldr r3, [%[rhs_ptr], #28]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 2
        "vmla.f32 q15, q3, d1[1]\n"  // Multiply 3rd Lhs cell with column 3

        // Level of depth 2
        "vldr d4, [%[lhs_ptr], #48]\n"  // Load 1st half of 2nd Lhs cell
        "vmov d1, r2, r3\n"             // Prepare 2nd half of Rhs cell
        "vmla.f32 q4, q1, d0[0]\n"      // Multiply 1st Lhs cell with column 0
        "ldr r2, [%[lhs_ptr], #56]\n"   // Load 2nd half of 2nd Lhs cell, part 1
        "vmla.f32 q5, q1, d0[1]\n"      // Multiply 1st Lhs cell with column 1
        "ldr r3, [%[lhs_ptr], #60]\n"   // Load 2nd half of 2nd Lhs cell, part 2
        "vmla.f32 q6, q1, d1[0]\n"      // Multiply 1st Lhs cell with column 2
        "subs %[depth], #2\n"           // Decrement depth counter

        "vldr d6, [%[lhs_ptr], #80]\n"  // Load 1st half of 3rd Lhs cell
        "vmov d5, r2, r3\n"             // Prepare 2nd half of 2nd Lhs cell
        "vmla.f32 q7, q1, d1[1]\n"      // Multiply 1st Lhs cell with column 3
        "ldr r2, [%[lhs_ptr], #88]\n"   // Load 2nd half of 3rd Lhs cell, part 1
        "vmla.f32 q8, q2, d0[0]\n"      // Multiply 2nd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #92]\n"   // Load 2nd half of 3rd Lhs cell, part 2
        "vmla.f32 q9, q2, d0[1]\n"      // Multiply 2nd Lhs cell with column 1
        "add %[rhs_ptr], %[rhs_ptr], #32\n"  // Move forward by 1 Rhs cell

        "vldr d2, [%[lhs_ptr], #96]\n"  // Load 1st half of 1st Lhs cell of next
        // iteration
        "vmov d7, r2, r3\n"             // Prepare 2nd half of 3rd Lhs cell
        "vmla.f32 q10, q2, d1[0]\n"     // Multiply 2nd Lhs cell with column 2
        "ldr r2, [%[lhs_ptr], #104]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 1
        "vmla.f32 q12, q3, d0[0]\n"     // Multiply 3rd Lhs cell with column 0
        "ldr r3, [%[lhs_ptr], #108]\n"  // Load 2nd half of 1st Lhs cell of next
        // iter, part 2
        "vmla.f32 q13, q3, d0[1]\n"  // Multiply 3rd Lhs cell with column 1
        "add %[lhs_ptr], %[lhs_ptr], #96\n"  // Move forward by 3 Lhs cells

        "vldr d0, [%[rhs_ptr]]\n"  // Load 1st half of Rhs cell of next
        // iteration
        "vmov d3, r2, r3\n"  // Prepare 2nd half of 1st Lhs cell of next
        // iteration
        "vmla.f32 q11, q2, d1[1]\n"   // Multiply 2nd Lhs cell with column 3
        "ldr r2, [%[rhs_ptr], #8]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 1
        "vmla.f32 q14, q3, d1[0]\n"    // Multiply 3rd Lhs cell with column 2
        "ldr r3, [%[rhs_ptr], #12]\n"  // Load 2nd half of Rhs cell of next
        // iteration, part 2
        "vmla.f32 q15, q3, d1[1]\n"  // Multiply 3rd Lhs cell with column 3

        // Loop branch.  This will dual issue in fmla cycle 3 of the 4th block.
        //"bne loop_%=\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5",
        "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
        "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "d28", "d29", "d30", "d31");
  }
};

// This rotating variant performs well when permutations (vext) can be
// dual-issued with arithmetic instructions.
struct NEON_32bit_GEMM_Float32_MLA_Rotating {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

#define NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS \
  "vtrn.32 q4, q5\n"                                                 \
  "vtrn.32 q6, q7\n"                                                 \
  "vswp d9, d12\n"                                                   \
  "vswp d11, d14\n"                                                  \
  "vtrn.32 q8, q9\n"                                                 \
  "vtrn.32 q10, q11\n"                                               \
  "vswp d17, d20\n"                                                  \
  "vswp d19, d22\n"                                                  \
  "vtrn.32 q12, q13\n"                                               \
  "vtrn.32 q14, q15\n"                                               \
  "vswp d25, d28\n"                                                  \
  "vswp d27, d30\n"

#define NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(a, b, c) \
  NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS             \
  "vext.32 q5, q5, q5, #" #a                                               \
  "\n"                                                                     \
  "vext.32 q6, q6, q6, #" #b                                               \
  "\n"                                                                     \
  "vext.32 q7, q7, q7, #" #c                                               \
  "\n"                                                                     \
  "vext.32 q9, q9, q9, #" #a                                               \
  "\n"                                                                     \
  "vext.32 q10, q10, q10, #" #b                                            \
  "\n"                                                                     \
  "vext.32 q11, q11, q11, #" #c                                            \
  "\n"                                                                     \
  "vext.32 q13, q13, q13, #" #a                                            \
  "\n"                                                                     \
  "vext.32 q14, q14, q14, #" #b                                            \
  "\n"                                                                     \
  "vext.32 q15, q15, q15, #" #c                                            \
  "\n" NEON_32BIT_ROTATING_FLOAT_KERNEL_TRANSPOSE_ACCUMULATOR_CELLS

        NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(1, 2, 3)

        //"loop_%=:\n"
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 1 Rhs cell of size 1x4
        "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vmla.f32 q4, q1, q0\n"
        "vmla.f32 q8, q2, q0\n"
        "vmla.f32 q12, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vmla.f32 q5, q1, q0\n"
        "vmla.f32 q9, q2, q0\n"
        "vmla.f32 q13, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vmla.f32 q6, q1, q0\n"
        "vmla.f32 q10, q2, q0\n"
        "vmla.f32 q14, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vmla.f32 q7, q1, q0\n"
        "vmla.f32 q11, q2, q0\n"
        "vmla.f32 q15, q3, q0\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        //"bne loop_%=\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov r0, %[accum_ptr]\n"

        NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(3, 2, 1)

            "vst1.32 {d8, d9},   [r0]!\n"
            "vst1.32 {d16, d17}, [r0]!\n"
            "vst1.32 {d24, d25}, [r0]!\n"
            "vst1.32 {d10, d11}, [r0]!\n"
            "vst1.32 {d18, d19}, [r0]!\n"
            "vst1.32 {d26, d27}, [r0]!\n"
            "vst1.32 {d12, d13}, [r0]!\n"
            "vst1.32 {d20, d21}, [r0]!\n"
            "vst1.32 {d28, d29}, [r0]!\n"
            "vst1.32 {d14, d15}, [r0]!\n"
            "vst1.32 {d22, d23}, [r0]!\n"
            "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

// This rotating variant performs well when permutations (vext) can be
// dual-issued with arithmetic instructions. It is relevant as the rotating
// approach removes the need for multiply-with-scalar instructions, and ARMv7
// FMA does not have a with-scalar variant.
struct NEON_32bit_GEMM_Float32_FMA_Rotating {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov r0, %[accum_ptr]\n"
        "vld1.32 {d8, d9},   [r0]!\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vld1.32 {d24, d25}, [r0]!\n"
        "vld1.32 {d10, d11}, [r0]!\n"
        "vld1.32 {d18, d19}, [r0]!\n"
        "vld1.32 {d26, d27}, [r0]!\n"
        "vld1.32 {d12, d13}, [r0]!\n"
        "vld1.32 {d20, d21}, [r0]!\n"
        "vld1.32 {d28, d29}, [r0]!\n"
        "vld1.32 {d14, d15}, [r0]!\n"
        "vld1.32 {d22, d23}, [r0]!\n"
        "vld1.32 {d30, d31}, [r0]!\n"

        NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(1, 2, 3)

        //"loop_%=:\n"
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 1 Rhs cell of size 1x4
        "vld1.32 {d0, d1}, [%[rhs_ptr]]!\n"

        // Load 3 Lhs cells of size 4x1 each
        "vld1.32 {d2, d3}, [%[lhs_ptr]]!\n"
        "vld1.32 {d4, d5}, [%[lhs_ptr]]!\n"
        "vld1.32 {d6, d7}, [%[lhs_ptr]]!\n"

        // Multiply-accumulate
        "vfma.f32 q4, q1, q0\n"
        "vfma.f32 q8, q2, q0\n"
        "vfma.f32 q12, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vfma.f32 q5, q1, q0\n"
        "vfma.f32 q9, q2, q0\n"
        "vfma.f32 q13, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vfma.f32 q6, q1, q0\n"
        "vfma.f32 q10, q2, q0\n"
        "vfma.f32 q14, q3, q0\n"
        "vext.f32 q0, q0, q0, #1\n"
        "vfma.f32 q7, q1, q0\n"
        "vfma.f32 q11, q2, q0\n"
        "vfma.f32 q15, q3, q0\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %[depth], #1\n"
        //"bne loop_%=\n"
        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        NEON_32BIT_ROTATING_FLOAT_KERNEL_ROTATE_ACCUMULATOR_CELLS(3, 2, 1)

        // Store accumulators
        "mov r0, %[accum_ptr]\n"
        "vst1.32 {d8, d9},   [r0]!\n"
        "vst1.32 {d16, d17}, [r0]!\n"
        "vst1.32 {d24, d25}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]!\n"
        "vst1.32 {d18, d19}, [r0]!\n"
        "vst1.32 {d26, d27}, [r0]!\n"
        "vst1.32 {d12, d13}, [r0]!\n"
        "vst1.32 {d20, d21}, [r0]!\n"
        "vst1.32 {d28, d29}, [r0]!\n"
        "vst1.32 {d14, d15}, [r0]!\n"
        "vst1.32 {d22, d23}, [r0]!\n"
        "vst1.32 {d30, d31}, [r0]!\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }
};

#endif  // __arm__

#ifdef __aarch64__

// This is the current standard kernel in gemmlowp, see:
// https://github.com/google/gemmlowp/blob/b1e2a29ff866680028f3080efc244e10e8dd7f46/internal/kernel_neon.h#L646
struct NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load 1 Rhs cell of size 2x8
        "ld1 {v5.8b}, [%[rhs_ptr]], #8\n"
        "ld1 {v6.8b}, [%[rhs_ptr]], #8\n"

        // Load 3 Lhs cells of size 4x2 each
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

        "subs %w[depth], %w[depth], #2\n"

        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Overview of register layout:
        //
        // A 2x8 block of 2 2x4 cells of Rhs is stored in 16bit in v0--v1.
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in v2--v4.
        // A 12x8 block of accumulators is stored in 32bit in v8--v31.
        //
        //                         +--------+--------+-----+--------+--------+
        //                         |v0.h[0] |v0.h[1] | ... |v1.h[2] |v1.h[3] |
        //                    Rhs  +--------+--------+-----+--------+--------+
        //                         |v0.h[4] |v0.h[5] | ... |v1.h[6] |v1.h[7] |
        //                         +--------+--------+-----+--------+--------+
        //
        //                         |        |        |     |        |        |
        //
        //    Lhs                  |        |        |     |        |        |
        //
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v2.h[0]|v2.h[4]|      |v8.s[0] |v9.s[0] | ... |v14.s[0]|v15.s[0]|
        //  |v2.h[1]|v2.h[5]|      |v8.s[1] |v9.s[1] | ... |v14.s[1]|v15.s[1]|
        //  |v2.h[2]|v2.h[6]|      |v8.s[2] |v9.s[2] | ... |v14.s[2]|v15.s[2]|
        //  |v2.h[3]|v2.h[7]|      |v8.s[3] |v9.s[3] | ... |v14.s[3]|v15.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v3.h[0]|v3.h[4]|      |v16.s[0]|v17.s[0]| ... |v22.s[0]|v23.s[0]|
        //  |v3.h[1]|v3.h[5]|      |v16.s[1]|v17.s[1]| ... |v22.s[1]|v23.s[1]|
        //  |v3.h[2]|v3.h[6]|      |v16.s[2]|v17.s[2]| ... |v22.s[2]|v23.s[2]|
        //  |v3.h[3]|v3.h[7]|      |v16.s[3]|v17.s[3]| ... |v22.s[3]|v23.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //  |v4.h[0]|v4.h[4]|      |v24.s[0]|v25.s[0]| ... |v30.s[0]|v31.s[0]|
        //  |v4.h[1]|v4.h[5]|      |v24.s[1]|v25.s[1]| ... |v30.s[1]|v31.s[1]|
        //  |v4.h[2]|v4.h[6]|      |v24.s[2]|v25.s[2]| ... |v30.s[2]|v31.s[2]|
        //  |v4.h[3]|v4.h[7]|      |v24.s[3]|v25.s[3]| ... |v30.s[3]|v31.s[3]|
        //  +-------+-------+ - -  +--------+--------+-----+--------+--------+
        //
        //                            Accumulator

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v5.8b\n"
        "ld1 {v5.8b}, [%[rhs_ptr]], #8\n"
        "uxtl v1.8h, v6.8b\n"
        "ld1 {v6.8b}, [%[rhs_ptr]], #8\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"

        // Multiply-accumulate, top third
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v2.4h, v1.h[0]\n"
        "umlal v13.4s, v2.4h, v1.h[1]\n"
        "umlal v14.4s, v2.4h, v1.h[2]\n"
        "umlal v15.4s, v2.4h, v1.h[3]\n"
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v2.8h, v1.h[4]\n"
        "umlal2 v13.4s, v2.8h, v1.h[5]\n"
        "umlal2 v14.4s, v2.8h, v1.h[6]\n"
        "umlal2 v15.4s, v2.8h, v1.h[7]\n"
        "ld1 {v2.8b}, [%[lhs_ptr]], #8\n"

        // Multiply-accumulate, middle third
        "umlal v16.4s, v3.4h, v0.h[0]\n"
        "umlal v17.4s, v3.4h, v0.h[1]\n"
        "umlal v18.4s, v3.4h, v0.h[2]\n"
        "umlal v19.4s, v3.4h, v0.h[3]\n"
        "umlal v20.4s, v3.4h, v1.h[0]\n"
        "umlal v21.4s, v3.4h, v1.h[1]\n"
        "umlal v22.4s, v3.4h, v1.h[2]\n"
        "umlal v23.4s, v3.4h, v1.h[3]\n"
        "umlal2 v16.4s, v3.8h, v0.h[4]\n"
        "umlal2 v17.4s, v3.8h, v0.h[5]\n"
        "umlal2 v18.4s, v3.8h, v0.h[6]\n"
        "umlal2 v19.4s, v3.8h, v0.h[7]\n"
        "umlal2 v20.4s, v3.8h, v1.h[4]\n"
        "umlal2 v21.4s, v3.8h, v1.h[5]\n"
        "umlal2 v22.4s, v3.8h, v1.h[6]\n"
        "umlal2 v23.4s, v3.8h, v1.h[7]\n"
        "ld1 {v3.8b}, [%[lhs_ptr]], #8\n"

        "subs %w[depth], %w[depth], #2\n"

        // Multiply-accumulate, bottom third
        "umlal v24.4s, v4.4h, v0.h[0]\n"
        "umlal v25.4s, v4.4h, v0.h[1]\n"
        "umlal v26.4s, v4.4h, v0.h[2]\n"
        "umlal v27.4s, v4.4h, v0.h[3]\n"
        "umlal v28.4s, v4.4h, v1.h[0]\n"
        "umlal v29.4s, v4.4h, v1.h[1]\n"
        "umlal v30.4s, v4.4h, v1.h[2]\n"
        "umlal v31.4s, v4.4h, v1.h[3]\n"
        "umlal2 v24.4s, v4.8h, v0.h[4]\n"
        "umlal2 v25.4s, v4.8h, v0.h[5]\n"
        "umlal2 v26.4s, v4.8h, v0.h[6]\n"
        "umlal2 v27.4s, v4.8h, v0.h[7]\n"
        "umlal2 v28.4s, v4.8h, v1.h[4]\n"
        "umlal2 v29.4s, v4.8h, v1.h[5]\n"
        "umlal2 v30.4s, v4.8h, v1.h[6]\n"
        "umlal2 v31.4s, v4.8h, v1.h[7]\n"
        "ld1 {v4.8b}, [%[lhs_ptr]], #8\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Expand Lhs/Rhs cells to 16 bit.
        "uxtl v0.8h, v5.8b\n"
        "uxtl v1.8h, v6.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"

        // Multiply-accumulate, level of depth 0
        "umlal v8.4s, v2.4h, v0.h[0]\n"
        "umlal v9.4s, v2.4h, v0.h[1]\n"
        "umlal v10.4s, v2.4h, v0.h[2]\n"
        "umlal v11.4s, v2.4h, v0.h[3]\n"
        "umlal v12.4s, v2.4h, v1.h[0]\n"
        "umlal v13.4s, v2.4h, v1.h[1]\n"
        "umlal v14.4s, v2.4h, v1.h[2]\n"
        "umlal v15.4s, v2.4h, v1.h[3]\n"
        "umlal v16.4s, v3.4h, v0.h[0]\n"
        "umlal v17.4s, v3.4h, v0.h[1]\n"
        "umlal v18.4s, v3.4h, v0.h[2]\n"
        "umlal v19.4s, v3.4h, v0.h[3]\n"
        "umlal v20.4s, v3.4h, v1.h[0]\n"
        "umlal v21.4s, v3.4h, v1.h[1]\n"
        "umlal v22.4s, v3.4h, v1.h[2]\n"
        "umlal v23.4s, v3.4h, v1.h[3]\n"
        "umlal v24.4s, v4.4h, v0.h[0]\n"
        "umlal v25.4s, v4.4h, v0.h[1]\n"
        "umlal v26.4s, v4.4h, v0.h[2]\n"
        "umlal v27.4s, v4.4h, v0.h[3]\n"
        "umlal v28.4s, v4.4h, v1.h[0]\n"
        "umlal v29.4s, v4.4h, v1.h[1]\n"
        "umlal v30.4s, v4.4h, v1.h[2]\n"
        "umlal v31.4s, v4.4h, v1.h[3]\n"

        // Multiply-accumulate, level of depth 1
        "umlal2 v8.4s, v2.8h, v0.h[4]\n"
        "umlal2 v9.4s, v2.8h, v0.h[5]\n"
        "umlal2 v10.4s, v2.8h, v0.h[6]\n"
        "umlal2 v11.4s, v2.8h, v0.h[7]\n"
        "umlal2 v12.4s, v2.8h, v1.h[4]\n"
        "umlal2 v13.4s, v2.8h, v1.h[5]\n"
        "umlal2 v14.4s, v2.8h, v1.h[6]\n"
        "umlal2 v15.4s, v2.8h, v1.h[7]\n"
        "umlal2 v16.4s, v3.8h, v0.h[4]\n"
        "umlal2 v17.4s, v3.8h, v0.h[5]\n"
        "umlal2 v18.4s, v3.8h, v0.h[6]\n"
        "umlal2 v19.4s, v3.8h, v0.h[7]\n"
        "umlal2 v20.4s, v3.8h, v1.h[4]\n"
        "umlal2 v21.4s, v3.8h, v1.h[5]\n"
        "umlal2 v22.4s, v3.8h, v1.h[6]\n"
        "umlal2 v23.4s, v3.8h, v1.h[7]\n"
        "umlal2 v24.4s, v4.8h, v0.h[4]\n"
        "umlal2 v25.4s, v4.8h, v0.h[5]\n"
        "umlal2 v26.4s, v4.8h, v0.h[6]\n"
        "umlal2 v27.4s, v4.8h, v0.h[7]\n"
        "umlal2 v28.4s, v4.8h, v1.h[4]\n"
        "umlal2 v29.4s, v4.8h, v1.h[5]\n"
        "umlal2 v30.4s, v4.8h, v1.h[6]\n"
        "umlal2 v31.4s, v4.8h, v1.h[7]\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// Faster kernel by ARM. Not expanding operands before multiplication.
// Tuned for A57. Compare to
// NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand
struct NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand_A57 {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<5, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    static const int kLhsWidth = Format::Lhs::kWidth;
    static const int kRhsWidth = Format::Rhs::kWidth;
    AccumulatorType rowmajor_accumulator_buffer[kLhsWidth * kRhsWidth];
    asm volatile(
        // Clear aggregators
        "dup v12.4s, wzr\n"
        "dup v13.4s, wzr\n"
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Overview of register layout:
        //
        // A 4x16 block of Rhs is stored in 8 bit in v0--v3.
        // A 5x16 block of Lhs is cycled through v4 and v5 in 8 bit.
        //
        // A 4x5 block of aggregators is stored in v12-v31 (as 4x32 bit
        // components which would need to be added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply to produce an intermediate result which is stored in
        // v6-v11.  Each intermediate result is 8x16 bits so this happens
        // twice for each Lhs/Rhs combination (once with UMULL for elements
        // 0-7 and once with UMULL2 for elements 8-15).
        //
        // UADALP is used to accumulate these intermediate results into the
        // result aggregators.
        //
        //
        //
        //                               +--------+--------+--------+--------+
        //                               |v0.b[0] |v1.b[0] |v2.b[0] |v3.b[0] |
        //                          Rhs  +--------+--------+--------+--------+
        //                               |  ...   |  ...   |  ...   |  ...   |
        //                               +--------+--------+--------+--------|
        //                               |v0.b[15]|v1.b[15]|v2.b[15]|v3.b[15]|
        //                               +--------+--------+--------+--------+
        //
        //                               |        |        |        |        |
        //
        //    Lhs                        |        |        |        |        |
        //
        //  +-------+-----+--------+ - - +--------+--------+--------+--------+
        //  |v4.b[0]| ... |v4.b[15]|     | v12.4s | v13.4s | v14.4s | v15.4s |
        //  |v5.b[0]| ... |v5.b[15]|     | v16.4s | v17.4s | v18.4s | v19.4s |
        //  |v4.b[0]| ... |v4.b[15]|     | v20.4s | v21.4s | v22.4s | v23.4s |
        //  |v5.b[0]| ... |v5.b[15]|     | v24.4s | v25.4s | v26.4s | v27.4s |
        //  |v4.b[0]| ... |v4.b[15]|     | v28.4s | v29.4s | v30.4s | v31.4s |
        //  +-------+--------------+ - - +--------+--------+--------+--------+
        //
        //                                                Accumulator
        //
        //
        // Further possible optimisations (not tried):
        //   - Move early loads into previous iteration (see Float32_WithScalar
        //   for example). - Unroll loop 2x to alternate more smoothly between
        //   v4 and v5. - A different number of temporary registers might work
        //   better. - Pairing umull with corresponding umull2 might allow
        //   better
        //     register loading (e.g. at the start of the loop)
        //   - Interleaving umull{2} and uadalp even more aggressively might
        //     help, (not sure about latency vs. dispatch rate).
        //
        //
        // Start loading Rhs - further loads are interleaved amongst the
        // multiplies for better dispatch on A57.
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"

        // Load first Lhs vector - further loads are interleaved amongst the
        // multiplies
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"

        "umull    v6.8h,  v0.8b,  v4.8b\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"  // 2nd RHS element
        "umull    v7.8h,  v1.8b,  v4.8b\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"  // 3rd RHS element
        "umull    v8.8h,  v2.8b,  v4.8b\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"  // 4th RHS element
        "umull    v9.8h,  v3.8b,  v4.8b\n"
        "umull2  v10.8h, v0.16b, v4.16b\n"
        "umull2  v11.8h, v1.16b, v4.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"  // 2nd LHS element

        "uadalp  v12.4s, v6.8h\n"
        "umull2   v6.8h, v2.16b, v4.16b\n"
        "uadalp  v13.4s, v7.8h\n"
        "umull2   v7.8h, v3.16b, v4.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"  // 1st LHS element done - Reuse v4
        // for 3rd LHS element
        "uadalp  v14.4s, v8.8h\n"
        "umull    v8.8h,  v0.8b,  v5.8b\n"
        "uadalp  v15.4s, v9.8h\n"
        "umull    v9.8h,  v1.8b,  v5.8b\n"
        "uadalp  v12.4s, v10.8h\n"
        "umull   v10.8h,  v2.8b,  v5.8b\n"
        "uadalp  v13.4s, v11.8h\n"
        "umull   v11.8h,  v3.8b,  v5.8b\n"

        "uadalp  v14.4s, v6.8h\n"
        "umull2   v6.8h, v0.16b, v5.16b\n"
        "uadalp  v15.4s, v7.8h\n"
        "umull2   v7.8h, v1.16b, v5.16b\n"
        "uadalp  v16.4s, v8.8h\n"
        "umull2   v8.8h, v2.16b, v5.16b\n"
        "uadalp  v17.4s, v9.8h\n"
        "umull2   v9.8h, v3.16b, v5.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"  // 2nd LHS element done - Reuse v5
        // for 4th LHS element
        "uadalp  v18.4s, v10.8h\n"
        "umull   v10.8h,  v0.8b,  v4.8b\n"
        "uadalp  v19.4s, v11.8h\n"
        "umull   v11.8h,  v1.8b,  v4.8b\n"

        "uadalp  v16.4s, v6.8h\n"
        "umull    v6.8h,  v2.8b,  v4.8b\n"
        "uadalp  v17.4s, v7.8h\n"
        "umull    v7.8h,  v3.8b,  v4.8b\n"
        "uadalp  v18.4s, v8.8h\n"
        "umull2   v8.8h, v0.16b, v4.16b\n"
        "uadalp  v19.4s, v9.8h\n"
        "umull2   v9.8h, v1.16b, v4.16b\n"
        "uadalp  v20.4s, v10.8h\n"
        "umull2  v10.8h, v2.16b, v4.16b\n"
        "uadalp  v21.4s, v11.8h\n"
        "umull2  v11.8h, v3.16b, v4.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"  // 3rd LHS element done - Reuse v4
        // for 5th LHS element

        "uadalp v22.4s, v6.8h\n"
        "umull    v6.8h,  v0.8b,  v5.8b\n"
        "uadalp v23.4s, v7.8h\n"
        "umull    v7.8h,  v1.8b,  v5.8b\n"
        "uadalp v20.4s, v8.8h\n"
        "umull    v8.8h,  v2.8b,  v5.8b\n"
        "uadalp v21.4s, v9.8h\n"
        "umull    v9.8h,  v3.8b,  v5.8b\n"
        "uadalp v22.4s, v10.8h\n"
        "umull2  v10.8h, v0.16b, v5.16b\n"
        "uadalp v23.4s, v11.8h\n"
        "umull2  v11.8h, v1.16b, v5.16b\n"

        "uadalp v24.4s, v6.8h\n"
        "umull2   v6.8h,  v2.16b, v5.16b\n"
        "uadalp v25.4s, v7.8h\n"
        "umull2   v7.8h,  v3.16b, v5.16b\n"
        "uadalp v26.4s, v8.8h\n"
        "umull    v8.8h,  v0.8b,  v4.8b\n"
        "uadalp v27.4s, v9.8h\n"
        "umull    v9.8h,  v1.8b,  v4.8b\n"
        "uadalp v24.4s, v10.8h\n"
        "umull   v10.8h,  v2.8b,  v4.8b\n"
        "uadalp v25.4s, v11.8h\n"
        "umull   v11.8h,  v3.8b,  v4.8b\n"

        "uadalp v26.4s, v6.8h\n"
        "umull2   v6.8h, v0.16b, v4.16b\n"
        "uadalp v27.4s, v7.8h\n"
        "umull2   v7.8h, v1.16b, v4.16b\n"
        "uadalp v28.4s, v8.8h\n"
        "umull2   v8.8h, v2.16b, v4.16b\n"
        "uadalp v29.4s, v9.8h\n"
        "umull2   v9.8h, v3.16b, v4.16b\n"
        "uadalp v30.4s, v10.8h\n"
        "uadalp v31.4s, v11.8h\n"

        "uadalp v28.4s, v6.8h\n"
        "uadalp v29.4s, v7.8h\n"
        // Loop. Decrement loop index (depth) by 16, since we just handled
        // 16 levels of depth.  Do this subs a bit before the end of the loop
        // for better dispatch on A57.
        "subs %w[depth], %w[depth], #16\n"
        "uadalp v30.4s, v8.8h\n"
        "uadalp v31.4s, v9.8h\n"

        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Reduce aggregators horizontally
        "addp v0.4s, v12.4s, v13.4s\n"
        "addp v1.4s, v14.4s, v15.4s\n"
        "addp v2.4s, v16.4s, v17.4s\n"
        "addp v3.4s, v18.4s, v19.4s\n"
        "addp v4.4s, v20.4s, v21.4s\n"
        "addp v5.4s, v22.4s, v23.4s\n"
        "addp v6.4s, v24.4s, v25.4s\n"
        "addp v7.4s, v26.4s, v27.4s\n"
        "addp v8.4s, v28.4s, v29.4s\n"
        "addp v9.4s, v30.4s, v31.4s\n"

        "addp v10.4s, v0.4s, v1.4s\n"
        "addp v11.4s, v2.4s, v3.4s\n"
        "addp v12.4s, v4.4s, v5.4s\n"
        "addp v13.4s, v6.4s, v7.4s\n"
        "addp v14.4s, v8.4s, v9.4s\n"

        "mov x0, %[rowmajor_accumulator_buffer]\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [rowmajor_accumulator_buffer] "r"(rowmajor_accumulator_buffer)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");

    // accumulate row-major accumulators into global (column-major) accumulators
    for (int l = 0; l < kLhsWidth; l++) {
      for (int r = 0; r < kRhsWidth; r++) {
        accum_ptr[l + kLhsWidth * r] +=
            rowmajor_accumulator_buffer[r + l * kRhsWidth];
      }
    }
  }
};

// Fast kernel operating on int8 operands.
// It is assumed that one of the two int8 operands only takes values
// in [-127, 127], while the other may freely range in [-128, 127].
// The issue with both operands taking the value -128 is that:
// -128*-128 + -128*-128 == -32768 overflows int16.
// Every other expression a*b + c*d, for any int8 a,b,c,d, fits in int16
// range. That is the basic idea of this kernel.
struct NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    std::size_t start_depth = 123;
    std::size_t run_depth = depth;
    std::size_t dst_col_stride = 4;
    AccumulatorType* dst_ptr = accum_ptr;
    asm volatile(
        // Overview of register layout:
        //
        // A 4x16 block of Rhs is stored in 8 bit in v0--v3.
        // A 4x16 block of Lhs is stored in 8 bit in v4--v7.
        //
        // A 4x4 block of accumulators is stored in v16-v31 (as 4x32 bit
        // components which need to be horizontally-added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        // Here comes the special trick: since the operands are signed int8,
        // their range being [ -2^7 , 2^7 ), their products are in range
        // [ -2^14 , 2^14 - 1 ), meaning that we can add two such values
        // without any risk of overflowing int16.
        // We thus proceed with the 8 next levels of depth, multiplying
        // again Lhs by Rhs, accumulating into this existing int16x8 vector.
        //
        // Only then, having processed 16 levels of depth, do we need to
        // horizontally add these int16x8 accumulators into the final
        // int32x4 accumulators.
        //
        // As we do not have enough registers to store all 16 int16x8
        // temporary-16bit-accumulators, we have them cycle through v8--v15.
        //
        //
        // Register layout (ignoring the v8--v15 temporary 16bit accumulators):
        //
        //                               +--------+--------+--------+--------+
        //                               |v0.b[0] |v1.b[0] |v2.b[0] |v3.b[0] |
        //                          Rhs  +--------+--------+--------+--------+
        //                               |  ...   |  ...   |  ...   |  ...   |
        //                               +--------+--------+--------+--------|
        //                               |v0.b[15]|v1.b[15]|v2.b[15]|v3.b[15]|
        //                               +--------+--------+--------+--------+
        //
        //                               |        |        |        |        |
        //
        //    Lhs                        |        |        |        |        |
        //
        //  +-------+-----+--------+ - - +--------+--------+--------+--------+
        //  |v4.b[0]| ... |v4.b[15]|     | v16.4s | v17.4s | v18.4s | v19.4s |
        //  |v5.b[0]| ... |v5.b[15]|     | v20.4s | v21.4s | v22.4s | v23.4s |
        //  |v6.b[0]| ... |v6.b[15]|     | v24.4s | v25.4s | v26.4s | v27.4s |
        //  |v7.b[0]| ... |v7.b[15]|     | v28.4s | v29.4s | v30.4s | v31.4s |
        //  +-------+--------------+ - - +--------+--------+--------+--------+
        //
        //                                                Accumulator
        //

        // Clear accumulators
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"
        "dup v16.4s, wzr\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"
        "dup v17.4s, wzr\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "dup v18.4s, wzr\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "dup v19.4s, wzr\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "dup v20.4s, wzr\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "dup v21.4s, wzr\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "dup v22.4s, wzr\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        "dup v23.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "dup v24.4s, wzr\n"
        "mov x0, %[dst_ptr]\n"
        "dup v25.4s, wzr\n"
        "dup v26.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v31.4s, wzr\n"

        "smull    v12.8h,  v0.8b,  v4.8b\n"
        "smull    v13.8h,  v1.8b,  v4.8b\n"
        "smull    v14.8h,  v0.8b,  v5.8b\n"
        "smull    v15.8h,  v1.8b,  v5.8b\n"
        "smlal2   v12.8h,  v0.16b,  v4.16b\n"
        "smlal2   v13.8h,  v1.16b,  v4.16b\n"
        "smlal2   v14.8h,  v0.16b,  v5.16b\n"
        "smlal2   v15.8h,  v1.16b,  v5.16b\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP "f\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        "subs %[run_depth], %[run_depth], #16\n"

        "sadalp  v16.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v13.8h\n"
        "smull    v13.8h,  v0.8b,  v7.8b\n"
        "sadalp  v20.4s, v14.8h\n"
        "smull    v14.8h,  v1.8b,  v6.8b\n"
        "sadalp  v21.4s, v15.8h\n"
        "smull    v15.8h,  v1.8b,  v7.8b\n"
        "smlal2   v12.8h,  v0.16b,  v6.16b\n"
        "smlal2   v13.8h,  v0.16b,  v7.16b\n"
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"
        "smlal2   v14.8h,  v1.16b,  v6.16b\n"
        "smlal2   v15.8h,  v1.16b,  v7.16b\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"
        "sadalp  v24.4s, v12.8h\n"
        "smull    v12.8h,  v2.8b,  v4.8b\n"
        "sadalp  v28.4s, v13.8h\n"
        "smull    v13.8h,  v3.8b,  v4.8b\n"
        "sadalp  v25.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v29.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        "smlal2   v12.8h,  v2.16b,  v4.16b\n"
        "smlal2   v13.8h,  v3.16b,  v4.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "sadalp  v18.4s, v12.8h\n"
        "smull    v12.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v13.8h\n"
        "smull    v13.8h,  v2.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v3.8b,  v6.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"
        "smlal2   v12.8h,  v2.16b,  v6.16b\n"
        "smlal2   v13.8h,  v2.16b,  v7.16b\n"
        "ld1 {v2.16b}, [%[rhs_ptr]], #16\n"
        "smlal2   v14.8h,  v3.16b,  v6.16b\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "sadalp  v26.4s, v12.8h\n"
        "ld1 {v3.16b}, [%[rhs_ptr]], #16\n"
        "smull    v12.8h,  v0.8b,  v4.8b\n"
        "sadalp  v30.4s, v13.8h\n"
        "smull    v13.8h,  v1.8b,  v4.8b\n"
        "sadalp  v27.4s, v14.8h\n"
        "smull    v14.8h,  v0.8b,  v5.8b\n"
        "sadalp  v31.4s, v15.8h\n"
        "smull    v15.8h,  v1.8b,  v5.8b\n"
        "smlal2   v12.8h,  v0.16b,  v4.16b\n"
        "smlal2   v13.8h,  v1.16b,  v4.16b\n"
        "smlal2   v14.8h,  v0.16b,  v5.16b\n"
        "smlal2   v15.8h,  v1.16b,  v5.16b\n"

        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Load accumulators from memory
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "mov x0, %[dst_ptr]\n"

        // Do the remaining arithmetic for the 16 last levels of depths.
        // All the operands are already loaded.
        "sadalp  v16.4s, v12.8h\n"
        "smull    v12.8h,  v0.8b,  v6.8b\n"
        "sadalp  v17.4s, v13.8h\n"
        "smull    v13.8h,  v0.8b,  v7.8b\n"
        "sadalp  v20.4s, v14.8h\n"
        "smull    v14.8h,  v1.8b,  v6.8b\n"
        "sadalp  v21.4s, v15.8h\n"
        "smull    v15.8h,  v1.8b,  v7.8b\n"
        "smlal2   v12.8h,  v0.16b,  v6.16b\n"
        "smlal2   v13.8h,  v0.16b,  v7.16b\n"
        "smlal2   v14.8h,  v1.16b,  v6.16b\n"
        "smlal2   v15.8h,  v1.16b,  v7.16b\n"
        "sadalp  v24.4s, v12.8h\n"
        "smull    v12.8h,  v2.8b,  v4.8b\n"
        "sadalp  v28.4s, v13.8h\n"
        "smull    v13.8h,  v3.8b,  v4.8b\n"
        "sadalp  v25.4s, v14.8h\n"
        "smull    v14.8h,  v2.8b,  v5.8b\n"
        "sadalp  v29.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v5.8b\n"
        "smlal2   v12.8h,  v2.16b,  v4.16b\n"
        "smlal2   v13.8h,  v3.16b,  v4.16b\n"
        "smlal2   v14.8h,  v2.16b,  v5.16b\n"
        "smlal2   v15.8h,  v3.16b,  v5.16b\n"
        "sadalp  v18.4s, v12.8h\n"
        "smull    v12.8h,  v2.8b,  v6.8b\n"
        "sadalp  v19.4s, v13.8h\n"
        "smull    v13.8h,  v2.8b,  v7.8b\n"
        "sadalp  v22.4s, v14.8h\n"
        "smull    v14.8h,  v3.8b,  v6.8b\n"
        "sadalp  v23.4s, v15.8h\n"
        "smull    v15.8h,  v3.8b,  v7.8b\n"
        "smlal2   v12.8h,  v2.16b,  v6.16b\n"
        "smlal2   v13.8h,  v2.16b,  v7.16b\n"
        "smlal2   v14.8h,  v3.16b,  v6.16b\n"
        "smlal2   v15.8h,  v3.16b,  v7.16b\n"
        "sadalp  v26.4s, v12.8h\n"
        "sadalp  v30.4s, v13.8h\n"
        "sadalp  v27.4s, v14.8h\n"
        "sadalp  v31.4s, v15.8h\n"

        // Reduce aggregators horizontally
        "addp v0.4s, v16.4s, v20.4s\n"
        "addp v1.4s, v17.4s, v21.4s\n"
        "addp v2.4s, v18.4s, v22.4s\n"
        "addp v3.4s, v19.4s, v23.4s\n"
        "addp v4.4s, v24.4s, v28.4s\n"
        "addp v5.4s, v25.4s, v29.4s\n"
        "addp v6.4s, v26.4s, v30.4s\n"
        "addp v7.4s, v27.4s, v31.4s\n"

        "addp v12.4s, v0.4s, v4.4s\n"
        "addp v13.4s, v1.4s, v5.4s\n"
        "addp v14.4s, v2.4s, v6.4s\n"
        "addp v15.4s, v3.4s, v7.4s\n"

        // Add to the accumulators loaded from memory
        "add v8.4s, v8.4s, v12.4s\n"
        "add v9.4s, v9.4s, v13.4s\n"
        "add v10.4s, v10.4s, v14.4s\n"
        "add v11.4s, v11.4s, v15.4s\n"

        // Store accumulators back to memory
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [run_depth] "+r"(run_depth),
        [dst_col_stride] "+r"(dst_col_stride)
        :  // inputs
        [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// Fast kernel operating on int8 operands with 7-bit range.
// It is assumed that one of the two operands only takes values in [-63, 63],
// while the other take values in [-64, 63].
// With this restriction, it is possible to multiply-accumulate operands into
// a 16-bit integer eight times without overflow.
struct NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<2, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
#define GEMMLOWP_LABEL_64_DEPTH_LOOP "1"
#define GEMMLOWP_LABEL_64_DEPTH_AFTER_LOOP "2"
#define GEMMLOWP_LABEL_16_DEPTH_LOOP "3"
#define GEMMLOWP_LABEL_16_DEPTH_AFTER_LOOP "4"

    AccumulatorType* dst_ptr = accum_ptr;
    asm volatile(
        // Overview of register layout:
        //
        // A 4x16 block of Lhs is stored in 8 bit in v0--v7.
        // A 2x16 block of Rhs is stored in 8 bit in v8--v15.
        //
        // A 4x2 block of global accumulators is stored in v24-v31 (as 4x32 bit
        // components which need to be horizontally-added at the end).
        //
        // A 4x2 block of local accumulators is stored in v16-v23 (as 8x16 bit
        // components which are added to global accumulators every 64 depth
        // iteration.
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        //
        // Like the trick used in the fast 8-bit kernel, the operands are
        // restricted to 7-bit range [-2^6, 2^6) so their products are in range
        // [-2^12, 2^12 -1). This enables adding eight such products without any
        // risk of overflowing int16, equating to 64 levels of depth before
        // horizontally adding these int16x8 accumulators into the final int32x4
        // accumulators.
        //
        // Register layout including both local and global accumulators.
        // Since we do not have enough registers to store all Lhs values, we
        // reuse the same registers v0--v7 to load the rest of the Lhs values.
        //
        //                            +-----+-----+
        //                            | v8  | v9  |
        //                       Rhs  +-----+-----+
        //                            | v10 | v11 |
        //                            +-----+-----+
        //                            | v12 | v13 |
        //                            +-----+-----+
        //                            | v14 | v15 |
        //    Lhs                     +-----+-----+
        //  +----+----+----+----+ - - +-----+-----+      +--------+--------+
        //  | v0 | v4 | v0 | v4 |     | v16 | v20 |      | v24.4s | v28.4s |
        //  | v1 | v5 | v1 | v5 |     | v17 | v21 |  ->  | v25.4s | v29.4s |
        //  | v2 | v6 | v2 | v6 |     | v18 | v22 |      | v26.4s | v30.4s |
        //  | v3 | v7 | v3 | v7 |     | v19 | v23 |      | v27.4s | v31.4s |
        //  +----+----+----+----+ - - +-----+-----+      +--------+--------+
        //
        //                           Local Accumulator    Global Accumulator
        //

        // Clear accumulators.
        "dup v16.4s, wzr\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "dup v24.4s, wzr\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "dup v17.4s, wzr\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "dup v25.4s, wzr\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "dup v18.4s, wzr\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"
        "dup v26.4s, wzr\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"
        "dup v19.4s, wzr\n"
        "dup v27.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v28.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v29.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v30.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v31.4s, wzr\n"

        "cmp %w[depth], #64\n"
        "blt " GEMMLOWP_LABEL_64_DEPTH_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        GEMMLOWP_LABEL_64_DEPTH_LOOP
        ":\n"
        "subs %w[depth], %w[depth], #64\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v24.4s, v16.8h\n"
        "smull v16.8h, v0.8b, v8.8b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v25.4s, v17.8h\n"
        "smull v17.8h, v1.8b, v8.8b\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v26.4s, v18.8h\n"
        "smull v18.8h, v2.8b, v8.8b\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v27.4s, v19.8h\n"
        "smull v19.8h, v3.8b, v8.8b\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v28.4s, v20.8h\n"
        "smull v20.8h, v0.8b, v9.8b\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v29.4s, v21.8h\n"
        "smull v21.8h, v1.8b, v9.8b\n"
        "ld1 {v12.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v30.4s, v22.8h\n"
        "smull v22.8h, v2.8b, v9.8b\n"
        "ld1 {v13.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v31.4s, v23.8h\n"
        "smull v23.8h, v3.8b, v9.8b\n"

        "cmp %w[depth], #64\n"
        "smlal2 v16.8h, v0.16b, v8.16b\n"
        "ld1 {v14.16b}, [%[rhs_ptr]], #16\n"
        "smlal2 v17.8h, v1.16b, v8.16b\n"
        "ld1 {v15.16b}, [%[rhs_ptr]], #16\n"
        "smlal2 v18.8h, v2.16b, v8.16b\n"
        "smlal2 v19.8h, v3.16b, v8.16b\n"

        "smlal2 v20.8h, v0.16b, v9.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v21.8h, v1.16b, v9.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v22.8h, v2.16b, v9.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v23.8h, v3.16b, v9.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "smlal v16.8h, v4.8b, v10.8b\n"
        "smlal v17.8h, v5.8b, v10.8b\n"
        "smlal v18.8h, v6.8b, v10.8b\n"
        "smlal v19.8h, v7.8b, v10.8b\n"
        "smlal v20.8h, v4.8b, v11.8b\n"

        "smlal v21.8h, v5.8b, v11.8b\n"
        "smlal v22.8h, v6.8b, v11.8b\n"
        "smlal v23.8h, v7.8b, v11.8b\n"

        "smlal2 v16.8h, v4.16b, v10.16b\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"
        "smlal2 v17.8h, v5.16b, v10.16b\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"
        "smlal2 v18.8h, v6.16b, v10.16b\n"
        "smlal2 v19.8h, v7.16b, v10.16b\n"

        "smlal2 v20.8h, v4.16b, v11.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v21.8h, v5.16b, v11.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v22.8h, v6.16b, v11.16b\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v23.8h, v7.16b, v11.16b\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"

        "smlal v16.8h, v0.8b, v12.8b\n"
        "smlal v17.8h, v1.8b, v12.8b\n"
        "smlal v18.8h, v2.8b, v12.8b\n"
        "smlal v19.8h, v3.8b, v12.8b\n"
        "smlal v20.8h, v0.8b, v13.8b\n"
        "smlal v21.8h, v1.8b, v13.8b\n"
        "smlal v22.8h, v2.8b, v13.8b\n"
        "smlal v23.8h, v3.8b, v13.8b\n"

        "smlal2 v16.8h, v0.16b, v12.16b\n"
        "smlal2 v17.8h, v1.16b, v12.16b\n"
        "smlal2 v18.8h, v2.16b, v12.16b\n"
        "smlal2 v19.8h, v3.16b, v12.16b\n"

        "smlal2 v20.8h, v0.16b, v13.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v21.8h, v1.16b, v13.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v22.8h, v2.16b, v13.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v23.8h, v3.16b, v13.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "smlal v16.8h, v4.8b, v14.8b\n"
        "smlal v17.8h, v5.8b, v14.8b\n"
        "smlal v18.8h, v6.8b, v14.8b\n"
        "smlal v19.8h, v7.8b, v14.8b\n"

        "smlal v20.8h, v4.8b, v15.8b\n"
        "smlal v21.8h, v5.8b, v15.8b\n"
        "smlal v22.8h, v6.8b, v15.8b\n"
        "smlal v23.8h, v7.8b, v15.8b\n"

        "smlal2 v16.8h, v4.16b, v14.16b\n"
        "smlal2 v17.8h, v5.16b, v14.16b\n"
        "smlal2 v18.8h, v6.16b, v14.16b\n"
        "smlal2 v19.8h, v7.16b, v14.16b\n"

        "smlal2 v20.8h, v4.16b, v15.16b\n"
        "smlal2 v21.8h, v5.16b, v15.16b\n"
        "smlal2 v22.8h, v6.16b, v15.16b\n"
        "smlal2 v23.8h, v7.16b, v15.16b\n"

        "bge " GEMMLOWP_LABEL_64_DEPTH_LOOP "b\n"

        GEMMLOWP_LABEL_64_DEPTH_AFTER_LOOP
        ":\n"

        "cmp %w[depth], #16\n"
        "blt " GEMMLOWP_LABEL_16_DEPTH_AFTER_LOOP "f\n"

        //"loop_%=:\n"
        GEMMLOWP_LABEL_16_DEPTH_LOOP
        ":\n"
        "sadalp v24.4s, v16.8h\n"
        "smull v16.8h, v0.8b, v8.8b\n"
        "subs %w[depth], %w[depth], #16\n"
        "sadalp v25.4s, v17.8h\n"
        "smull v17.8h, v1.8b, v8.8b\n"
        "sadalp v26.4s, v18.8h\n"
        "smull v18.8h, v2.8b, v8.8b\n"
        "sadalp v27.4s, v19.8h\n"
        "smull v19.8h, v3.8b, v8.8b\n"
        "sadalp v28.4s, v20.8h\n"
        "smull v20.8h, v0.8b, v9.8b\n"
        "sadalp v29.4s, v21.8h\n"
        "smull v21.8h, v1.8b, v9.8b\n"
        "sadalp v30.4s, v22.8h\n"
        "smull v22.8h, v2.8b, v9.8b\n"
        "sadalp v31.4s, v23.8h\n"
        "smull v23.8h, v3.8b, v9.8b\n"

        "cmp %w[depth], #16\n"
        "smlal2 v16.8h, v0.16b, v8.16b\n"
        "smlal2 v17.8h, v1.16b, v8.16b\n"
        "smlal2 v18.8h, v2.16b, v8.16b\n"
        "smlal2 v19.8h, v3.16b, v8.16b\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"

        "smlal2 v20.8h, v0.16b, v9.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v21.8h, v1.16b, v9.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v22.8h, v2.16b, v9.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "smlal2 v23.8h, v3.16b, v9.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"

        "bge " GEMMLOWP_LABEL_16_DEPTH_LOOP "b\n"

        GEMMLOWP_LABEL_16_DEPTH_AFTER_LOOP
        ":\n"

        "sadalp v24.4s, v16.8h\n"
        "sadalp v25.4s, v17.8h\n"
        "sadalp v26.4s, v18.8h\n"
        "sadalp v27.4s, v19.8h\n"
        "sadalp v28.4s, v20.8h\n"
        "sadalp v29.4s, v21.8h\n"
        "sadalp v30.4s, v22.8h\n"
        "sadalp v31.4s, v23.8h\n"

        // Reduce aggregators horizontally.
        "addp v0.4s, v24.4s, v25.4s\n"
        "addp v1.4s, v26.4s, v27.4s\n"
        "addp v2.4s, v28.4s, v29.4s\n"
        "addp v3.4s, v30.4s, v31.4s\n"

        "addp v4.4s, v0.4s, v1.4s\n"
        "addp v5.4s, v2.4s, v3.4s\n"

        // Load accumulators from memory.
        "mov x0, %[dst_ptr]\n"
        "ld1 {v6.16b}, [x0], #16\n"
        "ld1 {v7.16b}, [x0], #16\n"

        // Add to the accumulators loaded from memory.
        "add v6.4s, v6.4s, v4.4s\n"
        "add v7.4s, v7.4s, v5.4s\n"

        // Store accumulators back to memory.
        "mov x0, %[dst_ptr]\n"
        "st1 {v6.16b}, [x0], #16\n"
        "st1 {v7.16b}, [x0], #16\n"

        :
        // Outputs.
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [depth] "+r"(depth)
        :
        // Inputs.

        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "x0");
  }
};

SET_7BIT_RANGES(NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits);

// Kernel operating on int8 operands with 4.25-bit range.
// It is assumed that one of the two operands only takes values in [-7, 7],
// while the other take values in [-9, 9].
// With this restriction, it is possible to multiply-accumulate operands into
// a 16-bit integer thirty-two times without overflow.
struct NEON_64bit_GEMM_Int425Operands {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 32, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<2, 32, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
#define GEMMLOWP_LABEL_512_DEPTH_LOOP "1"
#define GEMMLOWP_LABEL_32_DEPTH_LOOP "2"
#define GEMMLOWP_LABEL_32_DEPTH_AFTER_LOOP "3"

    AccumulatorType* dst_ptr = accum_ptr;
    int outer_depth = depth / 512 + 1;

    asm volatile(
        // Overview of register layout:
        //
        // A 4x32 block of Lhs is stored in 8 bit in v0--v7.
        // A 2x32 block of Rhs is stored in 8 bit in v8--v11.
        //
        // A 4x2 block of global accumulators is stored in v24-v31 (as 4x32 bit
        // components which need to be horizontally-added at the end).
        //
        // A 4x2 block of local accumulators is stored in v16-v23 (as 8x16 bit
        // components which are horizontally-added to global accumulators every
        // 512 depth iteration.
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a multiply
        // over the 16 first levels of depth, producing int8x16 vectors of
        // products for each position in the accumulator matrix.
        //
        // Like the trick used in the fast 8-bit and 7-bit kernels, the operands
        // are restricted to 4.25-bit range, [-7, 7] for one operand and [-9, 9]
        // for the other operand. This enables adding two such products without
        // any risk of overflowing int8, and thiry-two such products without
        // overflowing int16. This equates to 512 levels of depth before
        // horizontally adding these int16x8 accumulators into the final int32x4
        // accumulators.
        //
        // Register layout (ignoring the v12--v15 temporary 8-bit accumulators).
        // Since we do not have enough registers to store all Lhs values and Rhs
        // values, we reuse the same registers v0--v7 to load subsequent Lhs
        // values and v8-v11 to subsequent Rhs values.
        //
        //                            +-----+-----+
        //                            | v8  | v9  |
        //                       Rhs  +-----+-----+
        //                            | v10 | v11 |
        //                            +-----+-----+
        //                            | v8  | v9  |
        //                            +-----+-----+
        //                            | v10 | v11 |
        //    Lhs                     +-----+-----+
        //  +----+----+----+----+ - - +-----+-----+      +--------+--------+
        //  | v0 | v4 | v0 | v4 |     | v16 | v17 |      | v24.4s | v25.4s |
        //  | v1 | v5 | v1 | v5 |     | v18 | v19 |  ->  | v26.4s | v27.4s |
        //  | v2 | v6 | v2 | v6 |     | v20 | v21 |      | v28.4s | v29.4s |
        //  | v3 | v7 | v3 | v7 |     | v22 | v23 |      | v30.4s | v31.4s |
        //  +----+----+----+----+ - - +-----+-----+      +--------+--------+
        //
        //                           Local Accumulator    Global Accumulator
        //

        // Clear global accumulators.
        "dup v24.4s, wzr\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"
        "dup v25.4s, wzr\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"
        "dup v26.4s, wzr\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"
        "dup v27.4s, wzr\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "dup v28.4s, wzr\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "dup v29.4s, wzr\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "dup v30.4s, wzr\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "dup v31.4s, wzr\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"

        //"loop_%=:\n"
        GEMMLOWP_LABEL_512_DEPTH_LOOP
        ":\n"
        // Clear local accumulators.
        "dup v16.8h, wzr\n"
        "dup v17.8h, wzr\n"
        "dup v18.8h, wzr\n"
        "mov x1, #512\n"
        "dup v19.8h, wzr\n"
        "dup v20.8h, wzr\n"
        "dup v21.8h, wzr\n"
        "dup v22.8h, wzr\n"
        "dup v23.8h, wzr\n"

        //"loop_%=:\n"
        GEMMLOWP_LABEL_32_DEPTH_LOOP
        ":\n"
        "mul v12.16b, v0.16b, v8.16b\n"
        "mul v13.16b, v0.16b, v10.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "mul v14.16b, v2.16b, v8.16b\n"
        "mul v15.16b, v2.16b, v10.16b\n"

        "mla v12.16b, v1.16b, v9.16b\n"
        "mla v13.16b, v1.16b, v11.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "mla v14.16b, v3.16b, v9.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "mla v15.16b, v3.16b, v11.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "sadalp v16.8h, v12.16b\n"
        "sadalp v17.8h, v13.16b\n"
        "subs %w[depth], %w[depth], #32\n"
        "sadalp v18.8h, v14.16b\n"
        "sadalp v19.8h, v15.16b\n"
        "subs x1, x1, #32\n"

        "mul v12.16b, v4.16b, v8.16b\n"
        "mul v13.16b, v4.16b, v10.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "mul v14.16b, v6.16b, v8.16b\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"
        "mul v15.16b, v6.16b, v10.16b\n"

        "mla v12.16b, v5.16b, v9.16b\n"
        "mla v13.16b, v5.16b, v11.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "mla v14.16b, v7.16b, v9.16b\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"
        "mla v15.16b, v7.16b, v11.16b\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"

        "sadalp v20.8h, v12.16b\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v21.8h, v13.16b\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v22.8h, v14.16b\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v23.8h, v15.16b\n"

        "mul v12.16b, v0.16b, v8.16b\n"
        "mul v13.16b, v0.16b, v10.16b\n"
        "ld1 {v0.16b}, [%[lhs_ptr]], #16\n"
        "mul v14.16b, v2.16b, v8.16b\n"
        "mul v15.16b, v2.16b, v10.16b\n"

        "mla v12.16b, v1.16b, v9.16b\n"
        "mla v13.16b, v1.16b, v11.16b\n"
        "ld1 {v1.16b}, [%[lhs_ptr]], #16\n"
        "mla v14.16b, v3.16b, v9.16b\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"
        "mla v15.16b, v3.16b, v11.16b\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"

        "sadalp v16.8h, v12.16b\n"
        "sadalp v17.8h, v13.16b\n"
        "sadalp v18.8h, v14.16b\n"
        "sadalp v19.8h, v15.16b\n"

        "mul v12.16b, v4.16b, v8.16b\n"
        "mul v13.16b, v4.16b, v10.16b\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"
        "mul v14.16b, v6.16b, v8.16b\n"
        "ld1 {v8.16b}, [%[rhs_ptr]], #16\n"
        "mul v15.16b, v6.16b, v10.16b\n"

        "mla v12.16b, v5.16b, v9.16b\n"
        "mla v13.16b, v5.16b, v11.16b\n"
        "ld1 {v5.16b}, [%[lhs_ptr]], #16\n"
        "mla v14.16b, v7.16b, v9.16b\n"
        "ld1 {v9.16b}, [%[rhs_ptr]], #16\n"
        "mla v15.16b, v7.16b, v11.16b\n"
        "ld1 {v10.16b}, [%[rhs_ptr]], #16\n"

        "sadalp v20.8h, v12.16b\n"
        "ld1 {v11.16b}, [%[rhs_ptr]], #16\n"
        "sadalp v21.8h, v13.16b\n"
        "ld1 {v6.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v22.8h, v14.16b\n"
        "ld1 {v7.16b}, [%[lhs_ptr]], #16\n"
        "sadalp v23.8h, v15.16b\n"

        "beq " GEMMLOWP_LABEL_32_DEPTH_AFTER_LOOP
        "f\n"

        "cmp %w[depth], #0\n"
        "bne " GEMMLOWP_LABEL_32_DEPTH_LOOP "b\n"

        GEMMLOWP_LABEL_32_DEPTH_AFTER_LOOP
        ":\n"

        // Pairwise add 16-bit local accums to 32-bit global accums.
        "sadalp v24.4s, v16.8h\n"
        "sadalp v25.4s, v17.8h\n"
        "sadalp v26.4s, v18.8h\n"
        "sadalp v27.4s, v19.8h\n"
        "sadalp v28.4s, v20.8h\n"
        "sadalp v29.4s, v21.8h\n"
        "sadalp v30.4s, v22.8h\n"
        "sadalp v31.4s, v23.8h\n"

        "bne " GEMMLOWP_LABEL_512_DEPTH_LOOP
        "b\n"

        // Reduce aggregators horizontally.
        "addp v0.4s, v24.4s, v26.4s\n"
        "addp v1.4s, v28.4s, v30.4s\n"
        "addp v2.4s, v25.4s, v27.4s\n"
        "addp v3.4s, v29.4s, v31.4s\n"

        "addp v4.4s, v0.4s, v1.4s\n"
        "addp v5.4s, v2.4s, v3.4s\n"

        // Load accumulators from memory.
        "mov x0, %[dst_ptr]\n"
        "ld1 {v6.16b}, [x0], #16\n"
        "ld1 {v7.16b}, [x0], #16\n"

        // Add to the accumulators loaded from memory.
        "add v6.4s, v6.4s, v4.4s\n"
        "add v7.4s, v7.4s, v5.4s\n"

        // Store accumulators back to memory.
        "mov x0, %[dst_ptr]\n"
        "st1 {v6.16b}, [x0], #16\n"
        "st1 {v7.16b}, [x0], #16\n"

        :
        // Outputs.
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [depth] "+r"(depth),
        [outer_depth] "+r"(outer_depth)
        :
        // Inputs.

        :
        // Clobbers.
        "cc", "memory",
        // We use these NEON registers
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
        "v31", "x0", "x1");
  }
};

SET_425BIT_RANGES(NEON_64bit_GEMM_Int425Operands);

#ifdef __ARM_FEATURE_DOTPROD
// Kernels utilizing the Armv8.2 Dot Product extension.
//
// The dot product instructions work by taking 4 consecutive 8-bit depth
// values from each operand, multiplying the 4 pairs together and
// accumulating all the results into the corresponding 32-bit accumulator
// lane.  As such, the operation is identical to a 32-bit instruction (like
// FMLA used in SGEMM), except that 4 depth values are processed at a time
// instead of 1.

// Thus, this first kernel is a carbon copy of
// "NEON_64bit_GEMM_Float32_WithScalar_A57" (which should provide good
// performance for most processors) below with the opcode (fmla -> udot) and
// types (float32 -> uint8/uint32) changed.
//
// A signed version of this kernel could be produced by replacing "udot"
// with "sdot" - performance should be identical to this udot kernel.
struct NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.4s}, [x0], #16\n"
        "ld1 {v16.4s}, [x0], #16\n"
        "ld1 {v24.4s}, [x0], #16\n"
        "ld1 {v9.4s}, [x0], #16\n"
        "ld1 {v17.4s}, [x0], #16\n"
        "ld1 {v25.4s}, [x0], #16\n"
        "ld1 {v10.4s}, [x0], #16\n"
        "ld1 {v18.4s}, [x0], #16\n"
        "ld1 {v26.4s}, [x0], #16\n"
        "ld1 {v11.4s}, [x0], #16\n"
        "ld1 {v19.4s}, [x0], #16\n"
        "ld1 {v27.4s}, [x0], #16\n"
        "ld1 {v12.4s}, [x0], #16\n"
        "ld1 {v20.4s}, [x0], #16\n"
        "ld1 {v28.4s}, [x0], #16\n"
        "ld1 {v13.4s}, [x0], #16\n"
        "ld1 {v21.4s}, [x0], #16\n"
        "ld1 {v29.4s}, [x0], #16\n"
        "ld1 {v14.4s}, [x0], #16\n"
        "ld1 {v22.4s}, [x0], #16\n"
        "ld1 {v30.4s}, [x0], #16\n"
        "ld1 {v15.4s}, [x0], #16\n"
        "ld1 {v23.4s}, [x0], #16\n"
        "ld1 {v31.4s}, [x0], #16\n"

        // The start of the loop assumes first Rhs cell is already loaded, so
        // do it here for first iteration.
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"

        // And the same for the first Lhs cell.
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Start the MACs at the head of the loop - 1st cell from each side
        // already loaded.
        "udot v8.4s, v2.16b, v0.b[0]\n"
        "udot v9.4s, v2.16b, v0.b[1]\n"
        "ld1 {v1.16b}, [%[rhs_ptr]], #16\n"  // Load second Rhs cell.
        "udot v10.4s, v2.16b, v0.b[2]\n"
        "udot v11.4s, v2.16b, v0.b[3]\n"
        "ld1 {v3.16b}, [%[lhs_ptr]], #16\n"  // Load second Lhs cell.
        "udot v12.4s, v2.16b, v1.b[0]\n"
        "udot v13.4s, v2.16b, v1.b[1]\n"
        "ld1 {v4.16b}, [%[lhs_ptr]], #16\n"  // Load third Lhs cell.
        "udot v14.4s, v2.16b, v1.b[2]\n"
        "udot v15.4s, v2.16b, v1.b[3]\n"
        "ld1 {v2.16b}, [%[lhs_ptr]], #16\n"  // Done with first Lhs cell - load
        // for the next iteration early.
        "udot v16.4s, v3.16b, v0.b[0]\n"
        "udot v17.4s, v3.16b, v0.b[1]\n"
        "udot v18.4s, v3.16b, v0.b[2]\n"
        "udot v19.4s, v3.16b, v0.b[3]\n"
        "udot v20.4s, v3.16b, v1.b[0]\n"
        "udot v21.4s, v3.16b, v1.b[1]\n"
        "udot v22.4s, v3.16b, v1.b[2]\n"
        "udot v23.4s, v3.16b, v1.b[3]\n"
        "udot v24.4s, v4.16b, v0.b[0]\n"
        "udot v25.4s, v4.16b, v0.b[1]\n"
        "udot v26.4s, v4.16b, v0.b[2]\n"
        "udot v27.4s, v4.16b, v0.b[3]\n"
        "ld1 {v0.16b}, [%[rhs_ptr]], #16\n"  // Done with the first Rhs cell -
        // load for the next iteration early.
        "udot v28.4s, v4.16b, v1.b[0]\n"
        "udot v29.4s, v4.16b, v1.b[1]\n"

        // Loop.  Decrement loop index (depth) by 4 as udot processes 4
        // depth values.
        "subs %w[depth], %w[depth], #4\n"
        "udot v30.4s, v4.16b, v1.b[2]\n"
        "udot v31.4s, v4.16b, v1.b[3]\n"

        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// As above, except tuned for Cortex-A55r1.
//
// Similarly, this is a clone of NEON_64bit_GEMM_Float32_WithScalar_A55r1
// with the names changed.
struct NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct_A55r1 {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 4, CellOrder::WidthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.4s}, [x0], #16\n"
        "ld1 {v16.4s}, [x0], #16\n"
        "ld1 {v24.4s}, [x0], #16\n"
        "ld1 {v9.4s}, [x0], #16\n"
        "ld1 {v17.4s}, [x0], #16\n"
        "ld1 {v25.4s}, [x0], #16\n"
        "ld1 {v10.4s}, [x0], #16\n"
        "ld1 {v18.4s}, [x0], #16\n"
        "ld1 {v26.4s}, [x0], #16\n"
        "ld1 {v11.4s}, [x0], #16\n"
        "ld1 {v19.4s}, [x0], #16\n"
        "ld1 {v27.4s}, [x0], #16\n"
        "ld1 {v12.4s}, [x0], #16\n"
        "ld1 {v20.4s}, [x0], #16\n"
        "ld1 {v28.4s}, [x0], #16\n"
        "ld1 {v13.4s}, [x0], #16\n"
        "ld1 {v21.4s}, [x0], #16\n"
        "ld1 {v29.4s}, [x0], #16\n"
        "ld1 {v14.4s}, [x0], #16\n"
        "ld1 {v22.4s}, [x0], #16\n"
        "ld1 {v30.4s}, [x0], #16\n"
        "ld1 {v15.4s}, [x0], #16\n"
        "ld1 {v23.4s}, [x0], #16\n"
        "ld1 {v31.4s}, [x0], #16\n"

        // For details on how this kernel works, see the Float32 kernel below.

        "ldr d0, [%[rhs_ptr]]\n"
        "ldr x18, [%[rhs_ptr], #8]\n"

        "ldr q2, [%[lhs_ptr]]\n"
        "ldr q3, [%[lhs_ptr], #16]\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        "udot v8.4s, v2.16b, v0.b[0]\n"
        "ldr d1, [%[rhs_ptr], #16]\n"         // Bottom half of v1
        "udot v9.4s, v2.16b, v0.b[1]\n"
        "ins v0.d[1], x18\n"                  // Finish loading v0
        "udot v16.4s, v3.16b, v0.b[0]\n"      // out of sequence - used to reduce load/use pressure.
        "ldr x18, [%[rhs_ptr], #24]\n"        // Top half of v1 to X register
        "udot v17.4s, v3.16b, v0.b[1]\n"      // out of sequence - used to reduce load/use pressure.
        "add %[rhs_ptr], %[rhs_ptr], #32\n"   // RHS loads complete - increment pointer.
        "udot v10.4s, v2.16b, v0.b[2]\n"
        "ldr d4, [%[lhs_ptr], #32]\n"         // Bottom half of v4
        "udot v11.4s, v2.16b, v0.b[3]\n"
        "ins v1.d[1], x18\n"                  // Finish loading v1
        "udot v12.4s, v2.16b, v1.b[0]\n"
        "ldr x18, [%[lhs_ptr], #40]\n"        // Top half of v4 to X register
        "udot v13.4s, v2.16b, v1.b[1]\n"
        "add %[lhs_ptr], %[lhs_ptr], #48\n"   // LHS loads complete - increment pointer.
        "udot v14.4s, v2.16b, v1.b[2]\n"

        "udot v15.4s, v2.16b, v1.b[3]\n"
        "ldr d2, [%[lhs_ptr]]\n"              // Bottom half of v2 (for next time)
        "udot v18.4s, v3.16b, v0.b[2]\n"
        "ins v4.d[1], x18\n"                  // Finish loading v4
        "udot v19.4s, v3.16b, v0.b[3]\n"
        "ldr x18, [%[lhs_ptr], #8]\n"         // Top half of next v2 to X register
        "udot v20.4s, v3.16b, v1.b[0]\n"
        "subs %w[depth], %w[depth], #4\n"
        "udot v21.4s, v3.16b, v1.b[1]\n"

        "udot v22.4s, v3.16b, v1.b[2]\n"

        "udot v23.4s, v3.16b, v1.b[3]\n"
        "ldr d3, [%[lhs_ptr], #16]\n"         // Bottom half of v3 (for next time)
        "udot v24.4s, v4.16b, v0.b[0]\n"
        "ins v2.d[1], x18\n"                  // Finish loading next v2
        "udot v25.4s, v4.16b, v0.b[1]\n"
        "ldr x18, [%[lhs_ptr], #24]\n"        // Top half of next v3 to X register
        "udot v26.4s, v4.16b, v0.b[2]\n"

        "udot v27.4s, v4.16b, v0.b[3]\n"
        "ldr d0, [%[rhs_ptr]]\n"              // Bottom half of v0 (for next time)
        "udot v28.4s, v4.16b, v1.b[0]\n"
        "ins v3.d[1], x18\n"                  // Finish loading next v3
        "udot v29.4s, v4.16b, v1.b[1]\n"
        "ldr x18, [%[rhs_ptr], #8]\n"         // Top half of next v0 to X register
        "udot v30.4s, v4.16b, v1.b[2]\n"

        "udot v31.4s, v4.16b, v1.b[3]\n"
        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.4s}, [x0], #16\n"
        "st1 {v16.4s}, [x0], #16\n"
        "st1 {v24.4s}, [x0], #16\n"
        "st1 {v9.4s}, [x0], #16\n"
        "st1 {v17.4s}, [x0], #16\n"
        "st1 {v25.4s}, [x0], #16\n"
        "st1 {v10.4s}, [x0], #16\n"
        "st1 {v18.4s}, [x0], #16\n"
        "st1 {v26.4s}, [x0], #16\n"
        "st1 {v11.4s}, [x0], #16\n"
        "st1 {v19.4s}, [x0], #16\n"
        "st1 {v27.4s}, [x0], #16\n"
        "st1 {v12.4s}, [x0], #16\n"
        "st1 {v20.4s}, [x0], #16\n"
        "st1 {v28.4s}, [x0], #16\n"
        "st1 {v13.4s}, [x0], #16\n"
        "st1 {v21.4s}, [x0], #16\n"
        "st1 {v29.4s}, [x0], #16\n"
        "st1 {v14.4s}, [x0], #16\n"
        "st1 {v22.4s}, [x0], #16\n"
        "st1 {v30.4s}, [x0], #16\n"
        "st1 {v15.4s}, [x0], #16\n"
        "st1 {v23.4s}, [x0], #16\n"
        "st1 {v31.4s}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "x18", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
  }
};
#endif  // __ARM_FEATURE_DOTPROD

// We don't actually use int32*int32 in production. This is just an
// experiment to help dissociate the effect of integer-vs-float, from the
// effect of operands width.
struct NEON_64bit_GEMM_Int32_WithScalar {
  typedef std::int32_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 2 Rhs cell of size 1x4 each
        "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[rhs_ptr]], #16\n"

        // Load 3 Lhs cells of size 4x1 each
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

        // Multiply-accumulate
        "mla v8.4s, v2.4s, v0.s[0]\n"
        "mla v9.4s, v2.4s, v0.s[1]\n"
        "mla v10.4s, v2.4s, v0.s[2]\n"
        "mla v11.4s, v2.4s, v0.s[3]\n"
        "mla v12.4s, v2.4s, v1.s[0]\n"
        "mla v13.4s, v2.4s, v1.s[1]\n"
        "mla v14.4s, v2.4s, v1.s[2]\n"
        "mla v15.4s, v2.4s, v1.s[3]\n"
        "mla v16.4s, v3.4s, v0.s[0]\n"
        "mla v17.4s, v3.4s, v0.s[1]\n"
        "mla v18.4s, v3.4s, v0.s[2]\n"
        "mla v19.4s, v3.4s, v0.s[3]\n"
        "mla v20.4s, v3.4s, v1.s[0]\n"
        "mla v21.4s, v3.4s, v1.s[1]\n"
        "mla v22.4s, v3.4s, v1.s[2]\n"
        "mla v23.4s, v3.4s, v1.s[3]\n"
        "mla v24.4s, v4.4s, v0.s[0]\n"
        "mla v25.4s, v4.4s, v0.s[1]\n"
        "mla v26.4s, v4.4s, v0.s[2]\n"
        "mla v27.4s, v4.4s, v0.s[3]\n"
        "mla v28.4s, v4.4s, v1.s[0]\n"
        "mla v29.4s, v4.4s, v1.s[1]\n"
        "mla v30.4s, v4.4s, v1.s[2]\n"
        "mla v31.4s, v4.4s, v1.s[3]\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %w[depth], %w[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// Not very efficient kernel, just an experiment to see what we can do
// without using NEON multiply-with-scalar instructions.
struct NEON_64bit_GEMM_Float32_WithVectorDuplicatingScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 2 Rhs cell of size 1x4 each
        "ld1 {v5.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v6.4s}, [%[rhs_ptr]], #16\n"

        // Load 3 Lhs cells of size 4x1 each
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

        // Multiply-accumulate
        "dup v0.4s, v5.s[0]\n"
        "dup v1.4s, v5.s[1]\n"
        "fmla v8.4s, v2.4s, v0.4s\n"
        "fmla v16.4s, v3.4s, v0.4s\n"
        "fmla v24.4s, v4.4s, v0.4s\n"
        "fmla v9.4s, v2.4s, v1.4s\n"
        "fmla v17.4s, v3.4s, v1.4s\n"
        "fmla v25.4s, v4.4s, v1.4s\n"
        "dup v0.4s, v5.s[2]\n"
        "dup v1.4s, v5.s[3]\n"
        "fmla v10.4s, v2.4s, v0.4s\n"
        "fmla v18.4s, v3.4s, v0.4s\n"
        "fmla v26.4s, v4.4s, v0.4s\n"
        "fmla v11.4s, v2.4s, v1.4s\n"
        "fmla v19.4s, v3.4s, v1.4s\n"
        "fmla v27.4s, v4.4s, v1.4s\n"
        "dup v0.4s, v6.s[0]\n"
        "dup v1.4s, v6.s[1]\n"
        "fmla v12.4s, v2.4s, v0.4s\n"
        "fmla v20.4s, v3.4s, v0.4s\n"
        "fmla v28.4s, v4.4s, v0.4s\n"
        "fmla v13.4s, v2.4s, v1.4s\n"
        "fmla v21.4s, v3.4s, v1.4s\n"
        "fmla v29.4s, v4.4s, v1.4s\n"
        "dup v0.4s, v6.s[2]\n"
        "dup v1.4s, v6.s[3]\n"
        "fmla v14.4s, v2.4s, v0.4s\n"
        "fmla v22.4s, v3.4s, v0.4s\n"
        "fmla v30.4s, v4.4s, v0.4s\n"
        "fmla v15.4s, v2.4s, v1.4s\n"
        "fmla v23.4s, v3.4s, v1.4s\n"
        "fmla v31.4s, v4.4s, v1.4s\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %w[depth], %w[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// This is the "most natural" kernel, using NEON multiply-with-scalar
// instructions.
struct NEON_64bit_GEMM_Float32_WithScalar {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Load 2 Rhs cell of size 1x4 each
        "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"
        "ld1 {v1.4s}, [%[rhs_ptr]], #16\n"

        // Load 3 Lhs cells of size 4x1 each
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"
        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"

        // Multiply-accumulate
        "fmla v8.4s, v2.4s, v0.s[0]\n"
        "fmla v9.4s, v2.4s, v0.s[1]\n"
        "fmla v10.4s, v2.4s, v0.s[2]\n"
        "fmla v11.4s, v2.4s, v0.s[3]\n"
        "fmla v12.4s, v2.4s, v1.s[0]\n"
        "fmla v13.4s, v2.4s, v1.s[1]\n"
        "fmla v14.4s, v2.4s, v1.s[2]\n"
        "fmla v15.4s, v2.4s, v1.s[3]\n"
        "fmla v16.4s, v3.4s, v0.s[0]\n"
        "fmla v17.4s, v3.4s, v0.s[1]\n"
        "fmla v18.4s, v3.4s, v0.s[2]\n"
        "fmla v19.4s, v3.4s, v0.s[3]\n"
        "fmla v20.4s, v3.4s, v1.s[0]\n"
        "fmla v21.4s, v3.4s, v1.s[1]\n"
        "fmla v22.4s, v3.4s, v1.s[2]\n"
        "fmla v23.4s, v3.4s, v1.s[3]\n"
        "fmla v24.4s, v4.4s, v0.s[0]\n"
        "fmla v25.4s, v4.4s, v0.s[1]\n"
        "fmla v26.4s, v4.4s, v0.s[2]\n"
        "fmla v27.4s, v4.4s, v0.s[3]\n"
        "fmla v28.4s, v4.4s, v1.s[0]\n"
        "fmla v29.4s, v4.4s, v1.s[1]\n"
        "fmla v30.4s, v4.4s, v1.s[2]\n"
        "fmla v31.4s, v4.4s, v1.s[3]\n"

        // Loop. Decrement loop index (depth) by 1, since we just handled 1
        // level of depth.
        "subs %w[depth], %w[depth], #1\n"
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

// Faster kernel contributed by ARM. Tuned for A57.
struct NEON_64bit_GEMM_Float32_WithScalar_A57 {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        // The start of the loop assumes first Rhs cell is already loaded, so
        // do it here for first iteration.
        "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"

        // And the same for the first Lhs cell.
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Start the MACs at the head of the loop - 1st cell from each side
        // already loaded.
        "fmla v8.4s, v2.4s, v0.s[0]\n"
        "fmla v9.4s, v2.4s, v0.s[1]\n"
        "ld1 {v1.4s}, [%[rhs_ptr]], #16\n"  // Load second Rhs cell.
        "fmla v10.4s, v2.4s, v0.s[2]\n"
        "fmla v11.4s, v2.4s, v0.s[3]\n"
        "ld1 {v3.4s}, [%[lhs_ptr]], #16\n"  // Load second Lhs cell.
        "fmla v12.4s, v2.4s, v1.s[0]\n"
        "fmla v13.4s, v2.4s, v1.s[1]\n"
        "ld1 {v4.4s}, [%[lhs_ptr]], #16\n"  // Load third Lhs cell.
        "fmla v14.4s, v2.4s, v1.s[2]\n"
        "fmla v15.4s, v2.4s, v1.s[3]\n"
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"  // Done with first Lhs cell - load
        // for the next iteration early.
        "fmla v16.4s, v3.4s, v0.s[0]\n"
        "fmla v17.4s, v3.4s, v0.s[1]\n"
        "fmla v18.4s, v3.4s, v0.s[2]\n"
        "fmla v19.4s, v3.4s, v0.s[3]\n"
        "fmla v20.4s, v3.4s, v1.s[0]\n"
        "fmla v21.4s, v3.4s, v1.s[1]\n"
        "fmla v22.4s, v3.4s, v1.s[2]\n"
        "fmla v23.4s, v3.4s, v1.s[3]\n"
        "fmla v24.4s, v4.4s, v0.s[0]\n"
        "fmla v25.4s, v4.4s, v0.s[1]\n"
        "fmla v26.4s, v4.4s, v0.s[2]\n"
        "fmla v27.4s, v4.4s, v0.s[3]\n"
        "ld1 {v0.4s}, [%[rhs_ptr]], #16\n"  // Done with the first Rhs cell -
        // load for the next iteration
        // early.
        "fmla v28.4s, v4.4s, v1.s[0]\n"
        "fmla v29.4s, v4.4s, v1.s[1]\n"
        // Loop. Decrement loop index (depth) by 1, since we just handled
        // 1 level of depth.  Do this a bit before the end of the loop for
        // better dispatch on A57.
        "subs %w[depth], %w[depth], #1\n"
        "fmla v30.4s, v4.4s, v1.s[2]\n"
        "fmla v31.4s, v4.4s, v1.s[3]\n"

        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29", "v30", "v31");
  }
};

#ifndef __APPLE__
// Faster kernel contributed by ARM. Tuned for A53.
struct NEON_64bit_GEMM_Float32_WithScalar_A53 {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.16b}, [x0], #16\n"
        "ld1 {v16.16b}, [x0], #16\n"
        "ld1 {v24.16b}, [x0], #16\n"
        "ld1 {v9.16b}, [x0], #16\n"
        "ld1 {v17.16b}, [x0], #16\n"
        "ld1 {v25.16b}, [x0], #16\n"
        "ld1 {v10.16b}, [x0], #16\n"
        "ld1 {v18.16b}, [x0], #16\n"
        "ld1 {v26.16b}, [x0], #16\n"
        "ld1 {v11.16b}, [x0], #16\n"
        "ld1 {v19.16b}, [x0], #16\n"
        "ld1 {v27.16b}, [x0], #16\n"
        "ld1 {v12.16b}, [x0], #16\n"
        "ld1 {v20.16b}, [x0], #16\n"
        "ld1 {v28.16b}, [x0], #16\n"
        "ld1 {v13.16b}, [x0], #16\n"
        "ld1 {v21.16b}, [x0], #16\n"
        "ld1 {v29.16b}, [x0], #16\n"
        "ld1 {v14.16b}, [x0], #16\n"
        "ld1 {v22.16b}, [x0], #16\n"
        "ld1 {v30.16b}, [x0], #16\n"
        "ld1 {v15.16b}, [x0], #16\n"
        "ld1 {v23.16b}, [x0], #16\n"
        "ld1 {v31.16b}, [x0], #16\n"

        // For A53, a very different-looking loop is needed.
        //
        // The main reason for this is that on A53 128-bit loads take two
        // cycles during which no dual issue can occur.  Doing two separate
        // 64-bit loads avoids this issue - they each take one cycle and are
        // able to dual issue.  Since vector register loads don't dual issue
        // with FMLA, we load half the register as normal and the other half
        // into an integer register.  This second half can then be moved into
        // place later with an INS instruction - which will dual issue with a
        // later FP load.
        //
        // For this kernel there are approximately 3 times as many multiplies
        // as loads, so it makes sense to structure the loop into blocks of 4
        // cycles, with 1 dedicated "load cycle" and 3 "multiply cycles" per
        // block.  Strictly preserving this structure with NOPs where no load
        // is needed seems to result in higher performance.
        //
        // Choice of x18 to store the upper halves on their way into the
        // vector registers is arbitrary.  Added to the clobber list so that
        // the compiler will make it available.
        //
        //
        // At the start of the loop, it is assumed that v0 is "half loaded" -
        // bottom half in place in d0 and the upper half in x18 ready to
        // insert.  So set that up here for the first iteration:
        "ldr d0, [%[rhs_ptr]]\n"             // Bottom half of first Rhs cell
        "ldr x18, [%[rhs_ptr], #8]\n"        // Upper half
        "add %[rhs_ptr], %[rhs_ptr], #16\n"  // Separate increment (needed as
        // there is no operation to load at
        // reg + 8 but then increment reg
        // by 16).

        // v2 should be fully loaded - as it's outside the loop proper it's fine
        // to use a 128-bit load here.
        "ld1 {v2.4s}, [%[lhs_ptr]], #16\n"  // first Lhs cell

        GEMMLOWP_LABEL_LOOP
        ":\n"

        // First block of four cycles.  Multplies all require v2 and v0; v2 is
        // loaded earlier and v0 is half loaded and completed in the load
        // cycle at the start.
        "ldr d1, [%[rhs_ptr]]\n"  // "load" cycle - loading bottom half of v1
        // (second Rhs cell).
        "ins v0.d[1], x18\n"  // "load" cycle - moving the upper half of v0 into
        // place.
        "fmla v8.4s, v2.4s, v0.s[0]\n"  // "fmla" cycle 1 - first multiply.
        "ldr x18, [%[rhs_ptr], #8]\n"  // "fmla" cycle 1 - load upper half of v1
        // into x18.
        "fmla v9.4s, v2.4s, v0.s[1]\n"       // "fmla" cycle 2 - second multiply
        "add %[rhs_ptr], %[rhs_ptr], #16\n"  // "fmla" cycle 2 - increment Rhs
        // pointer (if needed)
        "fmla v10.4s, v2.4s, v0.s[2]\n"  // "fmla" cycle 3 - third multiply.  No
        // more work to dual issue.

        // Second block.  Start loading v3 (second Lhs cell), finish loading v1.
        "ldr d3, [%[lhs_ptr]]\n"
        "ins v1.d[1], x18\n"  // v1 ready here.
        "fmla v11.4s, v2.4s, v0.s[3]\n"
        "ldr x18, [%[lhs_ptr], #8]\n"
        "fmla v12.4s, v2.4s, v1.s[0]\n"  // First use of v1.
        "add %[lhs_ptr], %[lhs_ptr], #16\n"
        "fmla v13.4s, v2.4s, v1.s[1]\n"

        // Third block.  Start loading v4 (third Lhs cell), finish loading v3.
        "ldr d4, [%[lhs_ptr]]\n"
        "ins v3.d[1], x18\n"  // v3 ready here.
        "fmla v14.4s, v2.4s, v1.s[2]\n"
        "ldr x18, [%[lhs_ptr], #8]\n"
        "fmla v15.4s, v2.4s, v1.s[3]\n"
        "add %[lhs_ptr], %[lhs_ptr], #16\n"
        "fmla v16.4s, v3.4s, v0.s[0]\n"  // First use of v3.

        // Fourth block.  v2 (first Lhs cell) is now finished with, so start
        // loading value for next iteration.  Finish loading v4.
        "ldr d2, [%[lhs_ptr]]\n"
        "ins v4.d[1], x18\n"  // v4 ready here.
        "fmla v17.4s, v3.4s, v0.s[1]\n"
        "ldr x18, [%[lhs_ptr], #8]\n"
        "fmla v18.4s, v3.4s, v0.s[2]\n"
        "add %[lhs_ptr], %[lhs_ptr], #16\n"
        "fmla v19.4s, v3.4s, v0.s[3]\n"

        // Fifth block, finish loading v2.  No new load to start as the other
        // registers are all still live.
        "ins v2.d[1], x18\n"
        "fmla v20.4s, v3.4s, v1.s[0]\n"
        "fmla v21.4s, v3.4s, v1.s[1]\n"
        "fmla v22.4s, v3.4s, v1.s[2]\n"

        // Sixth block, nothing to load.  2 nops needed as a single nop would
        // dual issue with the FMLA and break the timing.
        "nop\n"
        "nop\n"
        "fmla v23.4s, v3.4s, v1.s[3]\n"
        "fmla v24.4s, v4.4s, v0.s[0]\n"  // First use of v4.
        "fmla v25.4s, v4.4s, v0.s[1]\n"

        // Seventh block, nothing to load.  Decrement the loop counter in this
        // block as the last block is very full.
        "nop\n"
        "nop\n"
        "fmla v26.4s, v4.4s, v0.s[2]\n"
        "subs %w[depth], %w[depth], #1\n"
        "fmla v27.4s, v4.4s, v0.s[3]\n"
        "fmla v28.4s, v4.4s, v1.s[0]\n"

        // Eighth block - start loading v0 for next iteration.
        "ldr d0, [%[rhs_ptr]]\n"
        "fmla v29.4s, v4.4s, v1.s[1]\n"
        "ldr x18, [%[rhs_ptr], #8]\n"
        "fmla v30.4s, v4.4s, v1.s[2]\n"
        "add %[rhs_ptr], %[rhs_ptr], #16\n"
        "fmla v31.4s, v4.4s, v1.s[3]\n"

        // Loop branch.  This will dual issue in fmla cycle 3 of the 8th block.
        "bne " GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.16b}, [x0], #16\n"
        "st1 {v16.16b}, [x0], #16\n"
        "st1 {v24.16b}, [x0], #16\n"
        "st1 {v9.16b}, [x0], #16\n"
        "st1 {v17.16b}, [x0], #16\n"
        "st1 {v25.16b}, [x0], #16\n"
        "st1 {v10.16b}, [x0], #16\n"
        "st1 {v18.16b}, [x0], #16\n"
        "st1 {v26.16b}, [x0], #16\n"
        "st1 {v11.16b}, [x0], #16\n"
        "st1 {v19.16b}, [x0], #16\n"
        "st1 {v27.16b}, [x0], #16\n"
        "st1 {v12.16b}, [x0], #16\n"
        "st1 {v20.16b}, [x0], #16\n"
        "st1 {v28.16b}, [x0], #16\n"
        "st1 {v13.16b}, [x0], #16\n"
        "st1 {v21.16b}, [x0], #16\n"
        "st1 {v29.16b}, [x0], #16\n"
        "st1 {v14.16b}, [x0], #16\n"
        "st1 {v22.16b}, [x0], #16\n"
        "st1 {v30.16b}, [x0], #16\n"
        "st1 {v15.16b}, [x0], #16\n"
        "st1 {v23.16b}, [x0], #16\n"
        "st1 {v31.16b}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "x18", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
  }
};
#endif

// Faster kernel contributed by ARM. Tuned for A55r1.
struct NEON_64bit_GEMM_Float32_WithScalar_A55r1 {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "mov x0, %[accum_ptr]\n"
        "ld1 {v8.4s}, [x0], #16\n"
        "ld1 {v16.4s}, [x0], #16\n"
        "ld1 {v24.4s}, [x0], #16\n"
        "ld1 {v9.4s}, [x0], #16\n"
        "ld1 {v17.4s}, [x0], #16\n"
        "ld1 {v25.4s}, [x0], #16\n"
        "ld1 {v10.4s}, [x0], #16\n"
        "ld1 {v18.4s}, [x0], #16\n"
        "ld1 {v26.4s}, [x0], #16\n"
        "ld1 {v11.4s}, [x0], #16\n"
        "ld1 {v19.4s}, [x0], #16\n"
        "ld1 {v27.4s}, [x0], #16\n"
        "ld1 {v12.4s}, [x0], #16\n"
        "ld1 {v20.4s}, [x0], #16\n"
        "ld1 {v28.4s}, [x0], #16\n"
        "ld1 {v13.4s}, [x0], #16\n"
        "ld1 {v21.4s}, [x0], #16\n"
        "ld1 {v29.4s}, [x0], #16\n"
        "ld1 {v14.4s}, [x0], #16\n"
        "ld1 {v22.4s}, [x0], #16\n"
        "ld1 {v30.4s}, [x0], #16\n"
        "ld1 {v15.4s}, [x0], #16\n"
        "ld1 {v23.4s}, [x0], #16\n"
        "ld1 {v31.4s}, [x0], #16\n"

        // A55r1 requires a hybrid of the A53 and standard approaches.
        //
        // Like A53, this processor prefers 64-bit loads.
        //
        // Unlike A53, it is capable of dual-issuing a 64-bit vector load
        // (or INS) with a FMLA instruction.
        //
        // Therefore we aim to issue an FMLA instruction every cycle.
        // Alongside three FMLAs we can dual issue a (vector) 64-bit load, a
        // scalar 64-bit load and finally an INS to replicate the effect of
        // a single 128-bit load.
        //
        // The loop contains 24 FMLA instructions, and 5 vector registers
        // need to be loaded, consuming 15 dual issue slots.  This leaves 9
        // dual issue slots.  Four of these are used for loop housekeeping
        // (2 pointer adds, 1 counter update and 1 branch), leaving 5 left
        // over (marked by blank lines).
        //
        // Choice of x18 to store the upper halves on their way into the
        // vector registers is arbitrary.  Added to the clobber list so that
        // the compiler will make it available.


        // At the start of the loop, it is assumed that v0 is "half loaded" -
        // bottom half in place in d0 and the upper half in x18 ready to
        // insert.  So set that up here for the first iteration:
        "ldr d0, [%[rhs_ptr]]\n"             // Bottom half of first Rhs cell
        "ldr x18, [%[rhs_ptr], #8]\n"        // Upper half

        // v2-v3 should be fully loaded - as it's outside the loop proper it's fine
        // to use a 128-bit load here.
        "ldr q2, [%[lhs_ptr]]\n"      // first Lhs cell
        "ldr q3, [%[lhs_ptr], #16]\n" // second Lhs cell

        GEMMLOWP_LABEL_LOOP
        ":\n"

        "fmla v8.4s, v2.4s, v0.s[0]\n"
        "ldr d1, [%[rhs_ptr], #16]\n"         // Bottom half of v1
        "fmla v9.4s, v2.4s, v0.s[1]\n"
        "ins v0.d[1], x18\n"                  // Finish loading v0
        "fmla v16.4s, v3.4s, v0.s[0]\n"       // out of sequence - used to reduce load/use pressure.
        "ldr x18, [%[rhs_ptr], #24]\n"        // Top half of v1 to X register
        "fmla v17.4s, v3.4s, v0.s[1]\n"       // out of sequence - used to reduce load/use pressure.
        "add %[rhs_ptr], %[rhs_ptr], #32\n"   // RHS loads complete - increment pointer.
        "fmla v10.4s, v2.4s, v0.s[2]\n"
        "ldr d4, [%[lhs_ptr], #32]\n"         // Bottom half of v4
        "fmla v11.4s, v2.4s, v0.s[3]\n"
        "ins v1.d[1], x18\n"                  // Finish loading v1
        "fmla v12.4s, v2.4s, v1.s[0]\n"
        "ldr x18, [%[lhs_ptr], #40]\n"        // Top half of v4 to X register
        "fmla v13.4s, v2.4s, v1.s[1]\n"
        "add %[lhs_ptr], %[lhs_ptr], #48\n"   // LHS loads complete - increment pointer.
        "fmla v14.4s, v2.4s, v1.s[2]\n"

        "fmla v15.4s, v2.4s, v1.s[3]\n"
        "ldr d2, [%[lhs_ptr]]\n"              // Bottom half of v2 (for next time)
        "fmla v18.4s, v3.4s, v0.s[2]\n"
        "ins v4.d[1], x18\n"                  // Finish loading v4
        "fmla v19.4s, v3.4s, v0.s[3]\n"
        "ldr x18, [%[lhs_ptr], #8]\n"         // Top half of next v2 to X register
        "fmla v20.4s, v3.4s, v1.s[0]\n"
        "subs %w[depth], %w[depth], #1\n"
        "fmla v21.4s, v3.4s, v1.s[1]\n"

        "fmla v22.4s, v3.4s, v1.s[2]\n"

        "fmla v23.4s, v3.4s, v1.s[3]\n"
        "ldr d3, [%[lhs_ptr], #16]\n"         // Bottom half of v3 (for next time)
        "fmla v24.4s, v4.4s, v0.s[0]\n"
        "ins v2.d[1], x18\n"                  // Finish loading next v2
        "fmla v25.4s, v4.4s, v0.s[1]\n"
        "ldr x18, [%[lhs_ptr], #24]\n"        // Top half of next v3 to X register
        "fmla v26.4s, v4.4s, v0.s[2]\n"

        "fmla v27.4s, v4.4s, v0.s[3]\n"
        "ldr d0, [%[rhs_ptr]]\n"              // Bottom half of v0 (for next time)
        "fmla v28.4s, v4.4s, v1.s[0]\n"
        "ins v3.d[1], x18\n"                  // Finish loading next v3
        "fmla v29.4s, v4.4s, v1.s[1]\n"
        "ldr x18, [%[rhs_ptr], #8]\n"         // Top half of next v0 to X register
        "fmla v30.4s, v4.4s, v1.s[2]\n"

        "fmla v31.4s, v4.4s, v1.s[3]\n"
        "bne " GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators
        "mov x0, %[accum_ptr]\n"
        "st1 {v8.4s}, [x0], #16\n"
        "st1 {v16.4s}, [x0], #16\n"
        "st1 {v24.4s}, [x0], #16\n"
        "st1 {v9.4s}, [x0], #16\n"
        "st1 {v17.4s}, [x0], #16\n"
        "st1 {v25.4s}, [x0], #16\n"
        "st1 {v10.4s}, [x0], #16\n"
        "st1 {v18.4s}, [x0], #16\n"
        "st1 {v26.4s}, [x0], #16\n"
        "st1 {v11.4s}, [x0], #16\n"
        "st1 {v19.4s}, [x0], #16\n"
        "st1 {v27.4s}, [x0], #16\n"
        "st1 {v12.4s}, [x0], #16\n"
        "st1 {v20.4s}, [x0], #16\n"
        "st1 {v28.4s}, [x0], #16\n"
        "st1 {v13.4s}, [x0], #16\n"
        "st1 {v21.4s}, [x0], #16\n"
        "st1 {v29.4s}, [x0], #16\n"
        "st1 {v14.4s}, [x0], #16\n"
        "st1 {v22.4s}, [x0], #16\n"
        "st1 {v30.4s}, [x0], #16\n"
        "st1 {v15.4s}, [x0], #16\n"
        "st1 {v23.4s}, [x0], #16\n"
        "st1 {v31.4s}, [x0], #16\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "cc", "memory", "x0", "x18", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
  }
};

#endif  // __aarch64__

#if defined(__arm__) || defined(__aarch64__)
#ifndef __aarch64__
inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  const int32x2_t c = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
  const int32x2_t d = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
  return vcombine_s32(c, d);
}
#endif

// C++ intrinsics-based variant of the deep, int8, fast kernel
template <int Cols>
struct NEON_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<Cols, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    int32x4_t acc[4][Cols];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < Cols; j++) {
        acc[i][j] = vdupq_n_s32(0);
      }
    }
    for (int d = 0; d < depth; d += 16) {
      int8x16_t lhs[4];
      for (int i = 0; i < 4; i++) {
        lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
      }
      int8x16_t rhs[Cols];
      for (int i = 0; i < Cols; i++) {
        rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
      }
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < Cols; j++) {
          int16x8_t local_acc =
              vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
          local_acc =
              vmlal_s8(local_acc, vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
          acc[i][j] = vpadalq_s16(acc[i][j], local_acc);
        }
      }
      lhs_ptr += 64;
      rhs_ptr += 16 * Cols;
    }
    for (int i = 0; i < Cols; i++) {
      int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
      int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
      int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);
      int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
      dst_val = vaddq_s32(dst_val, acc_4x);
      vst1q_s32(accum_ptr + 4 * i, dst_val);
    }
  }
};

using NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics =
    NEON_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics<4>;

using NEON_32bit_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics =
    NEON_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics<2>;

// C++ intrinsics-based variant of the wide, uint8, general kernel
template <int RhsCells>
struct NEON_GEMM_Uint8Operands_Uint32Accumulators_intrinsics {
  typedef std::uint8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, RhsCells> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    int32x4_t acc[3][4 * RhsCells];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4 * RhsCells; j++) {
        acc[i][j] = vld1q_s32(accum_ptr + 4 * (i + 3 * j));
      }
    }
    for (int d = 0; d < depth; d += 2) {
      int16x8_t lhs[3];
      for (int i = 0; i < 3; i++) {
        lhs[i] = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(lhs_ptr + 8 * i)));
      }
      int16x8_t rhs[RhsCells];
      for (int i = 0; i < RhsCells; i++) {
        rhs[i] = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(rhs_ptr + 8 * i)));
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < RhsCells; j++) {
          acc[i][4 * j + 0] = vmlal_lane_s16(
              acc[i][4 * j + 0], vget_low_s16(lhs[i]), vget_low_s16(rhs[j]), 0);
          acc[i][4 * j + 1] = vmlal_lane_s16(
              acc[i][4 * j + 1], vget_low_s16(lhs[i]), vget_low_s16(rhs[j]), 1);
          acc[i][4 * j + 2] = vmlal_lane_s16(
              acc[i][4 * j + 2], vget_low_s16(lhs[i]), vget_low_s16(rhs[j]), 2);
          acc[i][4 * j + 3] = vmlal_lane_s16(
              acc[i][4 * j + 3], vget_low_s16(lhs[i]), vget_low_s16(rhs[j]), 3);
          acc[i][4 * j + 0] =
              vmlal_lane_s16(acc[i][4 * j + 0], vget_high_s16(lhs[i]),
                             vget_high_s16(rhs[j]), 0);
          acc[i][4 * j + 1] =
              vmlal_lane_s16(acc[i][4 * j + 1], vget_high_s16(lhs[i]),
                             vget_high_s16(rhs[j]), 1);
          acc[i][4 * j + 2] =
              vmlal_lane_s16(acc[i][4 * j + 2], vget_high_s16(lhs[i]),
                             vget_high_s16(rhs[j]), 2);
          acc[i][4 * j + 3] =
              vmlal_lane_s16(acc[i][4 * j + 3], vget_high_s16(lhs[i]),
                             vget_high_s16(rhs[j]), 3);
        }
      }
      lhs_ptr += 24;
      rhs_ptr += 8 * RhsCells;
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4 * RhsCells; j++) {
        vst1q_s32(accum_ptr + 4 * (i + 3 * j), acc[i][j]);
      }
    }
  }
};

using NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_intrinsics =
    NEON_GEMM_Uint8Operands_Uint32Accumulators_intrinsics<1>;

using NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_intrinsics =
    NEON_GEMM_Uint8Operands_Uint32Accumulators_intrinsics<2>;

template <int RhsCells>
struct NEON_GEMM_Float32_WithScalar_intrinsics {
  typedef float OperandType;
  typedef float AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 1, CellOrder::DepthMajor>, RhsCells> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    float32x4_t acc[3][4 * RhsCells];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4 * RhsCells; j++) {
        acc[i][j] = vld1q_f32(accum_ptr + 4 * (i + 3 * j));
      }
    }
    for (int d = 0; d < depth; d++) {
      float32x4_t lhs[3];
      for (int i = 0; i < 3; i++) {
        lhs[i] = vld1q_f32(lhs_ptr + 4 * i);
      }
      float32x4_t rhs[RhsCells];
      for (int i = 0; i < RhsCells; i++) {
        rhs[i] = vld1q_f32(rhs_ptr + 4 * i);
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < RhsCells; j++) {
          acc[i][4 * j + 0] = vmlaq_lane_f32(acc[i][4 * j + 0], lhs[i],
                                             vget_low_f32(rhs[j]), 0);
          acc[i][4 * j + 1] = vmlaq_lane_f32(acc[i][4 * j + 1], lhs[i],
                                             vget_low_f32(rhs[j]), 1);
          acc[i][4 * j + 2] = vmlaq_lane_f32(acc[i][4 * j + 2], lhs[i],
                                             vget_high_f32(rhs[j]), 0);
          acc[i][4 * j + 3] = vmlaq_lane_f32(acc[i][4 * j + 3], lhs[i],
                                             vget_high_f32(rhs[j]), 1);
        }
      }
      lhs_ptr += 12;
      rhs_ptr += 4 * RhsCells;
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4 * RhsCells; j++) {
        vst1q_f32(accum_ptr + 4 * (i + 3 * j), acc[i][j]);
      }
    }
  }
};

using NEON_32bit_GEMM_Float32_WithScalar_intrinsics =
    NEON_GEMM_Float32_WithScalar_intrinsics<1>;

using NEON_64bit_GEMM_Float32_WithScalar_intrinsics =
    NEON_GEMM_Float32_WithScalar_intrinsics<2>;

// C++ intrinsics-based variant of the deep, 7-bit, fast kernel
struct NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits_intrinsics {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 16, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<2, 16, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    int32x4_t acc[4][2];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        acc[i][j] = vdupq_n_s32(0);
      }
    }

    int d = 0;
    for (; d <= depth - 64; d += 64) {
      int16x8_t local_acc[4][2];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          local_acc[i][j] = vdupq_n_s16(0);
        }
      }

      // There are not enough registers to fit all lhs and rhs values for 64
      // depth. Instead, load values for 32 depth at a time.
      for (int k = 0; k < 2; k++) {
        int8x16_t lhs[4][2];
        for (int i = 0; i < 4; i++) {
          lhs[i][0] = vld1q_s8(lhs_ptr + 16 * i + 128 * k);
          lhs[i][1] = vld1q_s8(lhs_ptr + 64 + 16 * i + 128 * k);
        }

        int8x16_t rhs[4];
        for (int i = 0; i < 4; i++) {
          rhs[i] = vld1q_s8(rhs_ptr + 16 * i + 64 * k);
        }

        for (int i = 0; i < 4; i++) {
          if (k == 0) {
            local_acc[i][0] = vmull_s8(vget_low_s8(lhs[i][0]),
                                       vget_low_s8(rhs[0]));
            local_acc[i][0] = vmlal_s8(local_acc[i][0], vget_low_s8(lhs[i][1]),
                                       vget_low_s8(rhs[2]));
            local_acc[i][1] = vmull_s8(vget_low_s8(lhs[i][0]),
                                       vget_low_s8(rhs[1]));
            local_acc[i][1] = vmlal_s8(local_acc[i][1],
                                       vget_low_s8(lhs[i][1]),
                                       vget_low_s8(rhs[3]));
          } else {
            local_acc[i][0] = vmlal_s8(local_acc[i][0], vget_low_s8(lhs[i][0]),
                                       vget_low_s8(rhs[0]));
            local_acc[i][0] = vmlal_s8(local_acc[i][0], vget_low_s8(lhs[i][1]),
                                       vget_low_s8(rhs[2]));
            local_acc[i][1] = vmlal_s8(local_acc[i][1], vget_low_s8(lhs[i][0]),
                                       vget_low_s8(rhs[1]));
            local_acc[i][1] = vmlal_s8(local_acc[i][1], vget_low_s8(lhs[i][1]),
                                       vget_low_s8(rhs[3]));
          }

          local_acc[i][0] = vmlal_s8(local_acc[i][0], vget_high_s8(lhs[i][0]),
                                     vget_high_s8(rhs[0]));
          local_acc[i][0] = vmlal_s8(local_acc[i][0], vget_high_s8(lhs[i][1]),
                                     vget_high_s8(rhs[2]));
          local_acc[i][1] = vmlal_s8(local_acc[i][1], vget_high_s8(lhs[i][0]),
                                     vget_high_s8(rhs[1]));
          local_acc[i][1] = vmlal_s8(local_acc[i][1], vget_high_s8(lhs[i][1]),
                                     vget_high_s8(rhs[3]));
        }
      }

      for (int i = 0; i < 4; i++) {
        acc[i][0] = vpadalq_s16(acc[i][0], local_acc[i][0]);
        acc[i][1] = vpadalq_s16(acc[i][1], local_acc[i][1]);
      }

      lhs_ptr += 64 * 4;
      rhs_ptr += 64 * 2;
    }
    for (; d <= depth - 16; d += 16) {
      int8x16_t lhs[4];
      for (int i = 0; i < 4; i++) {
        lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
      }
      int8x16_t rhs[2];
      for (int i = 0; i < 2; i++) {
        rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
      }

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          int16x8_t local_acc =
              vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
          local_acc =
              vmlal_s8(local_acc, vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
          acc[i][j] = vpadalq_s16(acc[i][j], local_acc);
        }
      }
      lhs_ptr += 16 * 4;
      rhs_ptr += 16 * 2;
    }
    for (int i = 0; i < 2; i++) {
      int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
      int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
      int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);
      int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
      dst_val = vaddq_s32(dst_val, acc_4x);
      vst1q_s32(accum_ptr + 4 * i, dst_val);
    }
  }
};

SET_7BIT_RANGES(NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits_intrinsics);

// C++ intrinsics-based variant of the deep, 4.25-bit, fast kernel.
struct NEON_64bit_GEMM_Int425Operands_intrinsics {
  typedef std::int8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 32, CellOrder::WidthMajor>, 1>,
      KernelSideFormat<CellFormat<2, 32, CellOrder::WidthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    int32x4_t acc[4][2];
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        acc[i][j] = vdupq_n_s32(0);
      }
    }

    const int num_outer_depth_loop = depth / 512 + 1;
    int d = 0;
    for (int od = 0; od < num_outer_depth_loop; od++) {
      int16x8_t local_acc[4][2];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          local_acc[i][j] = vdupq_n_s16(0);
        }
      }
      for (int k = 0; k < 16 && d <= depth - 32; k++, d += 32) {
        int8x16_t lhs[8];
        for (int i = 0; i < 8; i++) {
          lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
        }
        int8x16_t rhs[4];
        for (int i = 0; i < 4; i++) {
          rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
        }
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 2; j++) {
            int8x16_t temp_acc = vmulq_s8(lhs[i * 2], rhs[j * 2]);
            temp_acc = vmlaq_s8(temp_acc, lhs[i * 2 + 1], rhs[j * 2 + 1]);
            local_acc[i][j] = vpadalq_s8(local_acc[i][j], temp_acc);
          }
        }
        lhs_ptr += 128;
        rhs_ptr += 64;
      }

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
          acc[i][j] = vpadalq_s16(acc[i][j], local_acc[i][j]);
        }
      }
    }

    for (int i = 0; i < 2; i++) {
      int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
      int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
      int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);

      int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
      dst_val = vaddq_s32(dst_val, acc_4x);
      vst1q_s32(accum_ptr + 4 * i, dst_val);
    }
  }
};

SET_425BIT_RANGES(NEON_64bit_GEMM_Int425Operands_intrinsics);

#endif  // __arm__ || __aarch64__

#ifdef __mips
static inline v4i32 workaround_msa_maddv_w(v4i32 a, v4i32 b, v4i32 c) {
  // Workaround for incorrect encoding of maddv.df in gcc (a exchanged with c).
#if 0
  return __builtin_msa_maddv_w(a, b, c);
#else
  asm volatile("maddv.w %w[a], %w[b], %w[c]\n"
               // Outputs
               : [a] "+f"(a)
               // Inputs
               : [b] "f"(b), [c] "f"(c));
  return a;
#endif
}

// Using 32x32=32 multiplications.
// 20 MSA regs used:
// - 12 accumulators
// - 6 lhs
// - 1 rhs
// - 1 temps/zeroes
// ~55 instructions in the loop.
struct MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_intrinsics {
  typedef std::uint8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    const v16i8 zeroes = __builtin_msa_ldi_b(0);
    v4i32 acc[3][4];
    // Load accumulators.
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        acc[i][j] = __builtin_msa_ld_w(accum_ptr + 4 * (i + 3 * j), 0);
      }
    }

    while (depth > 0) {
      // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
      v8i16 lhs[6];
      lhs[0] = reinterpret_cast<v8i16>(__builtin_msa_ld_b(const_cast<OperandType*>(lhs_ptr), 0));
      lhs[1] =
          reinterpret_cast<v8i16>(__builtin_msa_ld_b(const_cast<OperandType*>(lhs_ptr + 8), 0));

      // Zero-extend 8-bit elements of lhs[] to 16 bits.
      lhs[0] = reinterpret_cast<v8i16>(__builtin_msa_ilvr_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[0])));
      lhs[2] = reinterpret_cast<v8i16>(__builtin_msa_ilvl_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[1])));
      lhs[1] = reinterpret_cast<v8i16>(__builtin_msa_ilvr_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[1])));

      // Zero-extend 16-bit elements of lhs[] to 32 bits.
      lhs[3] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[0]);
      lhs[4] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[1]);
      lhs[5] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[2]);
      lhs[0] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[0]);
      lhs[1] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[1]);
      lhs[2] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[2]);

      // Depth 0.
      for (int j = 0; j < 4; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i]), rhs);
        }
      }

      // Depth 1.
      for (int j = 0; j < 4; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j + 4]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i + 3]), rhs);
        }
      }

      lhs_ptr += 24;
      rhs_ptr += 8;
      depth -= 2;
    }

    // Store accumulators.
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        __builtin_msa_st_w(acc[i][j], accum_ptr + 4 * (i + 3 * j), 0);
      }
    }
  }
};

// Assembly implementation of the above
// MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_intrinsics.
// Using 32x32=32 multiplications.
// 20 MSA regs used:
// - 12 accumulators
// - 6 lhs
// - 1 rhs
// - 1 temps/zeroes
// ~55 instructions in the loop.
struct MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_assembly {
  typedef std::uint8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(OperandType* lhs_ptr, OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "ld.w   $w0,   (0*16)(%[accum_ptr])\n"
        "ld.w   $w4,   (1*16)(%[accum_ptr])\n"
        "ld.w   $w8,   (2*16)(%[accum_ptr])\n"
        "ld.w   $w1,   (3*16)(%[accum_ptr])\n"
        "ld.w   $w5,   (4*16)(%[accum_ptr])\n"
        "ld.w   $w9,   (5*16)(%[accum_ptr])\n"
        "ld.w   $w2,   (6*16)(%[accum_ptr])\n"
        "ld.w   $w6,   (7*16)(%[accum_ptr])\n"
        "ld.w   $w10,  (8*16)(%[accum_ptr])\n"
        "ld.w   $w3,   (9*16)(%[accum_ptr])\n"
        "ld.w   $w7,  (10*16)(%[accum_ptr])\n"
        "ld.w   $w11, (11*16)(%[accum_ptr])\n"
        // Set a temp to all zeroes.
        "ldi.b  $w19, 0\n"

        GEMMLOWP_LABEL_LOOP ":\n"
        // Overview of register layout:
        //
        // A half of the 2x4 cell of Rhs is stored in 32bit in w18.
        // A 12x2 block of 3 4x2 cells Lhs is stored in 32bit in w12-w17.
        // A 12x4 block of accumulators is stored in 32bit in w0-w11.
        //
        //                    +------+------+------+------+
        //               Rhs  |w18[0]|w18[1]|w18[2]|w18[3]|
        //                    +------+------+------+------+
        //
        //                    |      |      |      |      |
        //
        //    Lhs             |      |      |      |      |
        //
        //  +---+---+ - - - - +------+------+------+------+
        //  |w12|w15|         | w0   | w1   | w2   | w3   |
        //  |w12|w15|         | w0   | w1   | w2   | w3   |
        //  |w12|w15|         | w0   | w1   | w2   | w3   |
        //  |w12|w15|         | w0   | w1   | w2   | w3   |
        //  +---+---+ - - - - +------+------+------+------+
        //  |w13|w16|         | w4   | w5   | w6   | w7   |
        //  |w13|w16|         | w4   | w5   | w6   | w7   |
        //  |w13|w16|         | w4   | w5   | w6   | w7   |
        //  |w13|w16|         | w4   | w5   | w6   | w7   |
        //  +---+---+ - - - - +------+------+------+------+
        //  |w14|w17|         | w8   | w9   | w10  | w11  |
        //  |w14|w17|         | w8   | w9   | w10  | w11  |
        //  |w14|w17|         | w8   | w9   | w10  | w11  |
        //  |w14|w17|         | w8   | w9   | w10  | w11  |
        //  +---+---+ - - - - +------+------+------+------+
        //
        //                            Accumulator

        // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
        "ld.b   $w12, 0(%[lhs_ptr])\n"
        "ld.b   $w13, 8(%[lhs_ptr])\n"

        // Load 4 bytes of rhs[] for depth 0.
        "lbu    $a0, 0(%[rhs_ptr])\n"
        "lbu    $a1, 1(%[rhs_ptr])\n"
        "lbu    $a2, 2(%[rhs_ptr])\n"
        "lbu    $a3, 3(%[rhs_ptr])\n"

        // Zero-extend 8-bit elements of lhs[] to 16 bits.
        "ilvr.b $w12, $w19, $w12\n"
        "ilvl.b $w14, $w19, $w13\n"
        "ilvr.b $w13, $w19, $w13\n"
        // Zero-extend 16-bit elements of lhs[] to 32 bits.
        "ilvl.h $w15, $w19, $w12\n"
        "ilvl.h $w16, $w19, $w13\n"
        "ilvl.h $w17, $w19, $w14\n"
        "ilvr.h $w12, $w19, $w12\n"
        "ilvr.h $w13, $w19, $w13\n"
        "ilvr.h $w14, $w19, $w14\n"

        // Depth 0.
        "fill.w  $w18, $a0\n"
        "lbu     $a0, 4(%[rhs_ptr])\n"
        "maddv.w $w0, $w12, $w18\n"
        "maddv.w $w4, $w13, $w18\n"
        "maddv.w $w8, $w14, $w18\n"
        "fill.w  $w18, $a1\n"
        "lbu     $a1, 5(%[rhs_ptr])\n"
        "maddv.w $w1, $w12, $w18\n"
        "maddv.w $w5, $w13, $w18\n"
        "maddv.w $w9, $w14, $w18\n"
        "fill.w  $w18, $a2\n"
        "lbu     $a2, 6(%[rhs_ptr])\n"
        "maddv.w $w2, $w12, $w18\n"
        "maddv.w $w6, $w13, $w18\n"
        "maddv.w $w10, $w14, $w18\n"
        "fill.w  $w18, $a3\n"
        "lbu     $a3, 7(%[rhs_ptr])\n"
        "maddv.w $w3, $w12, $w18\n"
        "maddv.w $w7, $w13, $w18\n"
        "maddv.w $w11, $w14, $w18\n"

        // Depth 1.
        "fill.w  $w18, $a0\n"
        "maddv.w $w0, $w15, $w18\n"
        "maddv.w $w4, $w16, $w18\n"
        "maddv.w $w8, $w17, $w18\n"
        "fill.w  $w18, $a1\n"
        "maddv.w $w1, $w15, $w18\n"
        "maddv.w $w5, $w16, $w18\n"
        "maddv.w $w9, $w17, $w18\n"
        "fill.w  $w18, $a2\n"
        "maddv.w $w2, $w15, $w18\n"
        "maddv.w $w6, $w16, $w18\n"
        "maddv.w $w10, $w17, $w18\n"
        "fill.w  $w18, $a3\n"
        "maddv.w $w3, $w15, $w18\n"
        "maddv.w $w7, $w16, $w18\n"
        "maddv.w $w11, $w17, $w18\n"

        "addiu  %[depth], -2\n"
        GEMMLOWP_MIPS_XADDIU " %[lhs_ptr], 24\n"
        GEMMLOWP_MIPS_XADDIU " %[rhs_ptr], 8\n"
        "bnez   %[depth]," GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators.
        "st.w   $w0,   (0*16)(%[accum_ptr])\n"
        "st.w   $w4,   (1*16)(%[accum_ptr])\n"
        "st.w   $w8,   (2*16)(%[accum_ptr])\n"
        "st.w   $w1,   (3*16)(%[accum_ptr])\n"
        "st.w   $w5,   (4*16)(%[accum_ptr])\n"
        "st.w   $w9,   (5*16)(%[accum_ptr])\n"
        "st.w   $w2,   (6*16)(%[accum_ptr])\n"
        "st.w   $w6,   (7*16)(%[accum_ptr])\n"
        "st.w   $w10,  (8*16)(%[accum_ptr])\n"
        "st.w   $w3,   (9*16)(%[accum_ptr])\n"
        "st.w   $w7,  (10*16)(%[accum_ptr])\n"
        "st.w   $w11, (11*16)(%[accum_ptr])\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "memory",
        "a0", "a1", "a2", "a3",
        "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7",
        "$f8", "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
        "$f16", "$f17", "$f18", "$f19");
  }
};

// Assembly implementation of the above
// MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_intrinsics2 (TODO).
// Using 16x16=32 multiplications.
// 20 MSA regs used:
// - 12 accumulators
// - 3 lhs
// - 4 rhs
// - 1 temps/zeroes
// ~45 instructions in the loop.
struct MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_assembly2 {
  typedef std::uint8_t OperandType;
  typedef std::int32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 1> >
      Format;
  static void Run(OperandType* lhs_ptr, OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "ld.w   $w0,   (0*16)(%[accum_ptr])\n"
        "ld.w   $w4,   (1*16)(%[accum_ptr])\n"
        "ld.w   $w8,   (2*16)(%[accum_ptr])\n"
        "ld.w   $w1,   (3*16)(%[accum_ptr])\n"
        "ld.w   $w5,   (4*16)(%[accum_ptr])\n"
        "ld.w   $w9,   (5*16)(%[accum_ptr])\n"
        "ld.w   $w2,   (6*16)(%[accum_ptr])\n"
        "ld.w   $w6,   (7*16)(%[accum_ptr])\n"
        "ld.w   $w10,  (8*16)(%[accum_ptr])\n"
        "ld.w   $w3,   (9*16)(%[accum_ptr])\n"
        "ld.w   $w7,  (10*16)(%[accum_ptr])\n"
        "ld.w   $w11, (11*16)(%[accum_ptr])\n"
        // Set a temp to all zeroes.
        "ldi.b  $w19, 0\n"

        GEMMLOWP_LABEL_LOOP ":\n"
        // Overview of register layout:
        //
        // A 2x4 cell of Rhs is stored in 16bit in w15-w18 (each register
        // contains 4 replicas of a pair of elements).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in w12-w14.
        // A 12x4 block of accumulators is stored in 32bit in w0-w11.
        //
        //                    +-----+-----+-----+-----+
        //               Rhs  | w15 | w16 | w17 | w18 |
        //                    +-----+-----+-----+-----+
        //
        //                    |     |     |     |     |
        //
        //       Lhs          |     |     |     |     |
        //
        //      +---+ - - - - +-----+-----+-----+-----+
        //      |w12|         | w0  | w1  | w2  | w3  |
        //      |w12|         | w0  | w1  | w2  | w3  |
        //      |w12|         | w0  | w1  | w2  | w3  |
        //      |w12|         | w0  | w1  | w2  | w3  |
        //      +---+ - - - - +-----+-----+-----+-----+
        //      |w13|         | w4  | w5  | w6  | w7  |
        //      |w13|         | w4  | w5  | w6  | w7  |
        //      |w13|         | w4  | w5  | w6  | w7  |
        //      |w13|         | w4  | w5  | w6  | w7  |
        //      +---+ - - - - +-----+-----+-----+-----+
        //      |w14|         | w8  | w9  | w10 | w11 |
        //      |w14|         | w8  | w9  | w10 | w11 |
        //      |w14|         | w8  | w9  | w10 | w11 |
        //      |w14|         | w8  | w9  | w10 | w11 |
        //      +---+ - - - - +-----+-----+-----+-----+
        //
        //                           Accumulators

        // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
        "ld.b   $w12, 0(%[lhs_ptr])\n"
        "ld.b   $w13, 8(%[lhs_ptr])\n"

        // Load 4 bytes of rhs[] for depth 0.
        "lbu    $a0, 0(%[rhs_ptr])\n"
        "lbu    $a1, 1(%[rhs_ptr])\n"
        "lbu    $a2, 2(%[rhs_ptr])\n"
        "lbu    $a3, 3(%[rhs_ptr])\n"
        // Load 4 bytes of rhs[] for depth 1.
        "lbu    $v0, 4(%[rhs_ptr])\n"
        "lbu    $v1, 5(%[rhs_ptr])\n"
        "lbu    $t8, 6(%[rhs_ptr])\n"
        "lbu    $t9, 7(%[rhs_ptr])\n"

        // Zero-extend 8-bit elements of lhs[] to 16 bits.
        "ilvr.b $w12, $w19, $w12\n"
        "ilvl.b $w14, $w19, $w13\n"
        "ilvr.b $w13, $w19, $w13\n"
        // Interleave depth 0 and depth 1 elements of lhs[] for dpadd_u.w.
        "ilvl.d $w15, $w19, $w12\n"
        "ilvl.d $w16, $w19, $w13\n"
        "ilvl.d $w17, $w19, $w14\n"
        "ilvr.h $w12, $w15, $w12\n"
        "ilvr.h $w13, $w16, $w13\n"
        "ilvr.h $w14, $w17, $w14\n"

        // Combine and interleave depth 0 and depth 1 elements of rhs[] for dpadd_u.w.
        "ins    $a0, $v0, 16, 8\n"
        "ins    $a1, $v1, 16, 8\n"
        "ins    $a2, $t8, 16, 8\n"
        "ins    $a3, $t9, 16, 8\n"
        // Make 4 replicas of every pair of rhs[] elements.
        "fill.w $w15, $a0\n"
        "fill.w $w16, $a1\n"
        "fill.w $w17, $a2\n"
        "fill.w $w18, $a3\n"

        // Depths 0 and 1.
        // Dot-product-(and)-add doubles multiplicand width.
        "dpadd_u.w $w0, $w12, $w15\n"
        "dpadd_u.w $w4, $w13, $w15\n"
        "dpadd_u.w $w8, $w14, $w15\n"
        "dpadd_u.w $w1, $w12, $w16\n"
        "dpadd_u.w $w5, $w13, $w16\n"
        "dpadd_u.w $w9, $w14, $w16\n"
        "dpadd_u.w $w2, $w12, $w17\n"
        "dpadd_u.w $w6, $w13, $w17\n"
        "dpadd_u.w $w10, $w14, $w17\n"
        "dpadd_u.w $w3, $w12, $w18\n"
        "dpadd_u.w $w7, $w13, $w18\n"
        "dpadd_u.w $w11, $w14, $w18\n"

        "addiu  %[depth], -2\n"
        GEMMLOWP_MIPS_XADDIU " %[lhs_ptr], 24\n"
        GEMMLOWP_MIPS_XADDIU " %[rhs_ptr], 8\n"
        "bnez   %[depth]," GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators.
        "st.w   $w0,   (0*16)(%[accum_ptr])\n"
        "st.w   $w4,   (1*16)(%[accum_ptr])\n"
        "st.w   $w8,   (2*16)(%[accum_ptr])\n"
        "st.w   $w1,   (3*16)(%[accum_ptr])\n"
        "st.w   $w5,   (4*16)(%[accum_ptr])\n"
        "st.w   $w9,   (5*16)(%[accum_ptr])\n"
        "st.w   $w2,   (6*16)(%[accum_ptr])\n"
        "st.w   $w6,   (7*16)(%[accum_ptr])\n"
        "st.w   $w10,  (8*16)(%[accum_ptr])\n"
        "st.w   $w3,   (9*16)(%[accum_ptr])\n"
        "st.w   $w7,  (10*16)(%[accum_ptr])\n"
        "st.w   $w11, (11*16)(%[accum_ptr])\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "memory",
        "v0", "v1",
        "a0", "a1", "a2", "a3",
        "t8", "t9",
        "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7",
        "$f8", "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
        "$f16", "$f17", "$f18", "$f19");
  }
};

// Using 32x32=32 multiplications.
// 32 MSA regs used:
// - 24 accumulators
// - 6 lhs
// - 1 rhs
// - 1 temps/zeroes
// ~95 instructions in the loop.
struct MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_intrinsics {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    const v16i8 zeroes = __builtin_msa_ldi_b(0);
    v4i32 acc[3][8];
    // Load accumulators.
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 8; j++) {
        acc[i][j] = __builtin_msa_ld_w(accum_ptr + 4 * (i + 3 * j), 0);
      }
    }

    while (depth > 0) {
      // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
      v8i16 lhs[6];
      lhs[0] = reinterpret_cast<v8i16>(__builtin_msa_ld_b(const_cast<OperandType*>(lhs_ptr), 0));
      lhs[1] =
          reinterpret_cast<v8i16>(__builtin_msa_ld_b(const_cast<OperandType*>(lhs_ptr + 8), 0));

      // Zero-extend 8-bit elements of lhs[] to 16 bits.
      lhs[0] = reinterpret_cast<v8i16>(__builtin_msa_ilvr_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[0])));
      lhs[2] = reinterpret_cast<v8i16>(__builtin_msa_ilvl_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[1])));
      lhs[1] = reinterpret_cast<v8i16>(__builtin_msa_ilvr_b(zeroes,
                                                            reinterpret_cast<v16i8>(lhs[1])));

      // Zero-extend 16-bit elements of lhs[] to 32 bits.
      lhs[3] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[0]);
      lhs[4] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[1]);
      lhs[5] = __builtin_msa_ilvl_h(reinterpret_cast<v8i16>(zeroes), lhs[2]);
      lhs[0] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[0]);
      lhs[1] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[1]);
      lhs[2] = __builtin_msa_ilvr_h(reinterpret_cast<v8i16>(zeroes), lhs[2]);

      // Depth 0.
      for (int j = 0; j < 4; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i]), rhs);
        }
      }
      for (int j = 4; j < 8; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j + 4]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i]), rhs);
        }
      }

      // Depth 1.
      for (int j = 0; j < 4; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j + 4]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i + 3]), rhs);
        }
      }
      for (int j = 4; j < 8; j++) {
        // Load 1 byte of rhs, making 4 32-bit replicas of it.
        v4i32 rhs = reinterpret_cast<v4i32>(__builtin_msa_fill_w(rhs_ptr[j + 8]));
        // Multiply-add into accumulators.
        for (int i = 0; i < 3; i++) {
          acc[i][j] = workaround_msa_maddv_w(
              acc[i][j], reinterpret_cast<v4i32>(lhs[i + 3]), rhs);
        }
      }

      lhs_ptr += 24;
      rhs_ptr += 16;
      depth -= 2;
    }

    // Store accumulators.
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 8; j++) {
        __builtin_msa_st_w(acc[i][j], accum_ptr + 4 * (i + 3 * j), 0);
      }
    }
  }
};

// Assembly implementation of the above
// MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_intrinsics.
// Using 32x32=32 multiplications.
// 32 MSA regs used:
// - 24 accumulators
// - 6 lhs
// - 1 rhs
// - 1 temps/zeroes
// ~95 instructions in the loop.
struct MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(OperandType* lhs_ptr, OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "ld.w   $w0,   (0*16)(%[accum_ptr])\n"
        "ld.w   $w4,   (1*16)(%[accum_ptr])\n"
        "ld.w   $w8,   (2*16)(%[accum_ptr])\n"
        "ld.w   $w1,   (3*16)(%[accum_ptr])\n"
        "ld.w   $w5,   (4*16)(%[accum_ptr])\n"
        "ld.w   $w9,   (5*16)(%[accum_ptr])\n"
        "ld.w   $w2,   (6*16)(%[accum_ptr])\n"
        "ld.w   $w6,   (7*16)(%[accum_ptr])\n"
        "ld.w   $w10,  (8*16)(%[accum_ptr])\n"
        "ld.w   $w3,   (9*16)(%[accum_ptr])\n"
        "ld.w   $w7,  (10*16)(%[accum_ptr])\n"
        "ld.w   $w11, (11*16)(%[accum_ptr])\n"
        "ld.w   $w12, (12*16)(%[accum_ptr])\n"
        "ld.w   $w16, (13*16)(%[accum_ptr])\n"
        "ld.w   $w20, (14*16)(%[accum_ptr])\n"
        "ld.w   $w13, (15*16)(%[accum_ptr])\n"
        "ld.w   $w17, (16*16)(%[accum_ptr])\n"
        "ld.w   $w21, (17*16)(%[accum_ptr])\n"
        "ld.w   $w14, (18*16)(%[accum_ptr])\n"
        "ld.w   $w18, (19*16)(%[accum_ptr])\n"
        "ld.w   $w22, (20*16)(%[accum_ptr])\n"
        "ld.w   $w15, (21*16)(%[accum_ptr])\n"
        "ld.w   $w19, (22*16)(%[accum_ptr])\n"
        "ld.w   $w23, (23*16)(%[accum_ptr])\n"
        // Set a temp to all zeroes.
        "ldi.b  $w31, 0\n"

        GEMMLOWP_LABEL_LOOP ":\n"
        // Overview of register layout:
        //
        // A quarter of the 2 2x4 cells of Rhs is stored in 32bit in w30.
        // A 12x2 block of 3 4x2 cells Lhs is stored in 32bit in w24-w29.
        // A 12x8 block of accumulators is stored in 32bit in w0-w23.
        //
        //                    +------+------+------+------+
        //               Rhs  |w30[0]|w30[1]|w30[2]|w30[3]|
        //                    +------+------+------+------+
        //
        //                    |      |      |      |      |
        //
        //    Lhs             |      |      |      |      |
        //
        //  +---+---+ - - - - +------+------+------+------+
        //  |w24|w27|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //  |w24|w27|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //  |w24|w27|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //  |w24|w27|         |w0/12 |w1/13 |w2/14 |w3/15 |
        //  +---+---+ - - - - +------+------+------+------+
        //  |w25|w28|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //  |w25|w28|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //  |w25|w28|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //  |w25|w28|         |w4/16 |w5/17 |w6/18 |w7/19 |
        //  +---+---+ - - - - +------+------+------+------+
        //  |w26|w29|         |w8/20 |w9/21 |w10/22|w11/23|
        //  |w26|w29|         |w8/20 |w9/21 |w10/22|w11/23|
        //  |w26|w29|         |w8/20 |w9/21 |w10/22|w11/23|
        //  |w26|w29|         |w8/20 |w9/21 |w10/22|w11/23|
        //  +---+---+ - - - - +------+------+------+------+
        //
        //                            Accumulator

        // Load 3 x 8 bytes of lhs[] with 2 16-byte overlapped loads.
        "ld.b   $w24, 0(%[lhs_ptr])\n"
        "ld.b   $w25, 8(%[lhs_ptr])\n"

        // Load 4 bytes of rhs[] for the first half of depth 0.
        "lbu    $a0, 0(%[rhs_ptr])\n"
        "lbu    $a1, 1(%[rhs_ptr])\n"
        "lbu    $a2, 2(%[rhs_ptr])\n"
        "lbu    $a3, 3(%[rhs_ptr])\n"

        // Zero-extend 8-bit elements of lhs[] to 16 bits.
        "ilvr.b $w24, $w31, $w24\n"
        "ilvl.b $w26, $w31, $w25\n"
        "ilvr.b $w25, $w31, $w25\n"
        // Zero-extend 16-bit elements of lhs[] to 32 bits.
        "ilvl.h $w27, $w31, $w24\n"
        "ilvl.h $w28, $w31, $w25\n"
        "ilvl.h $w29, $w31, $w26\n"
        "ilvr.h $w24, $w31, $w24\n"
        "ilvr.h $w25, $w31, $w25\n"
        "ilvr.h $w26, $w31, $w26\n"

        // Depth 0.
        "fill.w  $w30, $a0\n"
        "lbu     $a0, 8(%[rhs_ptr])\n"
        "maddv.w $w0, $w24, $w30\n"
        "maddv.w $w4, $w25, $w30\n"
        "maddv.w $w8, $w26, $w30\n"
        "fill.w  $w30, $a1\n"
        "lbu     $a1, 9(%[rhs_ptr])\n"
        "maddv.w $w1, $w24, $w30\n"
        "maddv.w $w5, $w25, $w30\n"
        "maddv.w $w9, $w26, $w30\n"
        "fill.w  $w30, $a2\n"
        "lbu     $a2, 10(%[rhs_ptr])\n"
        "maddv.w $w2, $w24, $w30\n"
        "maddv.w $w6, $w25, $w30\n"
        "maddv.w $w10, $w26, $w30\n"
        "fill.w  $w30, $a3\n"
        "lbu     $a3, 11(%[rhs_ptr])\n"
        "maddv.w $w3, $w24, $w30\n"
        "maddv.w $w7, $w25, $w30\n"
        "maddv.w $w11, $w26, $w30\n"

        "fill.w  $w30, $a0\n"
        "lbu     $a0, 4(%[rhs_ptr])\n"
        "maddv.w $w12, $w24, $w30\n"
        "maddv.w $w16, $w25, $w30\n"
        "maddv.w $w20, $w26, $w30\n"
        "fill.w  $w30, $a1\n"
        "lbu     $a1, 5(%[rhs_ptr])\n"
        "maddv.w $w13, $w24, $w30\n"
        "maddv.w $w17, $w25, $w30\n"
        "maddv.w $w21, $w26, $w30\n"
        "fill.w  $w30, $a2\n"
        "lbu     $a2, 6(%[rhs_ptr])\n"
        "maddv.w $w14, $w24, $w30\n"
        "maddv.w $w18, $w25, $w30\n"
        "maddv.w $w22, $w26, $w30\n"
        "fill.w  $w30, $a3\n"
        "lbu     $a3, 7(%[rhs_ptr])\n"
        "maddv.w $w15, $w24, $w30\n"
        "maddv.w $w19, $w25, $w30\n"
        "maddv.w $w23, $w26, $w30\n"

        // Depth 1.
        "fill.w  $w30, $a0\n"
        "lbu     $a0, 12(%[rhs_ptr])\n"
        "maddv.w $w0, $w27, $w30\n"
        "maddv.w $w4, $w28, $w30\n"
        "maddv.w $w8, $w29, $w30\n"
        "fill.w  $w30, $a1\n"
        "lbu     $a1, 13(%[rhs_ptr])\n"
        "maddv.w $w1, $w27, $w30\n"
        "maddv.w $w5, $w28, $w30\n"
        "maddv.w $w9, $w29, $w30\n"
        "fill.w  $w30, $a2\n"
        "lbu     $a2, 14(%[rhs_ptr])\n"
        "maddv.w $w2, $w27, $w30\n"
        "maddv.w $w6, $w28, $w30\n"
        "maddv.w $w10, $w29, $w30\n"
        "fill.w  $w30, $a3\n"
        "lbu     $a3, 15(%[rhs_ptr])\n"
        "maddv.w $w3, $w27, $w30\n"
        "maddv.w $w7, $w28, $w30\n"
        "maddv.w $w11, $w29, $w30\n"

        "fill.w  $w30, $a0\n"
        "maddv.w $w12, $w27, $w30\n"
        "maddv.w $w16, $w28, $w30\n"
        "maddv.w $w20, $w29, $w30\n"
        "fill.w  $w30, $a1\n"
        "maddv.w $w13, $w27, $w30\n"
        "maddv.w $w17, $w28, $w30\n"
        "maddv.w $w21, $w29, $w30\n"
        "fill.w  $w30, $a2\n"
        "maddv.w $w14, $w27, $w30\n"
        "maddv.w $w18, $w28, $w30\n"
        "maddv.w $w22, $w29, $w30\n"
        "fill.w  $w30, $a3\n"
        "maddv.w $w15, $w27, $w30\n"
        "maddv.w $w19, $w28, $w30\n"
        "maddv.w $w23, $w29, $w30\n"

        "addiu  %[depth], -2\n"
        GEMMLOWP_MIPS_XADDIU " %[lhs_ptr], 24\n"
        GEMMLOWP_MIPS_XADDIU " %[rhs_ptr], 16\n"
        "bnez   %[depth]," GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators.
        "st.w   $w0,   (0*16)(%[accum_ptr])\n"
        "st.w   $w4,   (1*16)(%[accum_ptr])\n"
        "st.w   $w8,   (2*16)(%[accum_ptr])\n"
        "st.w   $w1,   (3*16)(%[accum_ptr])\n"
        "st.w   $w5,   (4*16)(%[accum_ptr])\n"
        "st.w   $w9,   (5*16)(%[accum_ptr])\n"
        "st.w   $w2,   (6*16)(%[accum_ptr])\n"
        "st.w   $w6,   (7*16)(%[accum_ptr])\n"
        "st.w   $w10,  (8*16)(%[accum_ptr])\n"
        "st.w   $w3,   (9*16)(%[accum_ptr])\n"
        "st.w   $w7,  (10*16)(%[accum_ptr])\n"
        "st.w   $w11, (11*16)(%[accum_ptr])\n"
        "st.w   $w12, (12*16)(%[accum_ptr])\n"
        "st.w   $w16, (13*16)(%[accum_ptr])\n"
        "st.w   $w20, (14*16)(%[accum_ptr])\n"
        "st.w   $w13, (15*16)(%[accum_ptr])\n"
        "st.w   $w17, (16*16)(%[accum_ptr])\n"
        "st.w   $w21, (17*16)(%[accum_ptr])\n"
        "st.w   $w14, (18*16)(%[accum_ptr])\n"
        "st.w   $w18, (19*16)(%[accum_ptr])\n"
        "st.w   $w22, (20*16)(%[accum_ptr])\n"
        "st.w   $w15, (21*16)(%[accum_ptr])\n"
        "st.w   $w19, (22*16)(%[accum_ptr])\n"
        "st.w   $w23, (23*16)(%[accum_ptr])\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "memory",
        "a0", "a1", "a2", "a3",
        "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7",
        "$f8", "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
        "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
        "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31");
  }
};

// Assembly implementation of the above
// MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_intrinsics2 (TODO).
// Using 16x16=32 multiplications.
// 32 MSA regs used:
// - 24 accumulators
// - 3 lhs
// - 4 rhs
// - 1 temps/zeroes
// ~70 instructions in the loop.
struct MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly2 {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::DepthMajor>, 2> >
      Format;
  static void Run(OperandType* lhs_ptr, OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "ld.w   $w0,   (0*16)(%[accum_ptr])\n"
        "ld.w   $w4,   (1*16)(%[accum_ptr])\n"
        "ld.w   $w8,   (2*16)(%[accum_ptr])\n"
        "ld.w   $w1,   (3*16)(%[accum_ptr])\n"
        "ld.w   $w5,   (4*16)(%[accum_ptr])\n"
        "ld.w   $w9,   (5*16)(%[accum_ptr])\n"
        "ld.w   $w2,   (6*16)(%[accum_ptr])\n"
        "ld.w   $w6,   (7*16)(%[accum_ptr])\n"
        "ld.w   $w10,  (8*16)(%[accum_ptr])\n"
        "ld.w   $w3,   (9*16)(%[accum_ptr])\n"
        "ld.w   $w7,  (10*16)(%[accum_ptr])\n"
        "ld.w   $w11, (11*16)(%[accum_ptr])\n"
        "ld.w   $w12, (12*16)(%[accum_ptr])\n"
        "ld.w   $w16, (13*16)(%[accum_ptr])\n"
        "ld.w   $w20, (14*16)(%[accum_ptr])\n"
        "ld.w   $w13, (15*16)(%[accum_ptr])\n"
        "ld.w   $w17, (16*16)(%[accum_ptr])\n"
        "ld.w   $w21, (17*16)(%[accum_ptr])\n"
        "ld.w   $w14, (18*16)(%[accum_ptr])\n"
        "ld.w   $w18, (19*16)(%[accum_ptr])\n"
        "ld.w   $w22, (20*16)(%[accum_ptr])\n"
        "ld.w   $w15, (21*16)(%[accum_ptr])\n"
        "ld.w   $w19, (22*16)(%[accum_ptr])\n"
        "ld.w   $w23, (23*16)(%[accum_ptr])\n"
        // Set a temp to all zeroes.
        "ldi.b  $w31, 0\n"

        GEMMLOWP_LABEL_LOOP ":\n"
        // Overview of register layout:
        //
        // A half of the 2 2x4 cells of Rhs is stored in 16bit in w27-w30
        // (each register contains 4 replicas of a pair of elements).
        // A 12x2 block of 3 4x2 cells Lhs is stored in 16bit in w24-w26.
        // A 12x8 block of accumulators is stored in 32bit in w0-w23.
        //
        //                    +------+------+------+------+
        //               Rhs  |w27   |w28   |w29   |w30   |
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

        // Load 4 bytes of rhs[] for the first half of depth 0.
        "lbu    $a0, 0(%[rhs_ptr])\n"
        "lbu    $a1, 1(%[rhs_ptr])\n"
        "lbu    $a2, 2(%[rhs_ptr])\n"
        "lbu    $a3, 3(%[rhs_ptr])\n"
        // Load 4 bytes of rhs[] for the first half of depth 1.
        "lbu    $v0, 4(%[rhs_ptr])\n"
        "lbu    $v1, 5(%[rhs_ptr])\n"
        "lbu    $t8, 6(%[rhs_ptr])\n"
        "lbu    $t9, 7(%[rhs_ptr])\n"

        // Zero-extend 8-bit elements of lhs[] to 16 bits.
        "ilvr.b $w24, $w31, $w24\n"
        "ilvl.b $w26, $w31, $w25\n"
        "ilvr.b $w25, $w31, $w25\n"
        // Interleave depth 0 and depth 1 elements of lhs[] for dpadd_u.w.
        "ilvl.d $w27, $w31, $w24\n"
        "ilvl.d $w28, $w31, $w25\n"
        "ilvl.d $w29, $w31, $w26\n"
        "ilvr.h $w24, $w27, $w24\n"
        "ilvr.h $w25, $w28, $w25\n"
        "ilvr.h $w26, $w29, $w26\n"

        // Combine and interleave depth 0 and depth 1 elements of rhs[] for dpadd_u.w
        // (for the first half).
        "ins    $a0, $v0, 16, 8\n"
        "ins    $a1, $v1, 16, 8\n"
        "ins    $a2, $t8, 16, 8\n"
        "ins    $a3, $t9, 16, 8\n"
        // Make 4 replicas of every pair of rhs[] elements.
        "fill.w $w27, $a0\n"
        "fill.w $w28, $a1\n"
        "fill.w $w29, $a2\n"
        "fill.w $w30, $a3\n"

        // Load 4 bytes of rhs[] for the second half of depth 0.
        "lbu    $a0, 8(%[rhs_ptr])\n"
        "lbu    $a1, 9(%[rhs_ptr])\n"
        "lbu    $a2, 10(%[rhs_ptr])\n"
        "lbu    $a3, 11(%[rhs_ptr])\n"
        // Load 4 bytes of rhs[] for the second half of depth 1.
        "lbu    $v0, 12(%[rhs_ptr])\n"
        "lbu    $v1, 13(%[rhs_ptr])\n"
        "lbu    $t8, 14(%[rhs_ptr])\n"
        "lbu    $t9, 15(%[rhs_ptr])\n"

        // First half of depths 0 and 1.
        // Dot-product-(and)-add doubles multiplicand width.
        "dpadd_u.w $w0, $w24, $w27\n"
        "dpadd_u.w $w4, $w25, $w27\n"
        "dpadd_u.w $w8, $w26, $w27\n"
        "dpadd_u.w $w1, $w24, $w28\n"
        "dpadd_u.w $w5, $w25, $w28\n"
        "dpadd_u.w $w9, $w26, $w28\n"
        "dpadd_u.w $w2, $w24, $w29\n"
        "dpadd_u.w $w6, $w25, $w29\n"
        "dpadd_u.w $w10, $w26, $w29\n"
        "dpadd_u.w $w3, $w24, $w30\n"
        "dpadd_u.w $w7, $w25, $w30\n"
        "dpadd_u.w $w11, $w26, $w30\n"

        // Combine and interleave depth 0 and depth 1 elements of rhs[] for dpadd_u.w
        // (for the second half).
        "ins    $a0, $v0, 16, 8\n"
        "ins    $a1, $v1, 16, 8\n"
        "ins    $a2, $t8, 16, 8\n"
        "ins    $a3, $t9, 16, 8\n"
        // Make 4 replicas of every pair of rhs[] elements.
        "fill.w $w27, $a0\n"
        "fill.w $w28, $a1\n"
        "fill.w $w29, $a2\n"
        "fill.w $w30, $a3\n"

        // Second half of depths 0 and 1.
        // Dot-product-(and)-add doubles multiplicand width.
        "dpadd_u.w $w12, $w24, $w27\n"
        "dpadd_u.w $w16, $w25, $w27\n"
        "dpadd_u.w $w20, $w26, $w27\n"
        "dpadd_u.w $w13, $w24, $w28\n"
        "dpadd_u.w $w17, $w25, $w28\n"
        "dpadd_u.w $w21, $w26, $w28\n"
        "dpadd_u.w $w14, $w24, $w29\n"
        "dpadd_u.w $w18, $w25, $w29\n"
        "dpadd_u.w $w22, $w26, $w29\n"
        "dpadd_u.w $w15, $w24, $w30\n"
        "dpadd_u.w $w19, $w25, $w30\n"
        "dpadd_u.w $w23, $w26, $w30\n"

        "addiu  %[depth], -2\n"
        GEMMLOWP_MIPS_XADDIU " %[lhs_ptr], 24\n"
        GEMMLOWP_MIPS_XADDIU " %[rhs_ptr], 16\n"
        "bnez   %[depth]," GEMMLOWP_LABEL_LOOP "b\n"

        // Store accumulators.
        "st.w   $w0,   (0*16)(%[accum_ptr])\n"
        "st.w   $w4,   (1*16)(%[accum_ptr])\n"
        "st.w   $w8,   (2*16)(%[accum_ptr])\n"
        "st.w   $w1,   (3*16)(%[accum_ptr])\n"
        "st.w   $w5,   (4*16)(%[accum_ptr])\n"
        "st.w   $w9,   (5*16)(%[accum_ptr])\n"
        "st.w   $w2,   (6*16)(%[accum_ptr])\n"
        "st.w   $w6,   (7*16)(%[accum_ptr])\n"
        "st.w   $w10,  (8*16)(%[accum_ptr])\n"
        "st.w   $w3,   (9*16)(%[accum_ptr])\n"
        "st.w   $w7,  (10*16)(%[accum_ptr])\n"
        "st.w   $w11, (11*16)(%[accum_ptr])\n"
        "st.w   $w12, (12*16)(%[accum_ptr])\n"
        "st.w   $w16, (13*16)(%[accum_ptr])\n"
        "st.w   $w20, (14*16)(%[accum_ptr])\n"
        "st.w   $w13, (15*16)(%[accum_ptr])\n"
        "st.w   $w17, (16*16)(%[accum_ptr])\n"
        "st.w   $w21, (17*16)(%[accum_ptr])\n"
        "st.w   $w14, (18*16)(%[accum_ptr])\n"
        "st.w   $w18, (19*16)(%[accum_ptr])\n"
        "st.w   $w22, (20*16)(%[accum_ptr])\n"
        "st.w   $w15, (21*16)(%[accum_ptr])\n"
        "st.w   $w19, (22*16)(%[accum_ptr])\n"
        "st.w   $w23, (23*16)(%[accum_ptr])\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "memory",
        "v0", "v1",
        "a0", "a1", "a2", "a3",
        "t8", "t9",
        "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7",
        "$f8", "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
        "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
        "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31");
  }
};

// WidthMajor variant of the above.
// Using 16x16=32 multiplications.
// 32 MSA regs used:
// - 24 accumulators
// - 3 lhs
// - 5 rhs
// ~47 instructions in the loop.
struct MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly3 {
  typedef std::uint8_t OperandType;
  typedef std::uint32_t AccumulatorType;
  typedef KernelFormat<
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 3>,
      KernelSideFormat<CellFormat<4, 2, CellOrder::WidthMajor>, 2> >
      Format;
  static void Run(OperandType* lhs_ptr, OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    asm volatile(
        // Load accumulators
        "ld.w   $w0,   (0*16)(%[accum_ptr])\n"
        "ld.w   $w4,   (1*16)(%[accum_ptr])\n"
        "ld.w   $w8,   (2*16)(%[accum_ptr])\n"
        "ld.w   $w1,   (3*16)(%[accum_ptr])\n"
        "ld.w   $w5,   (4*16)(%[accum_ptr])\n"
        "ld.w   $w9,   (5*16)(%[accum_ptr])\n"
        "ld.w   $w2,   (6*16)(%[accum_ptr])\n"
        "ld.w   $w6,   (7*16)(%[accum_ptr])\n"
        "ld.w   $w10,  (8*16)(%[accum_ptr])\n"
        "ld.w   $w3,   (9*16)(%[accum_ptr])\n"
        "ld.w   $w7,  (10*16)(%[accum_ptr])\n"
        "ld.w   $w11, (11*16)(%[accum_ptr])\n"
        "ld.w   $w12, (12*16)(%[accum_ptr])\n"
        "ld.w   $w16, (13*16)(%[accum_ptr])\n"
        "ld.w   $w20, (14*16)(%[accum_ptr])\n"
        "ld.w   $w13, (15*16)(%[accum_ptr])\n"
        "ld.w   $w17, (16*16)(%[accum_ptr])\n"
        "ld.w   $w21, (17*16)(%[accum_ptr])\n"
        "ld.w   $w14, (18*16)(%[accum_ptr])\n"
        "ld.w   $w18, (19*16)(%[accum_ptr])\n"
        "ld.w   $w22, (20*16)(%[accum_ptr])\n"
        "ld.w   $w15, (21*16)(%[accum_ptr])\n"
        "ld.w   $w19, (22*16)(%[accum_ptr])\n"
        "ld.w   $w23, (23*16)(%[accum_ptr])\n"

        GEMMLOWP_LABEL_LOOP
        ":\n"
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

        "addiu  %[depth], -2\n" GEMMLOWP_MIPS_XADDIU
        " %[lhs_ptr], 24\n" GEMMLOWP_MIPS_XADDIU
        " %[rhs_ptr], 16\n"
        "bnez   %[depth]," GEMMLOWP_LABEL_LOOP
        "b\n"

        // Store accumulators.
        "st.w   $w0,   (0*16)(%[accum_ptr])\n"
        "st.w   $w4,   (1*16)(%[accum_ptr])\n"
        "st.w   $w8,   (2*16)(%[accum_ptr])\n"
        "st.w   $w1,   (3*16)(%[accum_ptr])\n"
        "st.w   $w5,   (4*16)(%[accum_ptr])\n"
        "st.w   $w9,   (5*16)(%[accum_ptr])\n"
        "st.w   $w2,   (6*16)(%[accum_ptr])\n"
        "st.w   $w6,   (7*16)(%[accum_ptr])\n"
        "st.w   $w10,  (8*16)(%[accum_ptr])\n"
        "st.w   $w3,   (9*16)(%[accum_ptr])\n"
        "st.w   $w7,  (10*16)(%[accum_ptr])\n"
        "st.w   $w11, (11*16)(%[accum_ptr])\n"
        "st.w   $w12, (12*16)(%[accum_ptr])\n"
        "st.w   $w16, (13*16)(%[accum_ptr])\n"
        "st.w   $w20, (14*16)(%[accum_ptr])\n"
        "st.w   $w13, (15*16)(%[accum_ptr])\n"
        "st.w   $w17, (16*16)(%[accum_ptr])\n"
        "st.w   $w21, (17*16)(%[accum_ptr])\n"
        "st.w   $w14, (18*16)(%[accum_ptr])\n"
        "st.w   $w18, (19*16)(%[accum_ptr])\n"
        "st.w   $w22, (20*16)(%[accum_ptr])\n"
        "st.w   $w15, (21*16)(%[accum_ptr])\n"
        "st.w   $w19, (22*16)(%[accum_ptr])\n"
        "st.w   $w23, (23*16)(%[accum_ptr])\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [depth] "+r"(depth)
        :  // inputs
        [accum_ptr] "r"(accum_ptr)
        :  // clobbers
        "memory", "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7", "$f8",
        "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15", "$f16", "$f17",
        "$f18", "$f19", "$f20", "$f21", "$f22", "$f23", "$f24", "$f25", "$f26",
        "$f27", "$f28", "$f29", "$f30", "$f31");
  }
};
#endif  // __mips

// BEGIN code copied from gemmlowp/internal/kernel_reference.h

// This kernel is templatized in an arbitrary Format template parameter,
// allowing it to have any arbitrary format.
template <typename tOperandType, typename tAccumulatorType, typename tFormat>
struct ReferenceKernel {
  typedef tOperandType OperandType;
  typedef tAccumulatorType AccumulatorType;
  typedef tFormat Format;

  static void Run(const OperandType* lhs_ptr, const OperandType* rhs_ptr,
                  AccumulatorType* accum_ptr, int depth) {
    const int depth_cells = static_cast<int>(depth / Format::kDepth);

    // The outer loop is over the depth dimension.
    for (int dc = 0; dc < depth_cells; dc++) {
      // The next two loops are over cells of the Lhs (stacked vertically),
      // and over cells of the Rhs (stacked horizontally).
      for (int rc = 0; rc < Format::Lhs::kCells; rc++) {
        const OperandType* lhs_cell_ptr =
            lhs_ptr + (dc * Format::Lhs::kCells + rc) *
                          Format::Lhs::Cell::kWidth * Format::kDepth;
        for (int cc = 0; cc < Format::Rhs::kCells; cc++) {
          const OperandType* rhs_cell_ptr =
              rhs_ptr + (dc * Format::Rhs::kCells + cc) *
                            Format::Rhs::Cell::kWidth * Format::kDepth;

          // Now we are inside one cell of the Lhs and inside one cell
          // of the Rhs, so the remaining inner loops are just
          // traditional three loops of matrix multiplication.
          for (int di = 0; di < Format::kDepth; di++) {
            for (int ri = 0; ri < Format::Lhs::Cell::kWidth; ri++) {
              for (int ci = 0; ci < Format::Rhs::Cell::kWidth; ci++) {
                const OperandType* lhs_coeff_ptr =
                    lhs_cell_ptr +
                    OffsetIntoCell<typename Format::Lhs::Cell>(ri, di);
                const OperandType* rhs_coeff_ptr =
                    rhs_cell_ptr +
                    OffsetIntoCell<typename Format::Rhs::Cell>(ci, di);
                AccumulatorType* accumulator_coeff_ptr =
                    accum_ptr + (ri + rc * Format::Lhs::Cell::kWidth) +
                    (ci + cc * Format::Rhs::Cell::kWidth) * Format::kRows;
                *accumulator_coeff_ptr += AccumulatorType(*lhs_coeff_ptr) *
                                          AccumulatorType(*rhs_coeff_ptr);
              }
            }
          }
        }
      }
    }
  }
};

// END code copied from gemmlowp/internal/kernel_reference.h

template <typename DataType>
class CacheLineAlignedBuffer {
 public:
  CacheLineAlignedBuffer(std::size_t size) : size_(size) {
    data_ = nullptr;
    // Adds a few bytes of padding here, because the 64-bit 'A57' kernel
    // reads one iteration past the end the buffer, causing a crash on iOS.
    int res = posix_memalign(reinterpret_cast<void**>(&data_), kCacheLineSize,
                             size_ * sizeof(DataType) + 16);
    (void)res;
  }

  ~CacheLineAlignedBuffer() { free(data_); }

  const DataType* data() const { return data_; }
  DataType* data() { return data_; }

  std::size_t size() const { return size_; }

 private:
  const std::size_t size_;
  DataType* data_;
};

template <typename DataType>
void FillRandom(CacheLineAlignedBuffer<DataType>* buffer, DataType min,
                DataType max) {
  static std::mt19937 generator(0);
  std::uniform_real_distribution<float> dist(min, max);
  for (std::size_t i = 0; i < buffer->size(); i++) {
    buffer->data()[i] = DataType(dist(generator));
  }
}

template <typename DataType>
void FillZero(CacheLineAlignedBuffer<DataType>* buffer) {
  for (std::size_t i = 0; i < buffer->size(); i++) {
    buffer->data()[i] = DataType(0);
  }
}

template <typename DataType>
void Copy(CacheLineAlignedBuffer<DataType>* dst,
          const CacheLineAlignedBuffer<DataType>& src) {
  assert(dst->size() == src.size());
  memcpy(dst->data(), src.data(), src.size() * sizeof(DataType));
}

template <typename DataType>
void PrintMatrix(int rows, int cols, int rowstride, int colstride,
                 const DataType* data) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      std::cerr << double(data[r * rowstride + c * colstride]) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

template <typename DataType>
bool approx_equals(DataType a, DataType b) {
  return a == b;
}

template <>
bool approx_equals(float a, float b) {
  if (!a && !b) {
    return true;
  }
  // 1e-1 is very coarse accuracy, we should switch to an overall L2 metric
  // and tighten the tolerance on that metric.
  return std::abs(a - b) < 1e-1f * std::min(std::abs(a), std::abs(b));
}

template <typename Kernel>
void test_kernel(int depth, const char* kernel_name) {
  typedef typename Kernel::OperandType OperandType;
  typedef typename Kernel::AccumulatorType AccumulatorType;
  typedef typename Kernel::Format Format;
  static const int kLhsWidth = Format::Lhs::kWidth;
  static const int kRhsWidth = Format::Rhs::kWidth;

  typedef ReferenceKernel<OperandType, AccumulatorType, Format> ReferenceKernel;

  CacheLineAlignedBuffer<OperandType> lhs(kLhsWidth * depth);
  CacheLineAlignedBuffer<OperandType> rhs(kRhsWidth * depth);
  CacheLineAlignedBuffer<AccumulatorType> accum_initial(kLhsWidth * kRhsWidth);
  CacheLineAlignedBuffer<AccumulatorType> accum(kLhsWidth * kRhsWidth);
  CacheLineAlignedBuffer<AccumulatorType> accum_reference(kLhsWidth *
                                                          kRhsWidth);

  FillRandom(&lhs, KernelOperandRanges<Kernel>::LhsMin(),
             KernelOperandRanges<Kernel>::LhsMax());
  FillRandom(&rhs, KernelOperandRanges<Kernel>::RhsMin(),
             KernelOperandRanges<Kernel>::RhsMax());
  FillRandom(&accum_initial,
             std::is_signed<AccumulatorType>::value
                 ? AccumulatorType(-100)
                 : AccumulatorType(0),
              AccumulatorType(100));

  Copy(&accum, accum_initial);
  Copy(&accum_reference, accum_initial);

  ReferenceKernel::Run(lhs.data(), rhs.data(), accum_reference.data(), depth);
  Kernel::Run(lhs.data(), rhs.data(), accum.data(), depth);

  for (int l = 0; l < kLhsWidth; l++) {
    for (int r = 0; r < kRhsWidth; r++) {
      const int index = l + kLhsWidth * r;
      if (!approx_equals(accum.data()[index], accum_reference.data()[index])) {
        std::cerr << "Arithmetic error in kernel:" << std::endl
                  << "    " << kernel_name << std::endl
                  << "Wrong accumulator for depth=" << depth << ", "
                  << "at l = " << l << ", r = " << r << std::endl;
        std::cerr << "reference value: " << accum_reference.data()[index]
                  << std::endl;
        std::cerr << "actual value:    " << accum.data()[index] << std::endl;
        if (depth <= 16) {
          std::cerr << "LHS matrix:" << std::endl;
          PrintMatrix(kLhsWidth, depth, 1, kLhsWidth, lhs.data());
          std::cerr << "RHS matrix:" << std::endl;
          PrintMatrix(depth, kRhsWidth, kRhsWidth, 1, rhs.data());
          std::cerr << "Initial Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth, accum_initial.data());
          std::cerr << "Reference Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth,
                      accum_reference.data());
          std::cerr << "Actual Accumulator matrix:" << std::endl;
          PrintMatrix(kLhsWidth, kRhsWidth, 1, kLhsWidth, accum.data());
        }
        abort();
      }
    }
  }
}

template <typename Kernel>
int ops(int depth) {
  // 2x the number of multiply-accumulate scalar ops.
  return 2 * Kernel::Format::Lhs::kWidth * Kernel::Format::Rhs::kWidth * depth;
}

template <unsigned Modulus, typename Integer>
Integer RoundDown(Integer i) {
  return i - (i % Modulus);
}

int CacheSizeInKB() {
  static const char* cache_size_k_env = getenv("CACHE_SIZE_KB");
  static const int cache_size_k =
      cache_size_k_env ? atoi(cache_size_k_env) : kDefaultCacheSizeK;
  return cache_size_k;
}

template <typename Kernel>
int BenchmarkDepthToFitInCache() {
  const int cache_size_bytes = 1024 * CacheSizeInKB();

  // Subtract the typical size of a few cache lines, so
  // we don't need to worry too hard about e.g. some stack data.
  const int conservative_cache_size_bytes =
      cache_size_bytes - 2 * kCacheLineSize;

  // We will subtract the memory occupied by accumulators.
  typedef typename Kernel::AccumulatorType AccumulatorType;
  const int kAccumulatorBytes = sizeof(AccumulatorType) *
                                Kernel::Format::Lhs::kWidth *
                                Kernel::Format::Rhs::kWidth;

  // Compute the depth.
  typedef typename Kernel::OperandType OperandType;
  const int kBytesPerUnitOfDepth =
      sizeof(OperandType) *
      (Kernel::Format::Lhs::kWidth + Kernel::Format::Rhs::kWidth);
  const int unrounded_depth =
      (conservative_cache_size_bytes - kAccumulatorBytes) /
      kBytesPerUnitOfDepth;

  // Cap depth, to avoid unfairly favoring narrower kernels
  const int kMaxDepth = 1024;
  const int clamped_unrounded_depth = std::min(kMaxDepth, unrounded_depth);

  // Round depth down to a multiple of cache line size, which helps because
  // our kernels may crash if depth is not a multiple of the number of
  // depth level that they want to
  // handle at each loop iteration, and we don't want to require kernels
  // to be more complex. Currently all kernels process 1, 2 or 8 levels of
  // depth at a time. The main reason why that might increase in the future
  // is if registers get wider, but I don't suppose that register could
  // ever get wider than cache lines.
  return RoundDown<kCacheLineSize>(clamped_unrounded_depth);
}

double current_time_in_seconds() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

template <typename Kernel>
double benchmark(int depth) {
  // Minimum duration for this benchmark to run. If the workload finishes
  // sooner, we retry with double the number of iterations.
  static const double min_benchmark_time_in_seconds = 1.0;

  typedef typename Kernel::OperandType OperandType;
  typedef typename Kernel::AccumulatorType AccumulatorType;

  CacheLineAlignedBuffer<OperandType> lhs(Kernel::Format::Lhs::kWidth * depth);
  CacheLineAlignedBuffer<OperandType> rhs(Kernel::Format::Rhs::kWidth * depth);
  CacheLineAlignedBuffer<AccumulatorType> accum(Kernel::Format::Lhs::kWidth *
                                                Kernel::Format::Rhs::kWidth);

  for (std::uint64_t iters_at_a_time = 1;; iters_at_a_time *= 2) {
    const double t_start = current_time_in_seconds();
    for (std::uint64_t i = 0; i < iters_at_a_time; i++) {
      Kernel::Run(lhs.data(), rhs.data(), accum.data(), depth);
    }
    const double t_end = current_time_in_seconds();
    const double elapsed = t_end - t_start;
    if (elapsed > min_benchmark_time_in_seconds) {
      return iters_at_a_time * ops<Kernel>(depth) / elapsed;
    }
  }
}

template <typename Kernel>
void benchmark_and_print_results(const char* kernel_name) {
  if (getenv("BENCHMARK_KERNEL")) {
    if (strcmp(getenv("BENCHMARK_KERNEL"), kernel_name)) {
      return;
    }
  }
  const int kKernelDepth = Kernel::Format::kDepth;
  for (int depth = kKernelDepth; depth <= 1024; depth += kKernelDepth) {
    test_kernel<Kernel>(depth, kernel_name);
  }

  if (getenv("BENCHMARK_ALL_DEPTHS")) {
    for (int depth = kKernelDepth;
         depth <= BenchmarkDepthToFitInCache<Kernel>(); depth *= 2) {
      std::cout << kernel_name << "," << depth << ","
                << benchmark<Kernel>(depth) * 1e-9f << std::endl;
    }
  } else {
    const int depth = BenchmarkDepthToFitInCache<Kernel>();
    std::cout << kernel_name << "," << benchmark<Kernel>(depth) * 1e-9f
              << std::endl;
  }
}

#define BENCHMARK(Kernel)                         \
  do {                                            \
    benchmark_and_print_results<Kernel>(#Kernel); \
  } while (false)

int main() {
  if (getenv("BENCHMARK_ALL_DEPTHS")) {
    std::cout << "kernel,depth,Gop/s" << std::endl;
  } else {
    std::cout << "kernel,Gop/s" << std::endl;
  }

#ifdef __arm__
  BENCHMARK(NEON_32bit_GEMM_Int8Operands_AccumTwoWithin16Bits);
  BENCHMARK(NEON_32bit_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics);
  BENCHMARK(NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators);
  BENCHMARK(NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_intrinsics);
  BENCHMARK(NEON_32bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand);
  BENCHMARK(NEON_32bit_GEMM_Int32_WithScalar);
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_WithVectorDuplicatingScalar);
#ifdef __ARM_FEATURE_FMA
  BENCHMARK(NEON_32bit_GEMM_Float32_FMA_WithVectorDuplicatingScalar);
#endif
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_WithScalar);
  BENCHMARK(NEON_32bit_GEMM_Float32_WithScalar_intrinsics);
  BENCHMARK(NEON_32bit_GEMM_Float32_WithScalar_A53);
  BENCHMARK(NEON_32bit_GEMM_Float32_WithScalar_A53_depth2);
  BENCHMARK(NEON_32bit_GEMM_Float32_MLA_Rotating);
#ifdef __ARM_FEATURE_FMA
  BENCHMARK(NEON_32bit_GEMM_Float32_FMA_Rotating);
#endif
#endif

#ifdef __aarch64__
  BENCHMARK(NEON_64bit_GEMM_Int425Operands);
  BENCHMARK(NEON_64bit_GEMM_Int425Operands_intrinsics);
  BENCHMARK(NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits);
  BENCHMARK(NEON_64bit_GEMM_Int7Operands_AccumEightWithin16Bits_intrinsics);
  BENCHMARK(NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits);
  BENCHMARK(NEON_64bit_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics);
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators);
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_intrinsics);
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_noexpand_A57);
#ifdef __ARM_FEATURE_DOTPROD
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct);
  BENCHMARK(NEON_64bit_GEMM_Uint8Operands_Uint32Accumulators_dotproduct_A55r1);
#endif
  BENCHMARK(NEON_64bit_GEMM_Int32_WithScalar);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithVectorDuplicatingScalar);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar_intrinsics);
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar_A57);
#ifndef __APPLE__
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar_A53);
#endif
  BENCHMARK(NEON_64bit_GEMM_Float32_WithScalar_A55r1);
#endif

#ifdef __mips
  BENCHMARK(MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_intrinsics);
  BENCHMARK(MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_assembly);
  BENCHMARK(MSA_GEMM_12x4_Uint8Operands_Uint32Accumulators_assembly2);
  BENCHMARK(MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_intrinsics);
  BENCHMARK(MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly);
  BENCHMARK(MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly2);
  BENCHMARK(MSA_GEMM_12x8_Uint8Operands_Uint32Accumulators_assembly3);
#endif

  return 0;
}
