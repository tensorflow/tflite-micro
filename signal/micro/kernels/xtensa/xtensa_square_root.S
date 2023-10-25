/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

.section .note.GNU-stack,"",@progbits

#include "xtensa/config/core-isa.h"

#ifdef __XTENSA_CALL0_ABI__
#define NO_REGISTER_WINDOW (1)
#endif

#if XCHAL_HAVE_WINDOWED == 0
#define NO_REGISTER_WINDOW
#endif

// Since the 64 bit sqrt jumps into the middle of the 32 bit sqrt under certain
// conditions, both functions should reserve the same amount of stack space.
#define XTENSA_SQRT_STACK_SIZE 32

.text
.type xtensa_sqrt_64, @function
.align 4
.global xtensa_sqrt_64

// Make macros for our 64 bit functions, since we don't have a carry/borrow bit
// in the base ISA, these take up way more cycles than they should. These are
// the "preferred instruction idioms" from 8.9.2 of the base ISA manual. Since
// these macros define a jump (and I couldn't find a way to be clever and use
// something like __LINE__/__FILE__ to define these automatically, you may also
// have to provide an 'opname' that contains a unique string to define a label
// for the macro.

// dest must not be the same as num2, or this function will not work!
#define ADD_64(dest, num1, num2, opname) \
  add.n dest##_low, num1##_low, num2##_low; \
  add.n dest##_high, num1##_high, num2##_high; \
  bgeu dest##_low, num2##_low, .add_64_jump_##opname; \
  addi.n dest##_high, dest##_high, 1; \
  .add_64_jump_##opname:

// All three registers must be unique, or this function will not work!
#define SUB_64(dest, num1, num2, opname) \
  sub dest##_low, num1##_low, num2##_low; \
  sub dest##_high, num1##_high, num2##_high; \
  bgeu num1##_low, num2##_low, .sub_64_jump_##opname; \
  addi.n dest##_high, dest##_high, -1; \
  .sub_64_jump_##opname:

#define SRLI_64(dest, val, imm) \
  slli scratch4, val##_high, (32 - imm); \
  srli dest##_high, val##_high, imm; \
  srli dest##_low, val##_low, imm; \
  or dest##_low, dest##_low, scratch4;

#define COND_MOV_64(op, dest, val, test) \
  mov##op dest##_low, val##_low, test; \
  mov##op dest##_high, val##_high, test

#define num_low a2
#define num_high a3
#define bit_low a4
#define bit_high a5
#define res_low a6
#define res_high a7
#define temp1_low a8
#define temp1_high a9
#define temp2_low a10
#define temp2_high a11
#define scratch1 a12
#define scratch2 a13
#define scratch3 a14
#define scratch4 a15
#define temp3_low scratch1
#define temp3_high scratch2

.align 4
xtensa_sqrt_64:
#ifdef NO_REGISTER_WINDOW
addi.n a1, a1, -XTENSA_SQRT_STACK_SIZE
s32i.n a0, a1, 4
s32i.n a11, a1, 8
s32i.n a12, a1, 12
s32i.n a13, a1, 16
s32i.n a14, a1, 20
s32i.n a15, a1, 24
#else
entry a1, XTENSA_SQRT_STACK_SIZE
#endif
// In the event that the upper word of the number is all zero, we can just
// pretend that we're doing a 32 bit sqrt (but the rounding condition at the
// end is slightly different, so we've got a bit of an anomly there. Such is
// life)
beqz.n num_high, .xtensa_sqrt_32_start
// ** uint64 res= 0;
movi.n res_low, 0
movi.n res_high, 0

movi.n scratch2, 1

// Setup 'bit' - first we need to know what bit to set it to.
// ** int max_bit_number = 64 - MostSignificantBit_64(num);
movi.n bit_low, 0
nsau scratch1, num_high

// ** max_bit_number |= 1;
or scratch1, scratch2, scratch1

// The ammount we shift by is 31 - what's in scratch1 for the max bit number.
// This is because we've got the two words, so we can't do a 64 bit shift.
movi.n scratch3, 31
sub scratch1, scratch3, scratch1

// Do the shift
// ** uint32 bit = 1 << (63 - max_bit_number);
ssl scratch1
sll bit_high, scratch2

// Figure out how many iterations we're going to need. However, we already have
// 31 - max_bit_number in scratch1, so just add 32 to that.
// ** int iterations = (63 - max_bit_number) / 2 + 1;
addi.n scratch1, scratch1, 32
srli scratch1, scratch1, 1
add scratch1, scratch1, scratch2

// If the number of iterations is equal to 32, this means that we're likely in
// an overflow spot if we try and do a subtraction (since the upper most bit is
// going to be set since the bit had to be shifted up so high). We have to do
// one iteration of the loop where we use the pipeline destroying branch call
// that can compare two unsigned numbers. If we need less than 32 iterations,
// we can skip this slow path and jump to the tight inner loop.
blti scratch1, 32, .xtensa_sqrt_64_inner_loop_start

// Cache bit + res.
ADD_64(temp1, bit, res, temp1_bit_res)
// Since we've stored a copy of bit + res, we can right shift res (since both
// branches of the conditional are going to need it, one branch just needs to
// perform an extra addition).
// ** res <<= 1;
SRLI_64(res, res, 1);

// ** if (num >= res_plus_bit) {
bltu num_high, temp1_high, .xtensa_sqrt_64_branch_skip
bne num_high, temp1_high, .xtensa_sqrt_64_comparison_failed
bltu num_low, temp1_low, .xtensa_sqrt_64_branch_skip
.xtensa_sqrt_64_comparison_failed:

// **   num -= res + bit;
SUB_64(temp2, num, temp1, temp2_num_temp1_early_branch)
// Since the sub can't use the same registers, we have to move it back to where
// it belongs.
mov.n num_low, temp2_low
mov.n num_high, temp2_high
// **   res += bit;
ADD_64(res, res, bit, res_res_bit_early_branch)
// ** }
.xtensa_sqrt_64_branch_skip:

// ** bit >>= 2;
SRLI_64(bit, bit, 2)
// Make sure we knock off this iteration when we fall into the inner loop.
sub scratch1, scratch1, scratch2

.xtensa_sqrt_64_inner_loop_start:
loop scratch1, .xtensa_sqrt_64_round

// We don't have enough registers to be as verbose as the 32 bit version, so
// this version is not as easy to read. Instead of having the two operations in
// the same style of conditional move, we sort of decide to do both branches at
// the same time of the if, then fix up what was incorrect at the end.
SRLI_64(temp1, res, 1)
ADD_64(res, res, bit, res_res_bit)

SUB_64(temp2, num, res, num_res_temp2)
ADD_64(res, temp1, bit, res_temp1_bit)

COND_MOV_64(gez, num, temp2, temp2_high)
COND_MOV_64(ltz, res, temp1, temp2_high)

// ** bit >>= 2;
SRLI_64(bit, bit, 2)

.xtensa_sqrt_64_round:

// Need to do if (num > res) { ++res; }, but we'll do it with conditional moves
// again. Except we're going to do it slightly backwards, since we need to move
// the result into the num register to be returned. We'll do this by setting
// the return value to res + 1, but in the event that it was a mistake, we'll
// conditionally move the raw result back into place.
SUB_64(temp1, res, num, res_num_temp1)
addi.n num_low, res_low, 1
movgez num_low, res_low, temp1_high

// But we may have overflowed num_low - set it back to result_low if it's been
// zeroed out.
moveqz num_low, res_low, num_low

#ifdef NO_REGISTER_WINDOW
l32i.n a0, a1, 4
l32i.n a11, a1, 8
l32i.n a12, a1, 12
l32i.n a13, a1, 16
l32i.n a14, a1, 20
l32i.n a15, a1, 24
addi a1, a1, XTENSA_SQRT_STACK_SIZE
ret.n
#else
retw.n
#endif
.xtensa_sqrt_64_end:
  .size xtensa_sqrt_64, . - xtensa_sqrt_64


#undef ADD_64
#undef SUB_64
#undef SRLI_64
#undef COND_MOV_64

#undef num_low
#undef num_high
#undef bit_low
#undef bit_high
#undef res_low
#undef res_high
#undef temp1_low
#undef temp1_high
#undef temp2_low
#undef temp2_high
#undef scratch1
#undef scratch2
#undef scratch3
#undef scratch4
#undef temp3_low
#undef temp3_high
.text
.type xtensa_sqrt_32, @function
.align 4
.global xtensa_sqrt_32

// Make the program more readable...
#define num a2
#define bit a4
#define res a5
#define one a6
#define max_bit_number a7
#define iterations max_bit_number
#define bit_plus_res a8
#define num_minus_bit_plus_res a9
#define res_shift_left_plus_bit a10
#define res_minus_num res_shift_left_plus_bit

xtensa_sqrt_32:
#ifdef NO_REGISTER_WINDOW
addi.n a1, a1, -XTENSA_SQRT_STACK_SIZE
s32i.n a0, a1, 4
s32i.n a11, a1, 8
s32i.n a12, a1, 12
s32i.n a13, a1, 16
s32i.n a14, a1, 20
s32i.n a15, a1, 24
#else
entry a1, XTENSA_SQRT_STACK_SIZE
#endif

.xtensa_sqrt_32_start:
// If the number is zero, just quickly exit without doing anything.
beqz.n num, .xtensa_sqrt_32_return

// ** uint32 res = 0;
movi.n res, 0
// Also, setup the handy constant we need a few times.
movi.n one, 1

// This will give us (32 - index of the first bit that is set).
// ** int max_bit_number = 32 - MostSignificantBit_32(num);
nsau max_bit_number, num

// ** max_bit_number |= one;
or max_bit_number, max_bit_number, one

// The ammount we shift by is 31 - what we stored in max_bit_number.
movi.n a15, 31
sub max_bit_number, a15, max_bit_number

// Do the shift.
// ** uint32 bit = 1 << (31 - max_bit_number);
ssl max_bit_number
sll bit, one

// Compute the number of iterations we're going to need.
// ** int iterations = (31 - max_bit_number) / 2 + 1;
srli iterations, max_bit_number, 1
add iterations, iterations, one

// If the number of iterations is equal to 16, this means that we're likely in
// an overflow spot if we try and do a subtraction (since the upper most bit is
// going to be set since the bit had to be shifted up so high). We have to do
// one iteration of the loop where we use the pipeline destroying branch call
// that can compare two unsigned numbers. If we need less than 16 iterations,
// we can skip this slow path and jump to the tight inner loop.
blti iterations, 16, .xtensa_sqrt_32_inner_loop_start

// Cache bit + res into another register.
add.n bit_plus_res, bit, res
// Since we've stored a copy of bit + res, we can right shift res (since both
// branches of the conditional are going to need it, one branch just needs to
// perform an extra addition).
// ** res <<= 1;
srli res, res, 1
// ** if (num >= res_plus_bit) {
bltu num, bit_plus_res, .xtensa_sqrt_32_branch_skip
// **   num -= res + bit;
sub num, num, bit_plus_res
// **   res += bit;
add res, res, bit
// ** }
.xtensa_sqrt_32_branch_skip:

// ** bit >>= 2;
srli bit, bit, 2
// Make sure we knock off this iteration when we fall into the inner loop.
sub iterations, iterations, one

.xtensa_sqrt_32_inner_loop_start:
// Start a zero overhead loop for the number of remaining iterations.
loop iterations, .xtensa_sqrt_32_round

// Cache bit + res into another register.
add.n bit_plus_res, bit, res
// ** res <<= 1;
srli res, res, 1

// We can dodge a hefty branch penalty by doing conditional moves - so we need
// to compute the values for num and res for what would happen if we took the
// if part of the condition. If the condition is true, then we'll copy stuff
// across.

// compute num - bit_plus_res. We can use this for the conditional check
// against zero.
sub num_minus_bit_plus_res, num, bit_plus_res
// compute the shifted res + bit.
add res_shift_left_plus_bit, res, bit

// Copy stuff if the condition is true.
movgez num, num_minus_bit_plus_res, num_minus_bit_plus_res
movgez res, res_shift_left_plus_bit, num_minus_bit_plus_res

// ** bit >>= 2;
srli bit, bit, 2

.xtensa_sqrt_32_round:

// Need to do if (num > res) { ++res; }, but we'll do it with conditional moves
// again. Except we're going to do it slightly backwards, since we need to move
// the result into the num register to be returned. We'll do this by setting
// the return value to res + 1, but in the event that it was a mistake, we'll
// conditionally move the raw result back into place.
sub res_minus_num, res, num
add.n num, res, one
movgez num, res, res_minus_num

// But we might have also pooched the rounding by adding an extra bit, make sure
// we don't explode when we overflow.
clamps num, num, 16

.xtensa_sqrt_32_return:
#ifdef NO_REGISTER_WINDOW
l32i.n a0, a1, 4
l32i.n a11, a1, 8
l32i.n a12, a1, 12
l32i.n a13, a1, 16
l32i.n a14, a1, 20
l32i.n a15, a1, 24
addi a1, a1, XTENSA_SQRT_STACK_SIZE
ret.n
#else
retw.n
#endif

#undef num
#undef bit
#undef res
#undef one
#undef max_bit_number
#undef iterations
#undef bit_plus_res
#undef num_minus_bit_plus_res
#undef res_shift_left_plus_bit
#undef res_minus_num
