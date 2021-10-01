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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/xtensa/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace ops {
namespace micro {
namespace lstm_eval {

#if defined(HIFI5)
void calc_cell_state_without_cifg(int16_t* cell_state,
                                  const int16_t* forget_gate,
                                  const int16_t* cell_gate,
                                  const int16_t* input_gate, int shift1,
                                  int shift2, int clip, int num_elms) {
  const ae_int16x8 *p16x8_cs_r, *p16x8_fg_r;
  const ae_int16x8 *p16x8_cg_r, *p16x8_ig_r;

  ae_int16x8* p16x8_cs_w;

  ae_valignx2 align_cs_r, align_fg_r;
  ae_valignx2 align_cg_r, align_ig_r;
  ae_valignx2 align_cs_w;

  ae_int16x4 d_cs_r_0, d_cs_r_1;
  ae_int16x4 d_fg_0, d_fg_1;
  ae_int16x4 d_cg_0, d_cg_1;
  ae_int16x4 d_ig_0, d_ig_1;
  ae_int16x4 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_mul_0, d_mul_1, d_mul_2, d_mul_3;
  ae_int32x2 d_mul_4, d_mul_5, d_mul_6, d_mul_7;

  ae_int16x4 d_min, d_max;

  int i = 0;
  p16x8_cs_r = (const ae_int16x8*)cell_state;
  p16x8_fg_r = (const ae_int16x8*)forget_gate;
  p16x8_cg_r = (const ae_int16x8*)cell_gate;
  p16x8_ig_r = (const ae_int16x8*)input_gate;

  p16x8_cs_w = (ae_int16x8*)cell_state;

  align_cs_r = AE_LA128_PP(p16x8_cs_r);
  align_fg_r = AE_LA128_PP(p16x8_fg_r);
  align_cg_r = AE_LA128_PP(p16x8_cg_r);
  align_ig_r = AE_LA128_PP(p16x8_ig_r);

  align_cs_w = AE_ZALIGN128();

  if (clip > 0) {
    d_min = AE_MOVDA16(-clip);
    d_max = AE_MOVDA16(clip);
  } else {
    d_min = AE_MOVDA16(-32768);
    d_max = AE_MOVDA16(32767);
  }

#pragma concurrent
  if (shift1 == 15) {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);
      AE_LA16X4X2_IP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r);

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);
      d_cs_w_1 = AE_MULFP16X4RS(d_cs_r_1, d_fg_1);

      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;
    p16_ig_r = (const ae_int16*)p16x8_ig_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];
      d_ig_0 = p16_ig_r[i];

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_ig_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  } else {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);
      AE_LA16X4X2_IP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_mul_1 = AE_SRAA32SYMS(d_mul_1, shift1);
      d_mul_2 = AE_SRAA32SYMS(d_mul_2, shift1);
      d_mul_3 = AE_SRAA32SYMS(d_mul_3, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
      d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);

      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16 *p16_cg_r, *p16_ig_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;
    p16_ig_r = (const ae_int16*)p16x8_ig_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];
      d_ig_0 = p16_ig_r[i];

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_ig_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  }
}

void calc_cell_state_with_cifg(int16_t* cell_state, const int16_t* forget_gate,
                               const int16_t* cell_gate, int shift1, int shift2,
                               int clip, int num_elms) {
  const ae_int16x8 *p16x8_cs_r, *p16x8_fg_r;
  const ae_int16x8* p16x8_cg_r;

  ae_int16x8* p16x8_cs_w;

  ae_valignx2 align_cs_r, align_fg_r;
  ae_valignx2 align_cg_r;
  ae_valignx2 align_cs_w;

  ae_int16x4 d_cs_r_0, d_cs_r_1;
  ae_int16x4 d_fg_0, d_fg_1;
  ae_int16x4 d_cg_0, d_cg_1;
  ae_int16x4 d_1mfg_0, d_1mfg_1;
  ae_int16x4 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_mul_0, d_mul_1, d_mul_2, d_mul_3;
  ae_int32x2 d_mul_4, d_mul_5, d_mul_6, d_mul_7;

  ae_int16x4 d_min, d_max, d_one;

  int i = 0;
  p16x8_cs_r = (const ae_int16x8*)cell_state;
  p16x8_fg_r = (const ae_int16x8*)forget_gate;
  p16x8_cg_r = (const ae_int16x8*)cell_gate;

  p16x8_cs_w = (ae_int16x8*)cell_state;

  align_cs_r = AE_LA128_PP(p16x8_cs_r);
  align_fg_r = AE_LA128_PP(p16x8_fg_r);
  align_cg_r = AE_LA128_PP(p16x8_cg_r);

  align_cs_w = AE_ZALIGN128();

  if (clip > 0) {
    d_min = AE_MOVDA16(-clip);
    d_max = AE_MOVDA16(clip);
  } else {
    d_min = AE_MOVDA16(-32768);
    d_max = AE_MOVDA16(32767);
  }
  d_one = AE_MOVDA16(32767);

#pragma concurrent
  if (shift1 == 15) {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);
      d_cs_w_1 = AE_MULFP16X4RS(d_cs_r_1, d_fg_1);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      d_1mfg_1 = AE_SUB16S(d_one, d_fg_1);
      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_1mfg_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_1mfg_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16* p16_cg_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];

      d_cs_w_0 = AE_MULFP16X4RS(d_cs_r_0, d_fg_0);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_1mfg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  } else {
    for (i = 0; i < (num_elms >> 3); i++) {
      AE_LA16X4X2_IP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r);
      AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
      AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_mul_1 = AE_SRAA32SYMS(d_mul_1, shift1);
      d_mul_2 = AE_SRAA32SYMS(d_mul_2, shift1);
      d_mul_3 = AE_SRAA32SYMS(d_mul_3, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
      d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      d_1mfg_1 = AE_SUB16S(d_one, d_fg_1);
      AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_1mfg_0);
      AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_1mfg_1);
      d_mul_4 = AE_SRAA32SYMS(d_mul_4, shift2);
      d_mul_5 = AE_SRAA32SYMS(d_mul_5, shift2);
      d_mul_6 = AE_SRAA32SYMS(d_mul_6, shift2);
      d_mul_7 = AE_SRAA32SYMS(d_mul_7, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
      d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      AE_MINMAX16(d_cs_w_1, d_min, d_max);

      AE_SA16X4X2_IP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w);
    }
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);  // finalize the stream

    const ae_int16 *p16_cs_r, *p16_fg_r;
    const ae_int16* p16_cg_r;

    ae_int16* p16_cs_w;

    p16_cs_r = (const ae_int16*)p16x8_cs_r;
    p16_fg_r = (const ae_int16*)p16x8_fg_r;
    p16_cg_r = (const ae_int16*)p16x8_cg_r;

    p16_cs_w = (ae_int16*)p16x8_cs_w;
// residue iterations
#pragma concurrent
#pragma loop_count max = 7
    for (i = 0; i < ((num_elms)&7); i++) {
      d_cs_r_0 = p16_cs_r[i];
      d_fg_0 = p16_fg_r[i];
      d_cg_0 = p16_cg_r[i];

      AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift1);
      d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_1mfg_0 = AE_SUB16S(d_one, d_fg_0);
      AE_MUL16X4(d_mul_0, d_mul_1, d_cg_0, d_1mfg_0);
      d_mul_0 = AE_SRAA32SYMS(d_mul_0, shift2);
      d_cg_0 = AE_SAT16X4(d_mul_0, d_mul_1);

      d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
      AE_MINMAX16(d_cs_w_0, d_min, d_max);
      p16_cs_w[i] = d_cs_w_0;
    }
  }
}

void xa_nn_elm_mul_16x16_asym8s(int8_t* output, const int16_t* input_1,
                                const int16_t* input_2, int32_t multiplier,
                                int32_t shift, int32_t zero_point,
                                int num_elms) {
  ae_int16x8* tmp_input_1;
  ae_int16x8* tmp_input_2;

  ae_valignx2 align_src_input_1, align_src_input_2;
  ae_valign align_dst_output;

  ae_int16x4 data_a_0, data_a_1;
  ae_int16x4 data_b_0, data_b_1;
  ae_int32x2 data_ab_0, data_ab_1, data_ab_2, data_ab_3;
  ae_int32x2 d_multiplier, d_left_shift;
  ae_int16x4 d_zp;
  ae_int16x4 data_c_0, data_c_1;
  ae_int8x8 data_c;

  int i = 0;
  int left_shift, right_shift;
  tmp_input_1 = (ae_int16x8*)(input_1);
  tmp_input_2 = (ae_int16x8*)(input_2);

  align_src_input_1 = AE_LA128_PP((ae_int16x8*)tmp_input_1);
  align_src_input_2 = AE_LA128_PP((ae_int16x8*)tmp_input_2);
  align_dst_output = AE_ZALIGN64();  // zero alignment reg

  d_multiplier = AE_MOVDA32(multiplier);
  d_zp = AE_MOVDA16(zero_point);

  left_shift = shift < 0 ? 0 : shift;
  right_shift = shift > 0 ? 0 : -shift;

  d_left_shift = AE_MOVDA32(1 << left_shift);
#pragma concurrent
  for (i = 0; i < (num_elms >> 3); i++) {
    AE_LA16X4X2_IP(data_a_0, data_a_1, align_src_input_1, tmp_input_1);
    AE_LA16X4X2_IP(data_b_0, data_b_1, align_src_input_2, tmp_input_2);

    AE_MUL16X4(data_ab_0, data_ab_1, data_a_0, data_b_0);
    AE_MUL16X4(data_ab_2, data_ab_3, data_a_1, data_b_1);
    AE_MUL2P32X4(data_ab_0, data_ab_1, data_ab_0, data_ab_1, d_left_shift,
                 d_left_shift);
    AE_MUL2P32X4(data_ab_2, data_ab_3, data_ab_2, data_ab_3, d_left_shift,
                 d_left_shift);
    AE_MULF2P32X4RAS(data_ab_0, data_ab_1, data_ab_0, data_ab_1, d_multiplier,
                     d_multiplier);
    AE_MULF2P32X4RAS(data_ab_2, data_ab_3, data_ab_2, data_ab_3, d_multiplier,
                     d_multiplier);
    data_ab_0 = AE_SRAA32SYMS(data_ab_0, right_shift);
    data_ab_1 = AE_SRAA32SYMS(data_ab_1, right_shift);
    data_ab_2 = AE_SRAA32SYMS(data_ab_2, right_shift);
    data_ab_3 = AE_SRAA32SYMS(data_ab_3, right_shift);
    data_c_0 = AE_SAT16X4(data_ab_0, data_ab_1);
    data_c_1 = AE_SAT16X4(data_ab_2, data_ab_3);
    data_c_0 = AE_SUB16S(data_c_0, d_zp);
    data_c_1 = AE_SUB16S(data_c_1, d_zp);
    data_c = AE_SAT8X8X16(data_c_0, data_c_1);
    AE_SA8X8_IP(data_c, align_dst_output, (ae_int8x8*)output);
  }

  AE_SA64POS_FP(align_dst_output, output);  // finalize the stream

// residue iterations
#pragma concurrent
#pragma loop_count max = 7
  for (int j = 0; j < ((num_elms)&7); j++) {
    AE_L16_IP(data_a_0, (ae_int16*)tmp_input_1, 2);
    AE_L16_IP(data_b_0, (ae_int16*)tmp_input_2, 2);

    AE_MUL16X4(data_ab_0, data_ab_1, data_a_0, data_b_0);
    data_ab_0 = AE_MULP32X2(data_ab_0, d_left_shift);
    data_ab_0 = AE_MULFP32X2RAS(data_ab_0, d_multiplier);
    data_ab_0 = AE_SRAA32SYMS(data_ab_0, right_shift);
    data_c_0 = AE_SAT16X4(data_ab_0, data_ab_1);
    data_c_0 = AE_SUB16S(data_c_0, d_zp);
    data_c = AE_SAT8X8X16(data_c_0, data_c_0);
    AE_S8_0_IP(data_c, (ae_int8*)output, 1);
  }
}
#endif  // defined(HIFI5)

}  // namespace lstm_eval
}  // namespace micro
}  // namespace ops
}  // namespace tflite
